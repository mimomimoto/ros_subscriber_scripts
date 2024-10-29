#!/usr/bin/env python3
import rospy
import pcl
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import ros_numpy
import datetime
import torch.multiprocessing as mp
import time
import open3d.core as o3c
import os, glob
import json
from np_socket import SocketNumpyArray
import struct
import pickle
import csv
import os
import sys
import pandas as pd
from scipy.spatial import cKDTree
import iperf3
import copy

# lab
ROUGH_DBSCAN_EPS = 0.1
ROUGH_DBSCAN_MIN_POINTS = 10
CLUSTER_MIN_HEIGHT_THERESH = -0.6
CLUSTER_MAX_HEIGHT_THERESH = 1
CLUSTER_POINTS_THERESH = 60
CLUSTER_BOX_SIZE_THERESH = 2
STRICT_DBSCAN_EPS = 0.2
STRICT_DBSCAN_MIN_POINTS_DIV = 10

# splab
# ROUGH_DBSCAN_EPS = 0.1
# ROUGH_DBSCAN_MIN_POINTS = 10
# CLUSTER_MIN_HEIGHT_THERESH = -3
# CLUSTER_MAX_HEIGHT_THERESH = 0.5
# CLUSTER_POINTS_THERESH = 200
# CLUSTER_BOX_SIZE_THERESH = 2
# STRICT_DBSCAN_EPS = 0.2
# STRICT_DBSCAN_MIN_POINTS_DIV = 10



# ROUGH_DBSCAN_EPS = 0.2
# ROUGH_DBSCAN_MIN_POINTS = 30
# DETAIL_DBSCAN_EPS = 0.15
# CLUSTER_HEIGHT_THERESH = -1.0
# CLUSTER_POINTS_THERESH = 100
# DOUBLE_CLUSTERING = 15

SOURCE_PCD = o3d.io.read_point_cloud("/work_space/lidar_data/base/base.pcd")

SERVER_FLAG = False

def put_dummy_on_cuda():
    ut = time.time()
    device = o3d.core.Device("CUDA:0")
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3d.core.Tensor(np.empty((1, 3)), dtype, device)
    print(pcd.is_cuda)
    print("************************************************")
    print("put_dummy_on_cuda: ", time.time() - ut)
    print("************************************************")

def divide_cluster(pcd, pcd_arrays, thr_points_num):
    global index1
    device = o3d.core.Device("CUDA:0")
    cpu_device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float64
    x_y_df = pd.DataFrame(pcd.points,
                  columns = ["x","y","z"],)
    x_y_df["z"] = 0
    
    z_df = pd.DataFrame(pcd.points,
                  columns = ["x","y","z"],)
    
    x_y_pcd = o3d.geometry.PointCloud()
    x_y_pcd.points = o3d.utility.Vector3dVector(x_y_df.to_numpy())
    thresh_min_points = int(len(x_y_df.index)/STRICT_DBSCAN_MIN_POINTS_DIV)
    
    tmp_x_y_pcd = o3d.t.geometry.PointCloud(device)
    tmp_x_y_pcd.point.positions = o3d.core.Tensor(np.asarray(x_y_pcd.points), dtype, device)
    
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        labels_tensor = tmp_x_y_pcd.cluster_dbscan(eps=STRICT_DBSCAN_EPS, min_points=thresh_min_points, print_progress=False)
        labels_tensor_cpu = labels_tensor.to(cpu_device)
        labels = np.array(labels_tensor_cpu.cpu().numpy())
        
    tmp_cluster_pcd = o3d.geometry.PointCloud()
    
    min_cluster_points = 0

    if labels.size != 0: 
        for i in range(labels.max() + 1):
            pc_indices = np.where(labels == i)[0]
            xyz = np.asarray(z_df.to_numpy())[pc_indices, :]
            if CLUSTER_MAX_HEIGHT_THERESH > xyz.T[2].max() > CLUSTER_MIN_HEIGHT_THERESH:
                if len(xyz) >= CLUSTER_POINTS_THERESH:
                    tmp_pcd = o3d.geometry.PointCloud()
                    tmp_pcd.points = o3d.utility.Vector3dVector(xyz)
                    
                    plane_xyz = np.asarray(x_y_df.to_numpy())[pc_indices, :]
                    plane_pcd = o3d.geometry.PointCloud()
                    plane_pcd.points = o3d.utility.Vector3dVector(plane_xyz)
                    if min_cluster_points == 0:
                        min_cluster_points = len(tmp_pcd.points)
                    elif min_cluster_points > len(tmp_pcd.points):
                        min_cluster_points = len(tmp_pcd.points)
                    pcd_arrays.append(xyz)
                    
                    tmp_cluster_pcd += tmp_pcd
        
        dists = pcd.compute_point_cloud_distance(tmp_cluster_pcd)
        dists = np.asarray(dists)
        ind = np.where(dists > 0.03 )[0]
        noise_pcd = pcd.select_by_index(ind)
        if len(noise_pcd.points) >= min_cluster_points:
            pcd_arrays = divide_cluster(noise_pcd, pcd_arrays, min_cluster_points)
    return pcd_arrays

def cluster(pcd_numpy):
    ut = time.time()
    cpu_device = o3d.core.Device("CPU:0")
    device = o3d.core.Device("CUDA:0")
    dtype = o3d.core.float64

    tmp_target = o3d.t.geometry.PointCloud()
    tmp_target.point.positions = o3d.core.Tensor(pcd_numpy, dtype, device)
    tmp_target_numpy = tmp_target.point.positions.cpu().numpy().copy()
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(tmp_target_numpy)
    dists = target.compute_point_cloud_distance(SOURCE_PCD)
    dists = np.asarray(dists)
    ind = np.where(dists > 0.1)[0]
    tmp_object_pcd = target.select_by_index(ind)
    cl, ind = tmp_object_pcd.remove_radius_outlier(nb_points=10, radius=0.5)
    tmp_object_pcd = tmp_object_pcd.select_by_index(ind)
    object_pcd = o3d.t.geometry.PointCloud(device)
    object_pcd.point.positions = o3d.core.Tensor(np.asarray(tmp_object_pcd.points), dtype, device)
    
    back_substruction_time = time.time() - ut
    
    ut = time.time()

    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        labels_tensor = object_pcd.cluster_dbscan(eps=ROUGH_DBSCAN_EPS, min_points=ROUGH_DBSCAN_MIN_POINTS, print_progress=False)
        labels_tensor_cpu = labels_tensor.to(cpu_device)
        labels = np.array(labels_tensor_cpu.cpu().numpy())


    cluster_pcd = o3d.geometry.PointCloud()
    
    pcd_arrays = []
    
    if labels.size != 0: 
        for i in range(labels.max() + 1):
            pc_indices = np.where(labels == i)[0]

            if pc_indices.size > 0:
                xyz = np.asarray(tmp_object_pcd.points)[pc_indices, :]
                
                if CLUSTER_MAX_HEIGHT_THERESH > xyz.T[2].max() > CLUSTER_MIN_HEIGHT_THERESH:
                    if len(xyz) >= CLUSTER_POINTS_THERESH:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(xyz)
                        bounding_box = pcd.get_oriented_bounding_box()
                        size_bounding_box = bounding_box.get_max_bound() - bounding_box.get_min_bound()
                        ts_size = size_bounding_box[0] * size_bounding_box[1]
                        if ts_size >= CLUSTER_BOX_SIZE_THERESH:
                            pcd_arrays = divide_cluster(pcd, pcd_arrays, 0)
                        else:
                            pcd_arrays.append(xyz)
                        cluster_pcd += pcd
    
    cluster_time = time.time() - ut
    
    # o3d.io.write_point_cloud(f"/work_space/lidar_data/eva/reid/5/{datetime.datetime.now() + datetime.timedelta(hours=9)}.pcd", cluster_pcd)

    return pcd_arrays, back_substruction_time, cluster_time, len(pcd_arrays)

def callback(point_cloud, args):
    ut = time.time()
    code = args[0]
    q = args[1]
    
    # デバイスの設定
    device = o3d.core.Device("CUDA:0")
    dtype = o3d.core.float32
    
    # 回転行列を含むJSONファイルの読み取り
    config = {}
    with open("/work_space/lidar_data/matrix_config/matrix_config.json", mode="r") as f:
            config = json.load(f)
    
    # 計測時の時間を取得
    dt_now = datetime.datetime.now()
    dt_now_str = dt_now.strftime('%Y_%m_%d_%H_%M_%S')
    
    # ポイントクラウドオブジュエクトからNumpy配列に変換
    pc = ros_numpy.numpify(point_cloud)
    points=np.zeros((pc.shape[0],3))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']
    
    # GPUのメモリにPCDデータを乗せる
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3d.core.Tensor(points, dtype, device)
    # ボクセル化(1cm)
    voxel_pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # 点群データの回転
    voxel_pcd_rotated = voxel_pcd.transform(np.array(config[code]))
    
    if q.full():
        # print('*******************************************************************')
        # print('*******************************************************************')
        print('full error')
        # print('*******************************************************************')
        # print('*******************************************************************')
    
    if q.qsize() >= 1:
        # print('clear que')
        q.get()
    q.put(voxel_pcd_rotated)
    
    print("************************************************")
    print(code, " que: ", time.time() - ut, time.time())
    print("************************************************")
    
    
    
    for file in glob.glob("/work_space/lidar_data/" + code + "/*.pcd", recursive=True):
        os.remove(file)

    o3d.t.io.write_point_cloud("/work_space/lidar_data/" + code + "/" + dt_now_str + ".pcd", pcd)
    # print("save " + code + " data")

    
def connect_ros(q_3JEDKBS001G9601, q_3JEDKC50014U011, q_3JEDL3N0015X621, q_3JEDL76001L4201):
    put_dummy_on_cuda()
    rospy.init_node('lidar_subscriber', anonymous=True)
    rospy.Subscriber("/livox/lidar_3JEDKBS001G9601", PointCloud2, callback, ("3JEDKBS001G9601", q_3JEDKBS001G9601))
    rospy.Subscriber("/livox/lidar_3JEDKC50014U011", PointCloud2, callback, ("3JEDKC50014U011", q_3JEDKC50014U011))
    rospy.Subscriber("/livox/lidar_3JEDL3N0015X621", PointCloud2, callback, ("3JEDL3N0015X621", q_3JEDL3N0015X621))
    rospy.Subscriber("/livox/lidar_3JEDL76001L4201", PointCloud2, callback, ("3JEDL76001L4201", q_3JEDL76001L4201))
    rospy.spin()
    
def combine_pcd(q_3JEDKBS001G9601, q_3JEDKC50014U011, q_3JEDL3N0015X621, q_3JEDL76001L4201, q_band_width, q_server_back_substruction_time, q_server_cluster_time, q_server_after_cluster_size):
    global REBOOT_FLAG
    global SERVER_FLAG
    sock_sender = SocketNumpyArray()
    sock_sender.initialize_sender('192.168.50.30', 49220)
    send_data_time =  time.time()
    band_with = 0
    server_back_substruction_time = 0
    server_cluster_time = 0
    server_cluster_size = 0
    
    while 1:
        # try:
        if q_band_width.qsize() >= 1:
            band_with = q_band_width.get()
        if q_server_back_substruction_time.qsize() >= 1:
            server_back_substruction_time = q_server_back_substruction_time.get()
        if q_server_cluster_time.qsize() >= 1:
            server_cluster_time = q_server_cluster_time.get()
        if q_server_after_cluster_size.qsize() >= 1:
            server_cluster_size = q_server_after_cluster_size.get()
        
        print('band_with: ', band_with)
            
        pcd_3JEDKBS001G9601 = q_3JEDKBS001G9601.get()

        pcd_3JEDKC50014U011 = q_3JEDKC50014U011.get()

        pcd_3JEDL3N0015X621 = q_3JEDL3N0015X621.get()
        
        pcd_3JEDL76001L4201 = q_3JEDL76001L4201.get()
        
        ut = time.time()


        combined_pcd = pcd_3JEDKBS001G9601 + pcd_3JEDKC50014U011 + pcd_3JEDL3N0015X621 + pcd_3JEDL76001L4201
        
        combined_voxel_pcd = combined_pcd.voxel_down_sample(voxel_size=0.05)

        now_dt = datetime.datetime.now()
        if 18 == now_dt.hour and 5 > now_dt.minute and now_dt.minute > 0:
            o3d.t.io.write_point_cloud("/work_space/lidar_data/base/base.pcd", combined_voxel_pcd)
            
        # bb = o3d.t.geometry.AxisAlignedBoundingBox(
        #     np.array([[0],[-6],[-100]]),
        #     np.array([[7],[1],[10]]),
        # )
        
        combined_voxel_numpy = combined_voxel_pcd.point.positions.cpu().numpy().copy()
        before_data = pickle.dumps(combined_voxel_numpy)
        before_data_len = len(before_data)/1024/1024


        # combined_voxel_pcd = combined_voxel_pcd.crop(bb)
        if SERVER_FLAG == True:
            be_pack_data = time.time()
            flag = struct.pack("I", 1)
            message_size = struct.pack("I", len(before_data))
            send_time = struct.pack("d", time.time())
            print('time: ', time.time())
            sock_sender.socket.sendall(flag + message_size + send_time + before_data)
            print('send data: ', time.time() - be_pack_data)
            print('left_time: ', (server_back_substruction_time + server_cluster_time))
            print('right_time: ', (before_data_len + server_cluster_size)*8/band_with)
            
            if band_with != 0 and server_cluster_size != 0 and server_back_substruction_time != 0 and server_cluster_time != 0:
                if (server_back_substruction_time + server_cluster_time) < (before_data_len + server_cluster_size)*8/band_with:
                    SERVER_FLAG = False
            with open('/work_space/lidar_data/process_change.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.datetime.now() + datetime.timedelta(hours=9), band_with, server_back_substruction_time, server_cluster_time, 1])
            

        else:
            combined_voxel_numpy, back_substruction_time, cluster_time, cluster_size = cluster(combined_voxel_numpy)
        
            # with open('/work_space/lidar_data/edge_process.csv', 'a') as f:
            #     writer = csv.writer(f)
            #     writer.writerow([datetime.datetime.now() + datetime.timedelta(hours=9), cluster_size, combined_data_time, back_substruction_time, cluster_time])
            
            be_pack_data = time.time()

            data = pickle.dumps(combined_voxel_numpy)
            after_data_len = len(data)/1024/1024
            
            flag = struct.pack("I", 0)
            message_size = struct.pack("I", len(data))
            send_time = struct.pack("d", time.time())
            print('time: ', time.time())
            sock_sender.socket.sendall(flag + message_size + send_time + data)
            
            print('send data: ', time.time() - be_pack_data)
            
            if band_with != 0:
                print('left_time: ', (back_substruction_time + cluster_time))
                print('right_time: ', (before_data_len + after_data_len)*8/band_with)
                if (back_substruction_time + cluster_time)/2 > (before_data_len + after_data_len)*8/band_with:
                    SERVER_FLAG = True
            with open('/work_space/lidar_data/process_change.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.datetime.now() + datetime.timedelta(hours=9), band_with, back_substruction_time, cluster_time, 0])

        send_data_time = time.time()
        
        # print("************************************************")
        # print("send data: ", time.time() - ut)
        # print("************************************************")
        
        
        dt_now = datetime.datetime.now()
        dt_now_str = dt_now.strftime('%Y_%m_%d_%H_%M_%S')

        # o3d.t.io.write_point_cloud(f"/work_space/lidar_data/eva/5/{datetime.datetime.now() + datetime.timedelta(hours=9)}.pcd", combined_voxel_pcd)
        o3d.t.io.write_point_cloud("/work_space/lidar_data/combined_pcd/combined.pcd", combined_voxel_pcd)
        print("save combined data")
            
        # except:
            
        #     continue

def iperf_client(q_band_width):
    while 1:
        client = iperf3.Client()
        client.duration = 5
        client.server_hostname = '192.168.50.30'
        client.port = 5201
        client.bytes = 10 * 1024 * 1024
        result = client.run()
        q_band_width.put(result.sent_Mbps)
        # print('sent_Mbps: ', result.sent_Mbps)
        client = None

        time.sleep(5)
        
def share_processing_time_server(q_server_back_substruction_time, q_server_cluster_time, q_server_after_cluster_size):
    sock_receiver = SocketNumpyArray()
    sock_receiver.address = '192.168.107.50'
    sock_receiver.port = 49227
    sock_receiver.socket.bind((sock_receiver.address, sock_receiver.port))
    print('Socket bind complete')
    sock_receiver.socket.listen(10)
    while True:
        sock_cl = copy.copy(sock_receiver)
        sock_cl.conn, addr = sock_receiver.socket.accept()

        print('Socket now listening')
        sock_cl.payload_size = struct.calcsize("I")  # CHANGED
        sock_cl.data = b''
        while True:
        
            while len(sock_cl.data) < 4:
                sock_cl.data += sock_cl.conn.recv(4096)

            packed_msg_size = sock_cl.data[:4]
            sock_cl.data = sock_cl.data[4:]
            msg_size = struct.unpack("I", packed_msg_size)[0]

            while len(sock_cl.data) < msg_size:
                sock_cl.data += sock_cl.conn.recv(4096)

            frame_data = sock_cl.data[:msg_size]
            sock_cl.data = sock_cl.data[msg_size:]
            
            

            server_processing_time = pickle.loads(frame_data)
            
            print('server_processing_time: ', server_processing_time)
            
            q_server_back_substruction_time.put(server_processing_time[0])
            q_server_cluster_time.put(server_processing_time[1])
            q_server_after_cluster_size.put(server_processing_time[2])

        

def main():
    if mp.get_start_method() == 'fork':
        mp.set_start_method('spawn', force=True)
        
    manager = mp.Manager()
        
    q_3JEDKBS001G9601 = manager.Queue()
    q_3JEDKC50014U011 = manager.Queue()
    q_3JEDL3N0015X621 = manager.Queue()
    q_3JEDL76001L4201 = manager.Queue()
    
    q_server_back_substruction_time = mp.Queue()
    q_server_cluster_time = mp.Queue()
    q_server_after_cluster_size = mp.Queue()
    q_band_width = mp.Queue()
    
    

    p_connect_ros = mp.Process(target=connect_ros, args=(q_3JEDKBS001G9601, q_3JEDKC50014U011, q_3JEDL3N0015X621, q_3JEDL76001L4201))
    p_connect_ros.start()
    
    p_combine_pcd = mp.Process(target=combine_pcd, args=(q_3JEDKBS001G9601, q_3JEDKC50014U011, q_3JEDL3N0015X621, q_3JEDL76001L4201, q_band_width, q_server_back_substruction_time, q_server_cluster_time, q_server_after_cluster_size))
    p_combine_pcd.start()
    
    p_iperf_client = mp.Process(target=iperf_client,args=(q_band_width,))
    p_iperf_client.start()
    
    p_share_processing_time_server = mp.Process(target=share_processing_time_server, args=(q_server_back_substruction_time, q_server_cluster_time, q_server_after_cluster_size))
    p_share_processing_time_server.start()
    
    p_connect_ros.join()
    p_combine_pcd.join()
    p_iperf_client.join()
    p_share_processing_time_server.join()
    


if __name__ == '__main__':
    main()
    
     
