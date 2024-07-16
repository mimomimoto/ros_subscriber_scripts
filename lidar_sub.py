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


ROUGH_DBSCAN_EPS = 0.2
ROUGH_DBSCAN_MIN_POINTS = 20
CLUSTER_HEIGHT_THERESH = -0.6
CLUSTER_POINTS_THERESH = 60
DOUBLE_CLUSTERING = 10

# ROUGH_DBSCAN_EPS = 0.2
# ROUGH_DBSCAN_MIN_POINTS = 30
# DETAIL_DBSCAN_EPS = 0.15
# CLUSTER_HEIGHT_THERESH = -1.0
# CLUSTER_POINTS_THERESH = 100
# DOUBLE_CLUSTERING = 15

SOURCE_PCD = o3d.io.read_point_cloud("/work_space/lidar_data/base/base.pcd")


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
    thresh_min_points = int(len(x_y_df.index)/DOUBLE_CLUSTERING)
    
    tmp_x_y_pcd = o3d.t.geometry.PointCloud(device)
    tmp_x_y_pcd.point.positions = o3d.core.Tensor(np.asarray(x_y_pcd.points), dtype, device)
    
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        labels_tensor = tmp_x_y_pcd.cluster_dbscan(eps=ROUGH_DBSCAN_EPS, min_points=thresh_min_points, print_progress=False)
        labels_tensor_cpu = labels_tensor.to(cpu_device)
        labels = np.array(labels_tensor_cpu.cpu().numpy())
        
    tmp_cluster_pcd = o3d.geometry.PointCloud()
    
    min_cluster_points = 0

    if labels.size != 0: 
        for i in range(labels.max() + 1):
            pc_indices = np.where(labels == i)[0]
            xyz = np.asarray(z_df.to_numpy())[pc_indices, :]
            if xyz.T[2].max() > CLUSTER_HEIGHT_THERESH:
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
                
                if xyz.T[2].max() > CLUSTER_HEIGHT_THERESH:
                    if len(xyz) >= CLUSTER_POINTS_THERESH:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(xyz)
                        bounding_box = pcd.get_oriented_bounding_box()
                        size_bounding_box = bounding_box.get_max_bound() - bounding_box.get_min_bound()
                        ts_size = size_bounding_box[0] * size_bounding_box[1]
                        if ts_size >= 2:
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
        print('*******************************************************************')
        print('*******************************************************************')
        print('full error')
        print('*******************************************************************')
        print('*******************************************************************')
    
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
    
def combine_pcd(q_3JEDKBS001G9601, q_3JEDKC50014U011, q_3JEDL3N0015X621, q_3JEDL76001L4201):
    global REBOOT_FLAG
    sock_sender = SocketNumpyArray()
    sock_sender.initialize_sender('192.168.50.30', 49220)
    send_data_time =  time.time()
    while 1:
        # try:
            
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

        combined_voxel_numpy = combined_voxel_pcd.point.positions.cpu().numpy().copy()
        
        combined_data_time = time.time() - ut
        
        
        combined_voxel_numpy, back_substruction_time, cluster_time, cluster_size = cluster(combined_voxel_numpy)
        
        # with open('/work_space/lidar_data/edge_process.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([datetime.datetime.now() + datetime.timedelta(hours=9), cluster_size, combined_data_time, back_substruction_time, cluster_time])

        data = pickle.dumps(combined_voxel_numpy)
        
        
        
        message_size = struct.pack("I", len(data))
        sock_sender.socket.sendall(message_size + data)
        send_data_time = time.time()
        print("************************************************")
        print("send data: ", time.time() - ut)
        print("************************************************")
        
        
        dt_now = datetime.datetime.now()
        dt_now_str = dt_now.strftime('%Y_%m_%d_%H_%M_%S')

        # o3d.t.io.write_point_cloud(f"/work_space/lidar_data/eva/5/{datetime.datetime.now() + datetime.timedelta(hours=9)}.pcd", combined_voxel_pcd)
        o3d.t.io.write_point_cloud("/work_space/lidar_data/combined_pcd/combined.pcd", combined_voxel_pcd)
        print("save combined data")
        # except:
            
        #     continue
        
        

def main():
    if mp.get_start_method() == 'fork':
        mp.set_start_method('spawn', force=True)
        
    manager = mp.Manager()
        
    q_3JEDKBS001G9601 = manager.Queue()
    q_3JEDKC50014U011 = manager.Queue()
    q_3JEDL3N0015X621 = manager.Queue()
    q_3JEDL76001L4201 = manager.Queue()

    p_connect_ros = mp.Process(target=connect_ros, args=(q_3JEDKBS001G9601, q_3JEDKC50014U011, q_3JEDL3N0015X621, q_3JEDL76001L4201))
    p_connect_ros.start()
    
    p_combine_pcd = mp.Process(target=combine_pcd, args=(q_3JEDKBS001G9601, q_3JEDKC50014U011, q_3JEDL3N0015X621, q_3JEDL76001L4201))
    p_combine_pcd.start()
    
    p_connect_ros.join()
    p_combine_pcd.join()


if __name__ == '__main__':
    main()
    
     