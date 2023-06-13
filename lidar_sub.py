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

def put_dummy_on_cuda():
    ut = time.time()
    device = o3d.core.Device("CUDA:0")
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3d.core.Tensor(np.empty((1, 3)), dtype, device)
    print("************************************************")
    print("put_dummy_on_cuda: ", time.time() - ut)
    print("************************************************")

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
    voxel_pcd = pcd.voxel_down_sample(voxel_size=0.01)
    
    # 点群データの回転
    voxel_pcd_rotated = voxel_pcd.transform(np.array(config[code]))
    print("************************************************")
    print(pcd.point)
    print(code, ": ", time.time() - ut, time.time())
    print("************************************************")
    
    q.put(voxel_pcd_rotated)
    
    
    
    for file in glob.glob("/work_space/lidar_data/" + code + "/*.pcd", recursive=True):
        os.remove(file)

    o3d.t.io.write_point_cloud("/work_space/lidar_data/" + code + "/" + dt_now_str + ".pcd", pcd)
    print("save " + code + " data")

    
def connect_ros(q_3JEDKBS001G9601, q_3JEDKC50014U011, q_3JEDL3N0015X621):
    put_dummy_on_cuda()
    rospy.init_node('lidar_subscriber', anonymous=True)
    rospy.Subscriber("/livox/lidar_3JEDKBS001G9601", PointCloud2, callback, ("3JEDKBS001G9601", q_3JEDKBS001G9601))
    rospy.Subscriber("/livox/lidar_3JEDKC50014U011", PointCloud2, callback, ("3JEDKC50014U011", q_3JEDKC50014U011))
    rospy.Subscriber("/livox/lidar_3JEDL3N0015X621", PointCloud2, callback, ("3JEDL3N0015X621", q_3JEDL3N0015X621))
    rospy.spin()
    
def combine_pcd(q_3JEDKBS001G9601, q_3JEDKC50014U011, q_3JEDL3N0015X621):
    sock_sender = SocketNumpyArray()
    sock_sender.initialize_sender('192.168.30.10', 49220)
    while 1:
        # try:
        ut = time.time()
        
        pcd_3JEDKBS001G9601 = q_3JEDKBS001G9601.get()

        pcd_3JEDKC50014U011 = q_3JEDKC50014U011.get()

        pcd_3JEDL3N0015X621 = q_3JEDL3N0015X621.get()


        combined_pcd = pcd_3JEDKBS001G9601 + pcd_3JEDKC50014U011 + pcd_3JEDL3N0015X621
        
        combined_voxel_pcd = combined_pcd.voxel_down_sample(voxel_size=0.01)
        
        print("************************************************")
        print("combined data: ", time.time() - ut, time.time())
        print("************************************************")

        combined_voxel_numpy = combined_voxel_pcd.point.positions.cpu().numpy().copy()
        print("time:", time.time())
        data = pickle.dumps(combined_voxel_numpy)
        
        message_size = struct.pack("I", len(data))
        sock_sender.socket.sendall(message_size + data)
        
        
        dt_now = datetime.datetime.now()
        dt_now_str = dt_now.strftime('%Y_%m_%d_%H_%M_%S')
        for file in glob.glob("/work_space/lidar_data/combined_pcd" + "/*.pcd", recursive=True):
            os.remove(file)

        o3d.t.io.write_point_cloud("/work_space/lidar_data/combined_pcd/" + dt_now_str + ".pcd", combined_voxel_pcd)
        print("save combined data")
        # except:
        #     sock_sender = SocketNumpyArray()
        #     sock_sender.initialize_sender('192.168.30.10', 49220)
        
        

def main():
    if mp.get_start_method() == 'fork':
        mp.set_start_method('spawn', force=True)
        
    manager = mp.Manager()
        
    q_3JEDKBS001G9601 = manager.Queue()
    q_3JEDKC50014U011 = manager.Queue()
    q_3JEDL3N0015X621 = manager.Queue()

    p_connect_ros = mp.Process(target=connect_ros, args=(q_3JEDKBS001G9601, q_3JEDKC50014U011, q_3JEDL3N0015X621,))
    p_connect_ros.start()
    
    p_combine_pcd = mp.Process(target=combine_pcd, args=(q_3JEDKBS001G9601, q_3JEDKC50014U011, q_3JEDL3N0015X621,))
    p_combine_pcd.start()
    
    p_connect_ros.join()
    p_combine_pcd.join()


if __name__ == '__main__':
    main()
    
    