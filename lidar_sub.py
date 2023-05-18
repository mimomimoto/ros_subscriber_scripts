#!/usr/bin/env python3
import rospy
import pcl
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import ros_numpy
import datetime
from multiprocessing import set_start_method
from multiprocessing import Value, Array, Process, Queue
import multiprocessing
import time
import open3d.core as o3c
import os, glob
import json

def put_dummy_on_cuda():
    ut = time.time()
    device = o3d.core.Device("CUDA:0")
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3d.core.Tensor(np.empty((1, 3)), dtype, device)
    print("************************************************")
    print("put_dummy_on_cuda: ", time.time() - ut)
    print("************************************************")
    
    


def callback(point_cloud, code):
    ut = time.time()
    
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
    print(code, ": ", time.time() - ut)
    print("************************************************")
    
    
    for file in glob.glob("/work_space/lidar_data/" + code + "/*.pcd", recursive=True):
        os.remove(file)

    o3d.t.io.write_point_cloud("/work_space/lidar_data/" + code + "/" + dt_now_str + ".pcd", voxel_pcd_rotated)
    print("save " + code + " data")


def connect_ros():
    rospy.init_node('lidar_subscriber', anonymous=True)
    rospy.Subscriber("/livox/lidar_3JEDKBS001G9601", PointCloud2, callback, callback_args="3JEDKBS001G9601")
    rospy.Subscriber("/livox/lidar_3JEDKC50014U011", PointCloud2, callback, callback_args="3JEDKC50014U011")
    rospy.Subscriber("/livox/lidar_3JEDL3N0015X621", PointCloud2, callback, callback_args="3JEDL3N0015X621")
    rospy.spin()

def main():
    put_dummy_on_cuda()
    # connect_ros()
    set_start_method('fork')
    # q = Queue()

    p_connect_ros = Process(target=connect_ros())
    p_connect_ros.start()

    # p_modify_pcd = Process(target=modify_pcd, args=(q,))
    # p_modify_pcd.start()


if __name__ == '__main__':
    main()
    
    