#!/usr/bin/env python3
import rospy
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

def modify_pcd(q):
    while 1:
        dt_now_str = q.get()
        points = q.get()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float32))
        voxel_pcd = pcd.voxel_down_sample(voxel_size=0.01)
        o3d.io.write_point_cloud("/work_space/lidar_data/voxel_data/" + dt_now_str + ".pcd", voxel_pcd)
        print("save voxeld pcd data")

def callback(point_cloud, q):
    dt_now = datetime.datetime.now()
    dt_now_str = dt_now.strftime('%Y_%m_%d_%H_%M_%S')
    pc = ros_numpy.numpify(point_cloud)
    points=np.zeros((pc.shape[0],3))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float32))
    o3d.io.write_point_cloud("/work_space/lidar_data/data/" + dt_now_str + ".pcd", pcd)
    print("save pcd data")
    q.put(dt_now_str)
    q.put(points)

def connect_ros(q):
    rospy.init_node('lidar_subscriber', anonymous=True)
    rospy.Subscriber("/livox/lidar", PointCloud2, callback, callback_args=q)
    rospy.spin()

def main():
    set_start_method('fork')
    q = Queue()

    p_connect_ros = Process(target=connect_ros, args=(q,))
    p_connect_ros.start()

    p_modify_pcd = Process(target=modify_pcd, args=(q,))
    p_modify_pcd.start()


if __name__ == '__main__':
    main()
    
    