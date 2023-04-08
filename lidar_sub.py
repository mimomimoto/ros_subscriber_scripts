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

def callback(point_cloud):
    dt_now = datetime.datetime.now()
    dt_now_str = dt_now.strftime('%Y_%m_%d_%H_%M_%S')
    pc = ros_numpy.numpify(point_cloud)
    points=np.zeros((pc.shape[0],3))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float32))
    o3d.io.write_point_cloud("/work_space/lidar_data/" + dt_now_str + ".pcd", pcd)
    print(1)

if __name__ == '__main__':
    rospy.init_node('lidar_subscriber', anonymous=True)
    rospy.Subscriber("/livox/lidar", PointCloud2, callback)
    rospy.spin()