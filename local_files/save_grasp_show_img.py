import os
import sys
import numpy as np
import cv2
import shutil
import argparse
from PIL import Image
import time
import scipy.io as scio
import torch
from tqdm import tqdm
import trimesh as tm
import open3d as o3d
import matplotlib.pyplot as plt
from graspnetAPI.graspnet_eval import GraspGroup
from utils_data import get_align_dex_hand_mesh


def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=1280, height=720)
    opt = vis.get_render_option()  # 可视化参数设置
    opt.background_color = np.asarray([0, 0, 0])  # 设置背景色
    opt.point_size = 1  # 设置点云大小
    opt.show_coordinate_frame = True

    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=1280, height=720)
    # vis.create_window(window_name='pcd')
    opt = vis.get_render_option()  # 可视化参数设置
    opt.background_color = np.asarray([0, 0, 0])  # 设置背景色
    opt.point_size = 1  # 设置点云大小
    opt.show_coordinate_frame = True

    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    vis.run()
    vis.destroy_window()


def save_view_parameters():
    ply_path = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/graspness_infer_result/20_sematic.ply"
    pcd = o3d.io.read_point_cloud(ply_path)  # 传入自己当前的pcd文件
    print("save view joson...")
    save_view_point(pcd, "grasp_view_point.json")  # 保存好得json文件位置
    print("load view joson...")
    load_view_point(pcd, "grasp_view_point.json")  # 加载修改时较后的pcd文件


def save_grasp_show_img(root, s_root):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=1280, height=720)
    opt = vis.get_render_option()  # 可视化参数设置
    opt.background_color = np.asarray([0, 0, 0])  # 设置背景色
    opt.point_size = 1  # 设置点云大小
    opt.show_coordinate_frame = True

    # 加载背景点云
    background_pcd_path = None
    if background_pcd_path is not None:
        road_pc = o3d.geometry.PointCloud()
        road = o3d.io.read_point_cloud(background_pcd_path)
        road = np.asarray(road.points).reshape((-1, 3))  # 转为numpy格式
        road_pc.points = o3d.utility.Vector3dVector(road)
        road_pc.paint_uniform_color([0, 1, 0])  # 调整点云颜色
        vis.add_geometry(road_pc)

    to_reset = True
    # 读取viewpoint参数
    ctr = vis.get_view_control()
    if to_reset:
        vis.reset_view_point(True)
        to_reset = False
    vis.poll_events()
    vis.update_renderer()

    grasp_names = [name for name in os.listdir(root) if name.endswith("_gg.npy")]
    grasp_names = list(sorted(grasp_names, key=lambda x: int(x.split("_")[0])))

    # dex_hand_path = "/home/pxn-lyj/Egolee/programs/graspness_implementation_liyj/local_files/data/test_grippers/align_tora_hand.obj"
    dex_hand_path = "/home/pxn-lyj/Egolee/data/tora_dethand/tora_hand_参数手势/20241014_0/tora_hand_R_90_180_0_-0.09_-0.08_0.04_0.0_0.0_0.0_0.0_0.0_0.0_0.0_0.0_0.0_90.0_0.0_15.0_0.0_90.0_15.0_15.0.obj"
    dex_hand_mesh = tm.load_mesh(dex_hand_path, process=False)

    for grasp_name in grasp_names:
        # pc_path = os.path.join(root, grasp_name.replace("_gg.npy", ".npy"))
        pc_path = os.path.join(root, grasp_name.replace("_gg.npy", ".ply"))
        grasp_path = os.path.join(root, grasp_name)
        preds = np.load(grasp_path)

        if pc_path.endswith(".npy"):
            pc = np.load(pc_path)
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
        else:
            cloud = o3d.io.read_point_cloud(pc_path)

        gg = GraspGroup(preds)
        gg = gg.nms()
        gg = gg.sort_by_score()
        if gg.__len__() > 30:
            gg = gg[:30]
        grippers = gg.to_open3d_geometry_list()
        # grippers = grippers[0:1]

        # align_dex_hands = get_align_dex_hand_mesh(dex_hand_mesh, gg)
        align_dex_hands = get_align_dex_hand_mesh(dex_hand_mesh, gg[0:1])

        # align_dex_hands = align_dex_hands[0:1]
        grippers = []
        # align_dex_hands = []

        # o3d.visualization.draw_geometries([cloud, *grippers])
        # o3d.visualization.draw_geometries([cloud, *align_dex_hands[:5]])
        # o3d.visualization.draw_geometries([cloud, align_dex_hands[0]])

        vis.add_geometry(cloud)
        for i in range(len(grippers)):
            vis.add_geometry(grippers[i])

        for i in range(len(align_dex_hands)):
            vis.add_geometry(align_dex_hands[i])

        view_point_param = o3d.io.read_pinhole_camera_parameters("grasp_view_point.json")

        ctr.convert_from_pinhole_camera_parameters(view_point_param, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()

        image = vis.capture_screen_float_buffer(False)
        s_img_path = os.path.join(s_root, grasp_name.replace("_gg.npy", "_3d.jpg"))
        plt.imsave(s_img_path, np.asarray(image), dpi=1)

        # vis.run()
        # 删除上一帧vis中的点云
        vis.remove_geometry(cloud)
        for i in range(len(grippers)):
            vis.remove_geometry(grippers[i])

        for i in range(len(align_dex_hands)):
            vis.remove_geometry(align_dex_hands[i])

    vis.destroy_window()


if __name__ == "__main__":
    print("Start")
    # 保存可视化视角文件
    # save_view_parameters()

    # root = "/home/pxn-lyj/Egolee/programs/graspness_implementation_liyj/local_files/data/test_grippers"
    # s_root = "/home/pxn-lyj/Egolee/programs/graspness_implementation_liyj/local_files/data/test_grippers/save_view_imgs"

    root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/graspness_infer_result"
    s_root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/graspness_infer_result_show"

    # root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241011/graspness_infer_result"
    # s_root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241011/graspness_infer_result_show"

    if os.path.exists(s_root):
        shutil.rmtree(s_root)
    os.mkdir(s_root)

    save_grasp_show_img(root, s_root)
    print("End")
