import copy
import os
import open3d as o3d
import numpy as np
import trimesh as tm
import matplotlib.pyplot as plt
from graspnetAPI.graspnet_eval import GraspGroup
from utils_data import generate_distinct_colors
from utils_data import get_align_dex_hand_mesh



def vis_point_clouds_grasp_pose():
    # root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009"
    # root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241012_laser"
    # root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241012_no_laser"
    root = "/home/pxn-lyj/Egolee/data/test/mesh_jz"

    infer_result_root = os.path.join(root, "graspness_infer_result")
    grasp_names = [name for name in os.listdir(infer_result_root) if name.endswith("_gg.npy")]
    grasp_names = list(sorted(grasp_names, key=lambda x: int(x.split("_")[0])))

    s_root = os.path.join(root, "grippers")
    os.makedirs(s_root, exist_ok=True)

    for grasp_name in grasp_names:
        # pc_path = os.path.join(root, grasp_name.replace("_gg.npy", ".npy"))
        pc_path = os.path.join(infer_result_root, grasp_name.replace("_gg.npy", ".ply"))
        grasp_path = os.path.join(infer_result_root, grasp_name)
        preds = np.load(grasp_path)

        if pc_path.endswith(".npy"):
            pc = np.load(pc_path)
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
        else:
            cloud = o3d.io.read_point_cloud(pc_path)

        # scores = preds[:, 0]
        # widths = preds[:, 1]
        # min_score = 0.1
        # max_width = 0.08
        # mask = (np.bitwise_and(scores > min_score, widths < max_width))
        # preds = preds[mask, :]

        gg = GraspGroup(preds)
        gg = gg.nms()
        gg = gg.sort_by_score()
        if gg.__len__() > 30:
            gg = gg[:30]
        grippers = gg.to_open3d_geometry_list()

        s_gripper_path = os.path.join(s_root, grasp_name.replace("_gg.npy", "_grippers.obj"))
        print(len(grippers), s_gripper_path)
        merged_mesh = o3d.geometry.TriangleMesh()
        for mesh in grippers:
            merged_mesh += mesh

        o3d.io.write_triangle_mesh(s_gripper_path, merged_mesh)

        o3d.visualization.draw_geometries([cloud, *grippers])
        o3d.visualization.draw_geometries([cloud])
        exit(1)



def vis_point_clouds_grasp_pose_with_dex_hand(root):

    # root = "/home/pxn-lyj/Egolee/programs/graspness_implementation_liyj/local_files/data/test_grippers"
    grasp_names = [name for name in os.listdir(root) if name.endswith("_gg.npy")]
    grasp_names = list(sorted(grasp_names, key=lambda x: int(x.split("_")[0])))

    dex_hand_path = "/home/pxn-lyj/Egolee/programs/graspness_implementation_liyj/local_files/data/test_grippers/align_tora_hand.obj"
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

        # scores = preds[:, 0]
        # widths = preds[:, 1]
        # min_score = 0.1
        # max_width = 0.08
        # mask = (np.bitwise_and(scores > min_score, widths < max_width))
        # preds = preds[mask, :]

        gg = GraspGroup(preds)
        gg = gg.nms()
        gg = gg.sort_by_score()
        if gg.__len__() > 30:
            gg = gg[:30]
        grippers = gg.to_open3d_geometry_list()
        align_dex_hands = get_align_dex_hand_mesh(dex_hand_mesh, gg)

        # o3d.io.write_triangle_mesh(os.path.join(root, "grippers.obj"), grippers[0])
        # o3d.io.write_triangle_mesh(os.path.join(root, "align_dex_hands.obj"), align_dex_hands[0])

        # o3d.visualization.draw_geometries([cloud, *grippers])
        # o3d.visualization.draw_geometries([cloud, *align_dex_hands[:5]])
        # o3d.visualization.draw_geometries([cloud, align_dex_hands[0]])
        o3d.visualization.draw_geometries([cloud, grippers[0]])
        # o3d.visualization.draw_geometries([cloud])
        exit(1)



if __name__ == "__main__":
    print("STart")
    # 在点云中可视化grippers
    vis_point_clouds_grasp_pose()

    # 同时可视化对齐的dex-hand
    # root = "/home/pxn-lyj/Egolee/programs/graspness_implementation_liyj/local_files/data/test_grippers"
    # vis_point_clouds_grasp_pose_with_dex_hand(root)
    print("End")