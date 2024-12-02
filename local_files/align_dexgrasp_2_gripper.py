import os
import open3d as o3d
import numpy as np
import torch
import trimesh as tm
import matplotlib.pyplot as plt
from graspnetAPI.graspnet_eval import GraspGroup
from tora_hand_builder import ToraHandBuilder



def debug_align_dexgrasp_2_gripper(hand_mesh_path=None):
    s_root = "/home/pxn-lyj/Egolee/data/test/mesh_jz/graspness_infer_result"

    # pc_path = "/home/pxn-lyj/Egolee/programs/graspness_implementation_liyj/local_files/data/test_grippers/20_sematic.npy"
    # gp_path = "/home/pxn-lyj/Egolee/programs/graspness_implementation_liyj/local_files/data/test_grippers/20_sematic_gg.npy"
    gp_path = os.path.join(s_root, "00445_sematic_gg.npy")
    grasp_poses = np.load(gp_path)
    # 选择第一个作为调试的对象
    # grasp_poses = grasp_poses[1:2, ]

    gg = GraspGroup(grasp_poses)
    gg = gg.nms()
    gg = gg.sort_by_score()
    if gg.__len__() > 30:
        gg = gg[:30]
    grippers = gg.to_open3d_geometry_list()
    #
    o3d.io.write_triangle_mesh(os.path.join(s_root, "grippers.obj"), grippers[17])

    # for i in range(len(grasp_poses)):
    for i in range(len(gg)):
        if i!= 17:
            continue
        # 长度为17
        # grasp_pose = grasp_poses[0]
        grasp_pose = gg[i]
        # scores [:,0]
        # widths [:,1]
        # heights [:,2]
        # depths [:,3]
        # rotation_matrices [:, 4:13]
        # translations [:,13:16]
        # object_ids [:,16]

        # scores = grasp_pose[0]
        # widths = grasp_pose[1]
        # heights = grasp_pose[2]
        # depths = grasp_pose[3]
        # rotation_matrices = grasp_pose[4:13]
        # translations = grasp_pose[13:16]
        # object_ids = grasp_pose[16]

        scores = grasp_pose.score
        widths = grasp_pose.width
        heights = grasp_pose.height
        depths = grasp_pose.depth
        rotation_matrices = grasp_pose.rotation_matrix
        translations = grasp_pose.translation
        object_ids = grasp_pose.object_id
        print(scores, widths, heights, depths, rotation_matrices, translations, object_ids)

        # 将dexhand对齐到场景中
        # hand_mesh_path = "/home/pxn-lyj/Egolee/programs/graspness_implementation_liyj/local_files/data/test_grippers/tora_hand.obj"
        # hand_mesh_path = "/home/pxn-lyj/Egolee/data/tora_dethand/tora_hand_参数手势/20241014_0/tora_hand_R_90_180_0_-0.09_-0.08_0.04_0.0_0.0_0.0_0.0_0.0_0.0_0.0_0.0_0.0_90.0_0.0_15.0_0.0_90.0_15.0_15.0.obj"
        if hand_mesh_path is None:
            hand_mesh_path = "/home/pxn-lyj/Egolee/data/tora_dethand/tora_hand_参数手势/concact_map/tora_hand_R_90_180_0_-0.09_-0.08_0.04_0.0_90.0_15.0_15.0_0.0_90.0_15.0_15.0_0.0_90.0_15.0_15.0_0.0_90.0_30.0_30.0.obj"
        dex_hand_mesh = tm.load_mesh(hand_mesh_path, process=False)

        align_vertices = np.dot(rotation_matrices.reshape(3, 3), dex_hand_mesh.vertices.T).T + translations
        dex_hand_mesh.vertices = align_vertices
        # dex_hand_mesh.export(os.path.join(s_root, "align_dex_hand.obj"))
        dex_hand_mesh.export(os.path.join(s_root, "align_dex_hand_{}.obj".format(i)))
        # o3d.io.write_triangle_mesh(os.path.join(s_root, "align_dex_hand.obj"), dex_hand_mesh)


def show_gripper_coordinate():
    s_root = "/home/pxn-lyj/Egolee/data/tora_dethand/tora_hand_参数手势/20241014_1"
    score = 1.0
    width = 0.02
    height = 0.02
    # depth = 0.02
    depth = 0.01
    rot = np.eye(3).reshape(-1).tolist()
    trans = np.zeros(3).tolist()
    object = 0

    grasp_pose = [score, width, height, depth, *rot, *trans, object]
    grasp_pose = np.array(grasp_pose).reshape(1, 17)
    gg = GraspGroup(grasp_pose)
    grippers = gg.to_open3d_geometry_list()

    cord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)

    # o3d.visualization.draw_geometries([cord, *grippers])
    # 主要根据 width和depth生成gripper，其他参数如是固定的
    #     height=0.004
    #     finger_width = 0.004
    #     tail_length = 0.04
    #     depth_base = 0.02


    # dexhand需要根据width调整手指的张开程度
    # depth代表的夹持的深度, 感觉dexhand与gripper之间的RT固定后，dexhand与物体之间的距离就是固定的
    # 需要dexhand与gripper之间固定的RT关系
    o3d.io.write_triangle_mesh(os.path.join(s_root, "grippers_width_0.02_depth_0.01.obj"), grippers[0])

    print("ff")


def show_tora_hand_mesh():
    mesh_dir = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_R/meshes"
    urdf_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_R/urdf/ZY_R.urdf"

    # mesh_dir = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_L/meshes"
    # urdf_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/tora_dethand/tora_hand_urdf/ZY_L/urdf/ZY_L.urdf"

    tora_hand_builder = ToraHandBuilder(mesh_dir, urdf_path)

    print(len(tora_hand_builder.mesh))

    # ['wmzcb', 'wmzjdxz', 'wmzzdxz', 'wmzydxz', 'zzcb', 'zzjdxz', 'zzzdxz', 'zzydxz', 'szcb', 'szjdxz', 'szzdxz', 'szydxz', 'mzcb', 'mzjdxz', 'mzzd', 'mzydxz'
    print(tora_hand_builder.chain.get_joint_parameter_names())

    hand_rotation = torch.eye(3)
    hand_translation = torch.zeros(3)
    hand_qpos = torch.zeros(16)

    # 调整旋转角度
    # roll, pitch, yaw = -90, 90, 0  # roll 对应x轴 pitch y轴
    # roll, pitch, yaw = -90, 90, -90  # roll 对应x轴 pitch y轴
    roll, pitch, yaw = 90, 180, 0  # roll 对应x轴 pitch y轴
    rot = np.array([roll, pitch, yaw])
    # roll, pitch, yaw = np.deg2rad(rot)
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.deg2rad(rot))
    R = R.astype(np.float32)

    # 验证是否为旋转矩阵
    # R = -np.eye(3)
    # from scipy.spatial.transform import Rotation as Rt
    # r = Rt.from_matrix(R)
    # euler_angles = r.as_euler('zyx', degrees=True)
    # print(euler_angles)
    #
    # r = Rt.from_euler('zyx', euler_angles, degrees=True)
    # R = r.as_matrix()
    # print(R)


    hand_rotation = torch.from_numpy(R)
    hand_rotation = hand_rotation.float()

    # # 调整平移
    hand_translation[0] = -0.09
    hand_translation[1] = -0.08
    hand_translation[2] = 0.035

    hand_qpos_np = np.zeros(16)

    # 调整拇指的手势
    # mzjdxz 调整为90度

    # hand_qpos_np[12] = 90  # 拇指
    hand_qpos_np[13] = 80  # 拇指
    # hand_qpos_np[14] = 15  # 拇指
    # hand_qpos_np[15] = 15  # 拇指

    hand_qpos_np[14] = 30  # 拇指
    hand_qpos_np[15] = 30  # 拇指


    # 无名指
    hand_qpos_np[0] = 0  # 左右
    hand_qpos_np[1] = 80  # 里面那节
    hand_qpos_np[2] = 5
    hand_qpos_np[3] = 10

    # 中指
    hand_qpos_np[4] = 0  # 左右
    hand_qpos_np[5] = 80  # 里面那节
    hand_qpos_np[6] = 15
    hand_qpos_np[7] = 15

    # 食指
    hand_qpos_np[8] = 0  # 左右
    hand_qpos_np[9] = 75  # 里面那节
    # hand_qpos_np[10] = 0
    # hand_qpos_np[11] = 15

    hand_qpos_np[10] = 30
    hand_qpos_np[11] = 30

    hand_qpos = torch.from_numpy(np.deg2rad(hand_qpos_np).astype(np.float32))

    hand_mesh = tora_hand_builder.get_hand_mesh(
        rotation_mat=hand_rotation,
        world_translation=hand_translation,
        qpos=hand_qpos,
    )
    hand_mesh = tm.Trimesh(
        vertices=hand_mesh.verts_list()[0].numpy(),
        faces=hand_mesh.faces_list()[0].numpy()
        # face_colors =
    )

    hand_mesh.visual.face_colors = [255, 255, 0, 255]
    params = [roll, pitch, yaw] + hand_translation.tolist() + hand_qpos_np.tolist()
    params_str = "_".join([str(round(param, 2)) for param in params])
    # s_root = "/home/pxn-lyj/Egolee/data/tora_dethand/tora_hand_参数手势/20241014_1"
    s_root = "/home/pxn-lyj/Egolee/data/tora_dethand/tora_hand_参数手势/concact_map"
    hand_mesh_path = os.path.join(s_root, "tora_hand_R_" + params_str + ".obj")
    hand_mesh.export(hand_mesh_path)
    return hand_mesh_path


if __name__ == "__main__":
    print("Start")
    # 通过mujoco-view可以观察到一些参数效果
    # python - m mujoco.viewer

    # 得到gripper的坐标位置和在没有旋转平移情况下的mesh模型
    # show_gripper_coordinate()

    # 调整dexhand的参数与gripper的位姿关系，保存mesh模型
    hand_mesh_path = show_tora_hand_mesh()

    # 将调整好参数的dexhand的mesh模型根据gripper的rt对齐到场景中
    debug_align_dexgrasp_2_gripper(hand_mesh_path)
    print("End")
