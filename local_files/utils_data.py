import os
import json
import torch
import numpy as np
# import pytorch3d.ops
import colorsys
import random
import open3d as o3d


def get_align_dex_hand_mesh(dex_hand_mesh, grasp_poses):
    align_dex_hand_meshs = []
    translations = grasp_poses.translations
    rotation_matrices = grasp_poses.rotation_matrices
    colors = generate_distinct_colors(len(rotation_matrices))

    for rot, trans, color in zip(rotation_matrices, translations, colors):
        # align_dex_hand_mesh = copy.deepcopy(dex_hand_mesh)

        # rot = grasp_pose.rotation_matrix()
        # trans = grasp_pose.translation()

        align_vertices = np.dot(rot, dex_hand_mesh.vertices.T).T + trans
        # align_dex_hand_mesh.vertices = align_vertices

        # 将 trimesh 的顶点和面数据转换为 Open3D 所需的格式
        o3d_vertices = o3d.utility.Vector3dVector(np.array(align_vertices))
        o3d_triangles = o3d.utility.Vector3iVector(np.array(dex_hand_mesh.faces))

        # 创建 Open3D 的三角网格对象
        align_dex_hand_mesh = o3d.geometry.TriangleMesh(vertices=o3d_vertices, triangles=o3d_triangles)
        # align_dex_hand_mesh.paint_uniform_color([1.0, 1.0, 0])
        # align_dex_hand_mesh.paint_uniform_color([200/255, 200/255, 200/255])
        # align_dex_hand_mesh.paint_uniform_color([255/255, 255/255, 200/255])
        align_dex_hand_mesh.paint_uniform_color([79/255, 79/255, 79/255])
        # align_dex_hand_mesh.paint_uniform_color([color[0]/255.0, color[1]/255.0, color[2]/255.0])

        align_dex_hand_meshs.append(align_dex_hand_mesh)

    return align_dex_hand_meshs

def generate_distinct_colors(num_colors):
    random.seed(42)  # 设置随机种子为 42，你可以根据需要设置其他值
    colors = []
    for i in range(num_colors):
        h = i / num_colors
        s = 0.8 + random.random() * 0.2
        v = 0.8 + random.random() * 0.2
        r, g, b = [int(255 * x) for x in colorsys.hsv_to_rgb(h, s, v)]
        colors.append((r, g, b))
    return colors


def image_seg_depth_to_point_cloud(image, seg, depth, fx, fy, cx, cy, max_depth=5.0):
    rows, cols = depth.shape
    u, v = np.meshgrid(range(cols), range(rows))
    z = depth
    # 将深度为 0 或小于等于某个极小值的点标记为无效
    invalid_mask = np.bitwise_or(np.bitwise_or(z <= 0, z < np.finfo(np.float32).eps), z > max_depth)
    x = np.where(~invalid_mask, (u - cx) * z / fx, 0)
    y = np.where(~invalid_mask, (v - cy) * z / fy, 0)
    # z = z[~invalid_mask]
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = image.reshape(-1, 3)
    sematics = seg.reshape(-1, 1)

    points = points[~invalid_mask.reshape(-1)]
    colors = colors[~invalid_mask.reshape(-1)]
    sematics = sematics[~invalid_mask.reshape(-1)]

    return points, colors, sematics


def image_and_depth_to_point_cloud(image, depth, fx, fy, cx, cy, max_depth=5.0):
    rows, cols = depth.shape
    u, v = np.meshgrid(range(cols), range(rows))
    z = depth
    # 将深度为 0 或小于等于某个极小值的点标记为无效
    invalid_mask = np.bitwise_or(np.bitwise_or(z <= 0, z < np.finfo(np.float32).eps), z > max_depth)
    x = np.where(~invalid_mask, (u - cx) * z / fx, 0)
    y = np.where(~invalid_mask, (v - cy) * z / fy, 0)
    # z = z[~invalid_mask]
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = image.reshape(-1, 3)

    points = points[~invalid_mask.reshape(-1)]
    colors = colors[~invalid_mask.reshape(-1)]

    return points, colors

def get_mesh_data_object_list(root_path, mode, scales=[0.06, 0.08, 0.1, 0.12, 0.15], n_samples=1, random_sample=False):
    data_root_path = os.path.join(root_path, 'DFCData', 'meshes')
    splits_path = os.path.join(root_path, 'DFCData', 'splits')

    object_code_list = []
    for splits_file_name in os.listdir(splits_path):
        with open(os.path.join(splits_path, splits_file_name), 'r') as f:
            splits_map = json.load(f)
        object_code_list += [os.path.join(splits_file_name[:-5], object_code) for object_code in splits_map[mode]]

    object_list = []
    for object_code in object_code_list:
        pose_matrices = np.load(os.path.join(data_root_path, object_code, 'poses.npy'))
        pcs_table = np.load(os.path.join(data_root_path, object_code, 'pcs_table.npy'))
        for scale in scales:
            indices = np.random.permutation(len(pose_matrices))[:n_samples]

            # 是否随机采样不同pose的sample
            if not random_sample:
                indices = np.arange(len(pose_matrices))[:n_samples]

            for index in indices:
                pose_matrix = pose_matrices[index]
                pose_matrix[:2, 3] = 0
                object_list.append((object_code, pcs_table[index], scale, pose_matrix))
    return object_list


def save_2_ply(file_path, x, y, z, color=None):
    points = []
    if color == None:
        color = [[255, 255, 255]] * len(x)
    for X, Y, Z, C in zip(x, y, z, color):
        points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, C[2], C[1], C[0]))

    # for X, Y, Z, C in zip(x, y, z, color):
    #     points.append("%f %f %f %d %d %d 0\n" % (Z, X, Y, C[0], C[1], C[2]))

    file = open(file_path, "w")
    file.write('''ply
          format ascii 1.0
          element vertex %d
          property float x
          property float y
          property float z
          property uchar red
          property uchar green
          property uchar blue
          property uchar alpha
          end_header
          %s
          ''' % (len(points), "".join(points)))
    file.close()

#
# def get_object_data(object, use_table_pc_extra=False):
#     object_code, pcs_table, scale, pose_matrix = object
#     object_pc = torch.from_numpy(scale * (pcs_table @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]))
#     only_object_pc = object_pc[:3000]
#     table_pc = object_pc[3000:]
#     max_diameter = 0.2
#     n_samples_table_extra = 2000
#     min_diameter = (only_object_pc[:, 0] ** 2 + only_object_pc[:, 1] ** 2).max()
#     distances = min_diameter + (max_diameter - min_diameter) * torch.rand(n_samples_table_extra,
#                                                                           dtype=torch.float) ** 0.5
#     theta = 2 * np.pi * torch.rand(n_samples_table_extra, dtype=torch.float)
#
#     if use_table_pc_extra:
#         table_pc_extra = torch.stack(
#             [distances * torch.cos(theta), distances * torch.sin(theta), torch.zeros_like(distances)], dim=1)
#         table_pc = torch.cat([table_pc, table_pc_extra])
#
#     table_pc_cropped = table_pc[table_pc[:, 0] ** 2 + table_pc[:, 1] ** 2 < max_diameter]
#     table_pc_cropped_sampled = pytorch3d.ops.sample_farthest_points(table_pc_cropped.unsqueeze(0), K=1000)[0][0]
#
#     object_pc = torch.cat([only_object_pc, table_pc_cropped_sampled])
#     # object_pc = torch.cat([only_object_pc])
#     plane = torch.zeros_like(torch.from_numpy(pose_matrix[2]))
#     plane[2] = 1
#     # plane = pose_matrix[2].copy()
#     # plane[3] *= scale
#     ret_dict = {
#         "object_code": object_code,
#         "obj_pc": object_pc,
#         "plane": plane,
#         "scale": scale,
#     }
#     return ret_dict


def format_batch_data(ret_dict):
    batch_ret_dict = {}
    batch_ret_dict["object_code"] = [ret_dict["object_code"]]
    batch_ret_dict["obj_pc"] = torch.stack([ret_dict["obj_pc"]])
    batch_ret_dict["plane"] = torch.stack([ret_dict["plane"]])
    batch_ret_dict["scale"] = torch.stack([torch.tensor(ret_dict["scale"])])
    return batch_ret_dict


if __name__ == "__main__":
    print("Start")
    root_path = "/home/pxn-lyj/Egolee/data/unidexgrasp_data"
    object_list = get_mesh_data_object_list(root_path, 'test', scales=[0.06], n_samples=5)

    # check obj-pts in table
    # poses主要是table的点云会有变化，物体的点云是不变的
    # for i in range(5):
    #     object = object_list[i]
    #     s_path = os.path.join("/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/object_pts_in_table", str(i) + "_" + object[0].replace("/", "_") + ".ply")
    #     save_2_ply(s_path, object[1][:, 0], object[1][:, 1], object[1][:, 2])

    # 会进行scale和pose的转换
    # 输入参数是否需要对tabel点云进行扩充
    # for i in range(5):
    #     object = object_list[i]
    #     object_data = get_object_data(object, use_table_pc_extra=False)
    #     s_path = os.path.join("/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/object_pts_in_pose_scale", str(i) + "_" + object[0].replace("/", "_") + "_pose_scale.ply")

        # object_data = get_object_data(object, use_table_pc_extra=True)
        # s_path = os.path.join("/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/object_pts_in_pose_scale", str(i) + "_" + object[0].replace("/", "_") + "_pose_scale_extra.ply")
        # object_pc = object_data["obj_pc"].detach().cpu().numpy()
        # save_2_ply(s_path, object_pc[:, 0], object_pc[:, 1], object_pc[:, 2])

    # batch data
    # for i in range(5):
    #     object = object_list[i]
    #     object_data = get_object_data(object, use_table_pc_extra=False)
    #     batch_object_data = format_batch_data(object_data)

    print("End")
