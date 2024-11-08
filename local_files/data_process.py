import copy
import os
import torch
import shutil
import numpy as np
import cv2
import open3d as o3d
from utils_data import image_seg_depth_to_point_cloud, save_2_ply, image_and_depth_to_point_cloud
import matplotlib.pyplot as plt
import trimesh as tm
from PIL import Image
from tqdm import tqdm


def save_pcs_with_sematic():
    root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009"
    # root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241011"
    # root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241012_laser"
    # root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241012_laser"

    img_root = os.path.join(root, "colors")
    vis_img_root = os.path.join(root, "colors_vis")
    mask_root = os.path.join(root, "masks_num")
    # mask_root = os.path.join(root, "masks_num_tmp")
    depth_root = os.path.join(root, "depths")

    img_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", ".png"]]
    img_names = list(sorted(img_names, key=lambda x: int(x.split(".")[0].split("_")[0])))

    cam_k_path = os.path.join(root, "cam_k.txt")
    cam_k = np.loadtxt(cam_k_path)
    fx = cam_k[0, 0]
    fy = cam_k[1, 1],
    cx = cam_k[0, 2]
    cy = cam_k[1, 2]

    # cam_k = torch.from_numpy(cam_k)

    setmatic_root = os.path.join(root, "pcs_sematic")
    if os.path.exists(setmatic_root):
        shutil.rmtree(setmatic_root)
    os.mkdir(setmatic_root)

    for img_name in img_names:

        img_path = os.path.join(img_root, img_name)
        vis_img_path = os.path.join(vis_img_root, img_name)
        mask_path = os.path.join(mask_root, img_name.replace("_color.jpg", "_mask.npy"))
        depth_path = os.path.join(depth_root, img_name.replace("_color.jpg", "_depth.npy"))

        img = cv2.imread(img_path)
        # vis_img = cv2.imread(vis_img_path)
        mask = np.load(mask_path)
        seg = mask.astype(np.uint8)
        depth = np.load(depth_path)

        pts, colors, sematics = image_seg_depth_to_point_cloud(img, seg, depth, fx=fx, fy=fy, cx=cx, cy=cy, max_depth=5.0)
        # s_ply_path = os.path.join(setmatic_root, img_name.replace("_color.jpg", "_ply.ply"))
        # save_2_ply(s_ply_path, pts[:, 0], pts[:, 1], pts[:, 2], colors.tolist())
        pts_with_sematics = np.concatenate([pts, colors, sematics], axis=1)
        s_samatic_path = os.path.join(setmatic_root, img_name.replace("_color.jpg", "_sematic.npy"))
        np.save(s_samatic_path, pts_with_sematics)


def show_mask():
    mask_path = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/masks_num_tmp/20_mask.npy"
    mask = np.load(mask_path)
    plt.imshow(mask)
    plt.show()


def save_mask_2_img():
    root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241012_laser"
    img_root = os.path.join(root, "colors")
    mask_root = os.path.join(root, "masks_num")
    s_root = os.path.join(root, "masks_num_img")
    if os.path.exists(s_root):
        shutil.rmtree(s_root)
    os.mkdir(s_root)

    mask_names = [name for name in os.listdir(mask_root) if name[-4:] in [".npy"]]
    for mask_name in mask_names:
        mask_path = os.path.join(mask_root, mask_name)
        mask = np.load(mask_path)
        mask = mask > 0

        mask_show_img = np.ones((mask.shape[0], mask.shape[1], 3)) * 255
        mask_show_img[mask] = (0, 255, 0)

        img_path = os.path.join(img_root, mask_name.replace("_mask.npy", "_color.jpg"))
        img = cv2.imread(img_path)

        show_img = copy.deepcopy(img)
        show_img[mask] = img[mask] * 0.7 + mask_show_img[mask] * 0.3

        s_img_path = os.path.join(s_root, mask_name.replace(".npy", ".jpg"))
        cv2.imwrite(s_img_path, show_img)


def combine_imgs():
    img_root0 = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/masks_num_img"
    img_root1 = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/graspness_infer_result_show0"
    img_root2 = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/graspness_infer_result_show1"
    img_root3 = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/graspness_infer_result_show2"
    s_root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/combine_show"
    if os.path.exists(s_root):
        shutil.rmtree(s_root)
    os.mkdir(s_root)


    cols_num = 2
    rows_num = 2
    roots = [img_root0, img_root1, img_root2, img_root3]
    root_img_names = []
    for root in roots:
        img_names = [name for name in os.listdir(root) if name[-4:] in [".jpg", ".png"]]
        img_names = list(sorted(img_names, key=lambda x: int(x.split(".")[0].split("_")[0])))
        root_img_names.append(img_names)

    img0 = cv2.imread(os.path.join(roots[0], root_img_names[0][0]))
    d_img_h, d_img_w, _ = img0.shape

    imgs_num = len(root_img_names[0])
    for img_num in range(imgs_num):
        count = 0
        row_imgs = []
        for i in range(rows_num):
            col_imgs = []
            for j in range(cols_num):
                d_root = roots[count]
                d_img_name = root_img_names[count][img_num]
                d_img_path = os.path.join(d_root, d_img_name)
                img = cv2.imread(d_img_path)
                img_h, img_w, _ = img.shape
                if img_h != d_img_h or img_w != d_img_w:
                    # crop first
                    img = img[310:714, 340:909, :]
                    img = cv2.resize(img, (d_img_w, d_img_h))

                col_imgs.append(img)
                count = count + 1
            col_imgs = cv2.hconcat(col_imgs)
            row_imgs.append(col_imgs)
        combine_imgs = cv2.vconcat(row_imgs)

        s_img_path = os.path.join(s_root, root_img_names[0][img_num])
        cv2.imwrite(s_img_path, combine_imgs)
        # plt.imshow(combine_imgs)
        # plt.show()
        # exit(1)


def get_object_mesh_in_scence():
    root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009"
    mesh_path = "/home/pxn-lyj/Egolee/programs/UniDexGrasp_liyj/dexgrasp_generation/local_files/data/render_show_obj000490/obj_000490_downsample/untitled.obj"
    pose_root = os.path.join(root, "poses")

    pose_names = [name for name in os.listdir(pose_root) if name[-4:] in [".npy"]]
    pose_names = list(sorted(pose_names, key=lambda x: int(x.split(".")[0].split("_")[0])))

    object_mesh = tm.load_mesh(mesh_path, process=False)
    s_root = os.path.join(root, "align_meshs")
    if os.path.exists(s_root):
        shutil.rmtree(s_root)
    os.mkdir(s_root)

    for pose_name in pose_names:
        pose_path = os.path.join(pose_root, pose_name)
        pose = np.load(pose_path)

        rot = pose[:3, :3]
        trans = pose[:3, 3]
        align_vertices = np.dot(rot, object_mesh.vertices.T).T + trans
        object_mesh.vertices = align_vertices

        object_mesh.export(os.path.join(s_root, pose_name.replace("_pose.npy", "_align.obj")))
        print("ff")
        exit(1)


def save_depth_2_ply():
    # root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009"
    root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241012"
    img_root = os.path.join(root, "colors")
    depth_root = os.path.join(root, "depths")

    img_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", ".png"]]
    img_names = list(sorted(img_names, key=lambda x: int(x.split(".")[0].split("_")[0])))

    cam_k_path = os.path.join(root, "cam_k.txt")
    cam_k = np.loadtxt(cam_k_path)
    fx = cam_k[0, 0]
    fy = cam_k[1, 1],
    cx = cam_k[0, 2]
    cy = cam_k[1, 2]

    ply_root = os.path.join(root, "plys")
    if os.path.exists(ply_root):
        shutil.rmtree(ply_root)
    os.mkdir(ply_root)

    for img_name in img_names:

        img_path = os.path.join(img_root, img_name)
        depth_path = os.path.join(depth_root, img_name.replace("_color.jpg", "_depth.npy"))

        img = cv2.imread(img_path)
        depth = np.load(depth_path)

        img_path = os.path.join(img_root, img_name)
        depth_path = os.path.join(depth_root, img_name.replace("_color.jpg", "_depth.npy"))

        img = cv2.imread(img_path)
        depth = np.load(depth_path)

        points, colors = image_and_depth_to_point_cloud(img, depth, fx=fx, fy=fy, cx=cx, cy=cy, max_depth=5.0)

        s_ply_path = os.path.join(ply_root, img_name.replace("_color.jpg", "_pc.ply"))
        save_2_ply(s_ply_path, points[:, 0], points[:, 1], points[:, 2], colors.tolist())


def save_align_mesh_2_sematic():
    root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009"

    img_root = os.path.join(root, "colors")
    vis_img_root = os.path.join(root, "colors_vis")
    mask_root = os.path.join(root, "masks_num")
    depth_root = os.path.join(root, "depths")
    object_mesh_root = os.path.join(root, "align_meshs")

    img_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", ".png"]]
    img_names = list(sorted(img_names, key=lambda x: int(x.split(".")[0].split("_")[0])))

    cam_k_path = os.path.join(root, "cam_k.txt")
    cam_k = np.loadtxt(cam_k_path)
    fx = cam_k[0, 0]
    fy = cam_k[1, 1],
    cx = cam_k[0, 2]
    cy = cam_k[1, 2]

    # cam_k = torch.from_numpy(cam_k)

    setmatic_root = os.path.join(root, "object_pcs_sematic")
    if os.path.exists(setmatic_root):
        shutil.rmtree(setmatic_root)
    os.mkdir(setmatic_root)

    for img_name in img_names:
        mesh_path = os.path.join(object_mesh_root, img_name.replace("_color.jpg", "_align.obj"))
        # object_mesh = o3d.io.read_triangle_mesh(mesh_path)
        # vertices = np.asarray(mesh.vertices)
        # colors = np.asarray(mesh.vertex_colors)

        object_mesh = tm.load_mesh(mesh_path)
        pts = object_mesh.vertices

        texture_image_path = os.path.join(object_mesh_root, "material_0.png")
        if os.path.exists(texture_image_path):
            texture_image = Image.open(texture_image_path).convert('RGB')
            texture_data = np.array(texture_image) / 255.0  # 归一化到[0, 1]

            # 映射UV坐标到颜色
            colors = np.zeros((len(pts), 3))
            uv_coords = object_mesh.visual.uv
            for i, (u, v) in enumerate(uv_coords):
                # 将UV坐标转换为图像坐标
                u_img = int(u * (texture_data.shape[1] - 1))
                v_img = int((1 - v) * (texture_data.shape[0] - 1))  # Y轴翻转
                colors[i] = texture_data[v_img, u_img]
        else:
            colors = np.ones_like(pts)

        colors = colors * 255
        colors = colors.astype(np.uint8)[:, ::-1]
        sematics = np.ones_like(colors)[:, 0:1]

        # s_ply_path = os.path.join(setmatic_root, img_name.replace("_color.jpg", "_ply.ply"))
        # save_2_ply(s_ply_path, pts[:, 0], pts[:, 1], pts[:, 2], colors.tolist())
        pts_with_sematics = np.concatenate([pts, colors, sematics], axis=1)
        s_samatic_path = os.path.join(setmatic_root, img_name.replace("_color.jpg", "_sematic.npy"))
        np.save(s_samatic_path, pts_with_sematics)
        exit(1)


def save_imgs_2_video():
    imgs_root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/combine_show"
    img_names = [name for name in os.listdir(imgs_root) if name.endswith(".jpg")]
    img_names = list(sorted(img_names, key=lambda x: int(x.split(".")[0].split("_")[0])))
    img_paths = [os.path.join(imgs_root, img_name) for img_name in img_names]

    img = cv2.imread(img_paths[0])
    img_height, img_width, _ = img.shape

    # img_height = img_height // 6
    # img_width = img_width // 6

    print(img_height, img_width)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_path = os.path.join(imgs_root, "out_video.mp4")
    out = cv2.VideoWriter(video_path, fourcc, 1, (img_width, img_height))

    for img_path in tqdm(img_paths):
        # print(img_name)
        frame = cv2.imread(img_path)
        frame = cv2.resize(frame, (img_width, img_height))

        img_h, img_w, _ = frame.shape
        out.write(frame)
    out.release()


if __name__ == "__main__":
    print("Start")
    # 采用numpy保存点云、rgb以及颜色信息
    # save_pcs_with_sematic()

    # show_mask()
    # save_mask_2_img()
    # combine_imgs()

    # 根据pose信息将物体转到场景点云中
    # get_object_mesh_in_scence()

    # 将depth保存为ply
    # save_depth_2_ply()

    # 将对齐后的object-mesh保存为_sematic
    # save_align_mesh_2_sematic()

    save_imgs_2_video()

    print("End")
