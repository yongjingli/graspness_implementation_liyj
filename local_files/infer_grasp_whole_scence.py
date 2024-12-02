import os
os.chdir("../")

import open3d as o3d
import numpy as np
from graspnetAPI.graspnet_eval import GraspGroup

import cv2
import sys
import shutil

from tqdm import tqdm
import collections.abc as container_abcs
import torch
import MinkowskiEngine as ME

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = ROOT_DIR.replace("local_files", "")
print("ROOT_DIR:", ROOT_DIR)
os.chdir(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

# from models.graspnet_pointnet import GraspNetPointnet
from models.graspnet import GraspNet, pred_decode

# from models.graspnet_pointnet_single_object import GraspNetPointnetSingleObject
from label_generation import batch_viewpoint_params_to_matrix
from collision_detector import ModelFreeCollisionDetector
from utils_data import save_2_ply


class GraspPoseInference(object):
    def __init__(self, checkpoint_path, device="cuda", seed_feat_dim=512, input_feature_dim=0, num_points=20000, voxel_size=0.005):
        self.num_points = num_points

        self.net = GraspNet(seed_feat_dim=seed_feat_dim, is_training=False)

        # self.net = GraspNetPointnet(seed_feat_dim=seed_feat_dim, is_training=False,
        #                             input_feature_dim=input_feature_dim)

        # self.net = GraspNetPointnetSingleObject(seed_feat_dim=seed_feat_dim, is_training=False,
        #                                         input_feature_dim=input_feature_dim)
        self.seed_feat_dim = seed_feat_dim
        self.input_feature_dim = input_feature_dim
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)" % (checkpoint_path, start_epoch))

        self.device = device
        self.net.to(device)

        self.GRASP_MAX_WIDTH = 0.1
        self.GRASPNESS_THRESHOLD = 0.1
        self.NUM_VIEW = 300
        self.NUM_ANGLE = 12
        self.NUM_DEPTH = 4
        self.M_POINT = 1024
        self.voxel_size = voxel_size

    def infer(self, input_data):
        batch_data = self.process_data(input_data)

        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(self.device)
            else:
                if batch_data[key] is not None:
                    batch_data[key] = batch_data[key].to(self.device)

        # Forward pass
        with torch.no_grad():
            end_points = self.net(batch_data)
            grasp_preds = self.pred_decode(end_points)
        return grasp_preds

    def pred_decode(self, end_points):
        batch_size = len(end_points['point_clouds'])
        grasp_preds = []
        for i in range(batch_size):
            grasp_center = end_points['xyz_graspable'][i].float()

            grasp_score = end_points['grasp_score_pred'][i].float()
            grasp_score = grasp_score.view(self.M_POINT, self.NUM_ANGLE * self.NUM_DEPTH)
            grasp_score, grasp_score_inds = torch.max(grasp_score, -1)  # [M_POINT]
            grasp_score = grasp_score.view(-1, 1)
            grasp_angle = (grasp_score_inds // self.NUM_DEPTH) * np.pi / 12
            grasp_depth = (grasp_score_inds % self.NUM_DEPTH + 1) * 0.01
            grasp_depth = grasp_depth.view(-1, 1)
            # grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
            grasp_width = 1.0 * end_points['grasp_width_pred'][i] / 10.
            grasp_width = grasp_width.view(self.M_POINT, self.NUM_ANGLE * self.NUM_DEPTH)
            grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
            grasp_width = torch.clamp(grasp_width, min=0., max=self.GRASP_MAX_WIDTH)

            approaching = -end_points['grasp_top_view_xyz'][i].float()
            grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
            grasp_rot = grasp_rot.view(self.M_POINT, 9)

            # merge preds
            grasp_height = 0.02 * torch.ones_like(grasp_score)
            obj_ids = -1 * torch.ones_like(grasp_score)

            grasp_preds.append(
                torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids],
                          axis=-1))
        return grasp_preds

    def process_data(self, input_data):
        point_clouds = input_data["point_clouds"]
        point_clouds_seg = input_data["point_clouds_seg"]
        point_clouds_sematic = input_data["point_clouds_sematic"]

        if len(point_clouds) >= self.num_points:
            idxs = np.random.choice(len(point_clouds), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(point_clouds))
            idxs2 = np.random.choice(len(point_clouds), self.num_points - len(point_clouds), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)


        point_clouds = point_clouds[idxs]

        point_clouds_seg = point_clouds_seg[idxs] if point_clouds_seg is not None else None
        point_clouds_sematic = point_clouds_sematic[idxs] if point_clouds_sematic is not None else None

        input_data = {'point_clouds': point_clouds.astype(np.float32),
                      'coors': point_clouds.astype(np.float32) / self.voxel_size,
                      'feats': np.ones_like(point_clouds).astype(np.float32),
                      'point_clouds_seg': point_clouds_seg,
                      'point_clouds_sematic': point_clouds_sematic
                      }

        list_data = [input_data]

        coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                    [d["feats"] for d in list_data])
        coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
            coordinates_batch, features_batch, return_index=True, return_inverse=True)
        res = {
            "coors": coordinates_batch,
            "feats": features_batch,
            "quantize2original": quantize2original
        }

        def collate_fn_(batch):
            if type(batch[0]).__module__ == 'numpy':
                return torch.stack([torch.from_numpy(b) for b in batch], 0)
            elif isinstance(batch[0], container_abcs.Sequence):
                return [[torch.from_numpy(sample) for sample in b] for b in batch]
            elif isinstance(batch[0], container_abcs.Mapping):
                for key in batch[0]:
                    if key == 'coors' or key == 'feats':
                        continue
                    res[key] = collate_fn_([d[key] for d in batch])
                return res

        res = collate_fn_(list_data)
        return res


def infer_grasp_whole_scence():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 提供的预训练权重
    checkpoint_path = "/home/pxn-lyj/Egolee/programs/graspness_implementation_liyj/trained_weights/minkresunet_epoch10-1.tar"
    grasp_pose_infer = GraspPoseInference(checkpoint_path, device=device, seed_feat_dim=512,
                                                      input_feature_dim=0, num_points=20000)

    # root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009"
    # root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241012_laser"
    # root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241012_no_laser"
    # root = "/home/pxn-lyj/Egolee/data/test/mesh_jz"
    root = "/home/pxn-lyj/Egolee/programs/splatter-image-liyj/local_files/tmp"
    point_clouds_root = os.path.join(root, "pcs_sematic")
    object_names = [name for name in os.listdir(point_clouds_root) if name.endswith(".npy") or name.endswith(".ply")]
    object_names = list(sorted(object_names, key=lambda x: int(x.split("_")[0])))

    # s_root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/graspness_infer_result2"
    s_root = os.path.join(root, "graspness_infer_result")
    if os.path.exists(s_root):
        shutil.rmtree(s_root)
    os.mkdir(s_root)

    for object_name in tqdm(object_names):
        # if object_name != "60_sematic.npy":
        #     continue

        point_clouds_path = os.path.join(point_clouds_root, object_name)

        if object_name.endswith(".npy"):
            point_clouds_data = np.load(point_clouds_path)
            if point_clouds_data.shape[-1] > 3:
                point_clouds = point_clouds_data[:, :3]
                colors = point_clouds_data[:, 3:6]
                sematics = point_clouds_data[:, 6:7]

            else:
                point_clouds = point_clouds_data[:3]
                colors = None
                sematics = None
        else:
            point_clouds_o3d = o3d.io.read_point_cloud(point_clouds_path)
            point_clouds = np.asarray(point_clouds_o3d.points)
            colors = np.asarray(point_clouds_o3d.colors) * 255
            colors = colors.astype(np.uint8)[:, ::-1]
            sematics = None

        # 将补全的物体点云放置到场景中
        if 0:
            obj_point_clouds_path = os.path.join(root, "object_pcs_sematic", object_name)
            obj_point_clouds_data = np.load(obj_point_clouds_path)
            if point_clouds_data.shape[-1] > 3:
                obj_point_clouds = obj_point_clouds_data[:, :3]
                obj_colors = obj_point_clouds_data[:, 3:6]
                obj_sematics = obj_point_clouds_data[:, 6:7]
            else:
                obj_point_clouds = point_clouds_data[:3]
                obj_colors = None
                obj_sematics = None

            point_clouds = np.concatenate([point_clouds, obj_point_clouds], axis=0)

            if colors is not None and obj_colors is not None:
                colors = np.concatenate([colors, obj_colors], axis=0)
            if sematics is not None and obj_sematics is not None:
                sematics = np.concatenate([sematics, obj_sematics], axis=0)

        # sematics = None
        point_clouds_seg = sematics.astype(np.float32) if sematics is not None else None
        # point_clouds_sematic = (sematics == 1).astype(np.int64) if sematics is not None else None  # select object id
        point_clouds_sematic = (sematics > 0).astype(np.int64) if sematics is not None else None
        input_data = {'point_clouds': point_clouds.astype(np.float32),
                      'point_clouds_seg': point_clouds_seg,
                      'point_clouds_sematic': point_clouds_sematic,
                        }

        grasp_preds = grasp_pose_infer.infer(input_data)

        preds = grasp_preds[0].detach().cpu().numpy()

        gg = GraspGroup(preds)

        if input_data["point_clouds_sematic"] is not None:
            if np.sum(input_data["point_clouds_sematic"]) == 0:
                gg = GraspGroup()   # empty

        # collision detection, 是否进行碰撞检测
        collision_det = 1
        if collision_det > 0:
            cloud = input_data['point_clouds']
            voxel_size_cd = 0.01
            collision_thresh = 0.01
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size_cd)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
            gg = gg[~collision_mask]

        print("gg size:", len(gg))
        s_gg_path = os.path.join(s_root, object_name[:-4] + "_gg.npy")
        gg.save_npy(s_gg_path)

        s_pc_ply_path = os.path.join(s_root, object_name[:-4] + ".ply")
        save_2_ply(s_pc_ply_path, point_clouds[:, 0], point_clouds[:, 1], point_clouds[:, 2], colors.tolist() if colors is not None else colors)

        gg = gg.nms()
        gg = gg.sort_by_score()
        if gg.__len__() > 30:
            gg = gg[:30]
        grippers = gg.to_open3d_geometry_list()
        s_gripper_path = os.path.join(s_root, object_name[:-4] + "_grippers.obj")
        merged_mesh = o3d.geometry.TriangleMesh()
        for mesh in grippers:
            merged_mesh += mesh
        o3d.io.write_triangle_mesh(s_gripper_path, merged_mesh)

        #
        # # s_pc_ply_path = os.path.join(s_root, object_name[:-4] + "_object.ply")
        # # save_pts_2_ply(s_pc_ply_path, input_data['point_clouds'].astype(np.float32)[objectness_label != 0])
        #
        # s_pc_path = os.path.join(s_root, object_name[:-4] + ".npy")
        # np.save(s_pc_path, input_data['point_clouds'].astype(np.float32))
        # exit(1)


if __name__ == "__main__":
    print("STart")
    infer_grasp_whole_scence()
    print("End")
