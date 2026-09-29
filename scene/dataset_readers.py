#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from datetime import datetime
import torch

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    #### emjay
    normal: np.array
    normal_path: str
    K: np.array
    ####
    # #### cvpr
    # depth: np.array = None
    # depth_weight: np.array = None
    # depthloss: float=1e5
    # ####


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)    
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"


        image_path = os.path.join(images_folder, os.path.basename(extr.name.replace("\\",'/')))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readKMjackalCameras(cam_extrinsics, cam_intrinsics, images_folder, normal_folder, depth_model_type, pcd):

    cam_infos = []

    # #### cvpr
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # if depth_model_type=="zoe":
    #     # model_zoe = torch.hub.load("./ZoeDepth", "ZoeD_NK", source="local", pretrained=True).to('cuda')
    #     # source_depth = model_zoe.infer_pil(image.convert("RGB"))
    #     repo = "isl-org/ZoeDepth"
    #     model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
    #     depth_model = model_zoe_n.to(device)
    # elif depth_model_type=="metric3d":
    #     model_metric3d = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
    #     depth_model = model_metric3d.to(device)
    # ####


    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
   
        # K = np.array([
        #     [intr.params[0],0., intr.params[2]],
        #     [0., intr.params[1], intr.params[3]], 
        #     [0.,0.,1.]
        #         ])

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = focal_length_x
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            cx = intr.params[1]
            cy = intr.params[2]
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            cx = intr.params[2]
            cy = intr.params[3]
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"


        K = np.array([
            [focal_length_x,0., cx],
            [0., focal_length_y, cy], 
            [0.,0.,1.]
            ])


        image_path = os.path.join(images_folder, os.path.basename(extr.name.replace("\\",'/')))
        # image_name = os.path.basename(image_path).split(".png")[0]
        image_name,ext = os.path.splitext(os.path.basename(image_path))
        image = Image.open(image_path)


        #### normal 
        if depth_model_type == 'None':
            normal_path = ""
        else:
            if extr.name.find('.jpg') > -1:
                normal_path = os.path.join(normal_folder, os.path.basename(extr.name.replace('.jpg', '.npy')))
            else:
                normal_path = os.path.join(normal_folder, os.path.basename(extr.name.replace('.png', '.npy')))

        normal = None
        ####

        # #### depth
        # depthmap, depth_weight = None, None
        # resolution = 1

        # depthmap, depth_weight = np.zeros((height//resolution,width//resolution)), np.zeros((height//resolution,width//resolution))
        # # K = np.array([[focal_length_x, 0, width//resolution/2],[0,focal_length_y,height//resolution/2],[0,0,1]])
        # cam_coord = np.matmul(K, np.matmul(R.transpose(), pcd.points.transpose()) + T.reshape(3,1)) ### for coordinate definition, see getWorld2View2() function
        # valid_idx = np.where(np.logical_and.reduce((cam_coord[2]>0, cam_coord[0]/cam_coord[2]>=0, cam_coord[0]/cam_coord[2]<=width//resolution-1, cam_coord[1]/cam_coord[2]>=0, cam_coord[1]/cam_coord[2]<=height//resolution-1)))[0]
        # pts_depths = cam_coord[-1:, valid_idx]
        # cam_coord = cam_coord[:2, valid_idx]/cam_coord[-1:, valid_idx]
        # depthmap[np.round(cam_coord[1]).astype(np.int32).clip(0,height//resolution-1), np.round(cam_coord[0]).astype(np.int32).clip(0,width//resolution-1)] = pts_depths
        # # depth_weight[np.round(cam_coord[1]).astype(np.int32).clip(0,height//resolution-1), np.round(cam_coord[0]).astype(np.int32).clip(0,width//resolution-1)] = 1/pcd.errors[valid_idx] if pcd.errors is not None else 1
        # depth_weight[np.round(cam_coord[1]).astype(np.int32).clip(0,height//resolution-1), np.round(cam_coord[0]).astype(np.int32).clip(0,width//resolution-1)] = 1
        # depth_weight = depth_weight/depth_weight.max()
        # depthloss = 0.

        # source_depth = predict_depth(depth_model, depth_model_type, np.asarray(image))

        # target=depthmap.copy()           
        # target=((target != 0) * 255).astype(np.uint8)

        # depthmap, depthloss = optimize_depth(source=source_depth.cuda(), target=depthmap, mask=depthmap>0.0, depth_weight=depth_weight)


        # import cv2
        # from render import depth_colorize_with_mask
        # source, refined = depth_colorize_with_mask(source_depth.cpu().numpy()[None,:,:],dmindmax=(0.0,5.0)).squeeze(), \
        #                   depth_colorize_with_mask(depthmap[None,:,:], dmindmax=(20.0, 130.0)).squeeze() 
        

        # cv2.imwrite(f"debug/{idx:03d}_source.png", (source[:,:,::-1]*255).astype(np.uint8))
        # cv2.imwrite(f"debug/{idx:03d}_refined.png", (refined[:,:,::-1]*255).astype(np.uint8))
        # cv2.imwrite(f"debug/{idx:03d}_target.png", target)
        ####


        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, 
                              width=width, height=height, 
                              normal_path=normal_path, normal=normal,
                              K=K)
                            #   depth=depthmap, depth_weight=depth_weight,depthloss=depthloss)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos
    
# def predict_depth(depth_model, depth_model_type, img):
#     if depth_model_type=="zoe":
#         source_depth = depth_model.infer_pil(Image.fromarray(img.astype(np.uint8)),
#                                             output_type='tensor')
#     elif depth_mode_type=="metric3d":        
#         # # keep ratio resize
#         input_size = (616, 1064) # for vit model
#         # input_size = (544, 1216) # for convnext model
#         h, w = img.shape[:2]
#         scale = min(input_size[0] / h, input_size[1] / w)
#         rgb = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
#         # padding to input_size
#         padding = [123.675, 116.28, 103.53]
#         h, w = rgb.shape[:2]
#         pad_h = input_size[0] - h
#         pad_w = input_size[1] - w
#         pad_h_half = pad_h // 2
#         pad_w_half = pad_w // 2
#         rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
#         pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

#         ### normalize
#         mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
#         std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
#         rgb = torch.from_numpy(np.copy(img).transpose((2, 0, 1))).float()
#         rgb = torch.div((rgb - mean), std)
#         rgb = rgb[None, :, :, :].cuda()

#         print(img.shape)
#         # print(rgb.shape)
#         # image = torch.from_numpy(img/255.).unsqueeze(0)
#         pred_depth,_,_ = depth_model.inference({'input':rgb})
#         pred_depth = pred_depth.squeeze()
#         pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
        
#         # upsample to original size
#         source_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], img.shape[:2], mode='bilinear').squeeze()

#     return source_depth

# def optimize_depth(source, target, mask, depth_weight, prune_ratio=0.001):
#     """
#     Arguments
#     =========
#     source: np.array(h,w)
#     target: np.array(h,w)
#     mask: np.array(h,w):
#         array of [True if valid pointcloud is visible.]
#     depth_weight: np.array(h,w):
#         weight array at loss.
#     Returns
#     =======
#     refined_source: np.array(h,w)
#         literally "refined" source.
#     loss: float
#     """
#     # source = torch.from_numpy(source).cuda()
#     target = torch.from_numpy(target).cuda()
#     mask = torch.from_numpy(mask).cuda()
#     depth_weight = torch.from_numpy(depth_weight).cuda()

#     # Prune some depths considered "outlier"     
#     with torch.no_grad():
#         target_depth_sorted = target[target>1e-7].sort().values
#         min_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*prune_ratio)]
#         max_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*(1.0-prune_ratio))]

#         mask2 = target > min_prune_threshold
#         mask3 = target < max_prune_threshold
#         mask = torch.logical_and( torch.logical_and(mask, mask2), mask3)

#     source_masked = source[mask]
#     target_masked = target[mask]
#     depth_weight_masked = depth_weight[mask]
#     # tmin, tmax = target_masked.min(), target_masked.max()

#     # # Normalize
#     # target_masked = target_masked - tmin 
#     # target_masked = target_masked / (tmax-tmin)

#     scale = torch.ones(1).cuda().requires_grad_(True)
#     shift = (torch.ones(1) * 0.5).cuda().requires_grad_(True)

#     optimizer = torch.optim.Adam(params=[scale, shift], lr=1.0)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))
#     loss = torch.ones(1).cuda() * 1e5

#     iteration = 1
#     loss_prev = 1e6
#     loss_ema = 0.0
    
#     while abs(loss_ema - loss_prev) > 1e-5:
#         source_hat = scale*source_masked + shift
#         loss = torch.mean(((target_masked - source_hat)**2)*depth_weight_masked)

#         # penalize depths not in [0,1]
#         loss_hinge1 = loss_hinge2 = 0.0
#         if (source_hat<=0.0).any():
#             loss_hinge1 = 2.0*((source_hat[source_hat<=0.0])**2).mean()
#         # if (source_hat>=1.0).any():
#         #     loss_hinge2 = 0.3*((source_hat[source_hat>=1.0])**2).mean() 
        
#         loss = loss + loss_hinge1 + loss_hinge2

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
        
#         iteration+=1
#         if iteration % 1000 == 0:
#             print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
#             loss_prev = loss.item()
#         loss_ema = loss.item() * 0.2 + loss_ema * 0.8

#     loss = loss.item()
#     print(f"loss ={loss:10.5f}")

#     with torch.no_grad():
#         refined_source = (scale*source + shift) 
#     torch.cuda.empty_cache()
#     return refined_source.cpu().numpy(), loss

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)
    

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def refineColmapWithIndex(path,cam_extrinsics,cam_intrinsics):
    """ result
    'cam_extrinsics' and 'point3D.ply' contains the points observed in (at least 2) train-views 
    """
    
    # cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    # cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    # cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    # cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    try:
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        xyz, rgb, err = read_points3D_binary(bin_path)
    except:
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        xyz, rgb, err = read_points3D_text(txt_path)
    
    total_ptsidxlist = []
    for tidx, key in enumerate(sorted(cam_extrinsics, key=lambda x:cam_extrinsics[x].name)):
        total_ptsidxlist.append(cam_extrinsics[key].point3D_ids) # cam_extrinsics number starts from 1
    
    ### valid 2D points (select the points in train-view)
    ptsidx, cnt = np.unique(np.concatenate(total_ptsidxlist), return_counts=True) # for 2D points (extr.xys, extr.point3D_ids)
    
    valid_ptsidx = ptsidx[cnt>=2][1:] # 2view -> 3view: more restrict condition (COLMAP uses 3-view observed feature points)
    
    for tidx, key in enumerate(sorted(cam_extrinsics, key=lambda x:cam_extrinsics[x].name)):
        cam_valid = np.isin(cam_extrinsics[key].point3D_ids, valid_ptsidx)
        cam_extrinsics[key] = cam_extrinsics[key]._replace(point3D_ids = cam_extrinsics[key].point3D_ids[cam_valid],
                                                                 xys =  cam_extrinsics[key].xys[cam_valid])

    ### valid 3D points (removing the points only detected in 1 camera)
    ptsidx, cnt = np.unique(np.concatenate(total_ptsidxlist), return_counts=True) # for 3D points (xyz, rgb, err from points3D.bin)
    valid_totalptsidx = ptsidx[cnt>=2][1:] # remove invalid(-1) pts
    xyz = xyz[1:]
    rgb = rgb[1:]
    err = err[1:]
    assert len(valid_totalptsidx)==len(xyz), f"Lengths of valid_totalptsidx ({len(valid_totalptsidx)}) and xyz ({len(xyz)} do not match.)"
    
    valid3didx = np.isin(valid_totalptsidx, valid_ptsidx) # select the points seen from train-view
    xyz = xyz[valid3didx]
    rgb = rgb[valid3didx]
    err = err[valid3didx]
    
    ### save in ply format
    os.makedirs(os.path.join(path, 'plydummy'), exist_ok=True)
    ply_path = os.path.join(path, 'plydummy', f"points3D_{int(datetime.now().timestamp())}.ply") # k-shot train
    storePly(ply_path, xyz, rgb)
    
    return ply_path, cam_extrinsics, cam_intrinsics


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readKMjackalSceneInfo(path, images, eval, depth_model_type=None, llffhold=8):

    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)


    ### refineColmapWithIndex() remove the cameras and features except the train set
    try:
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        xyz, rgb, err = read_points3D_binary(bin_path)
    except:
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        xyz, rgb, err = read_points3D_text(txt_path)        
    # os.makedirs(os.path.join(path, 'plydummy'), exist_ok=True)
    # ply_path = os.path.join(path, 'plydummy', f"points3D_{int(datetime.now().timestamp())}.ply") # k-shot train
    # storePly(ply_path, xyz, rgb)    
    # # ply_path, cam_extrinsics, cam_intrinsics = refineColmapWithIndex(path, cam_extrinsics, cam_intrinsics)    
    # ### making pcd with the features captured from train_cam
    # pcd = fetchPly(ply_path)


    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None



    reading_dir = "images" if images == None else images
    reding_normal_dir = "normals"
    # cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos_unsorted = readKMjackalCameras(cam_extrinsics=cam_extrinsics, 
                                            cam_intrinsics=cam_intrinsics, 
                                            images_folder=os.path.join(path, reading_dir), 
                                            normal_folder=os.path.join(path,reding_normal_dir),
                                            depth_model_type=depth_model_type,
                                            pcd=pcd)

    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # split
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    ### [4] Calculate normalizations ###
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # if not os.path.exists(ply_path):
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     try:
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #     except:
    #         xyz, rgb, _ = read_points3D_text(txt_path)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    ### [5] Set ply storage path
    # ply_dir = ".cache"
    # os.makedirs(ply_dir, exist_ok=True)
    # ply_name = f"points3d_{ply_path.split('/data/')[-1].split('/')[0]}.ply"
    # ply_path = os.path.join(ply_dir, ply_name)
    
    # if not os.path.exists(ply_path):
    #     disp_name = os.path.join("data_3d_semantics", "train", seq, "gaussians", ply_name)
    #     print(f"Setting up {disp_name}. Will happen only the first time you open the scene / frame segement.")
    #     stored_pcd = storePly(ply_path, pcd.points, np.uint8(pcd.colors * 255))
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    

    return scene_info 


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "KMjackal": readKMjackalSceneInfo
}