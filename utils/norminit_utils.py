import copy
import math

import torch
from tqdm import tqdm

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import normal_to_rot, cam_normal_to_world_normal, standardize_quaternion, \
    matrix_to_quaternion, quaternion_to_matrix

import cv2


@torch.no_grad()
def initialize_gaussians_with_normals(gaussians, scene, pipe, background):
    print(__name__)

    # validataion
    ptr_gs_rot = gaussians._rotation.data_ptr
    ptr_gs_scale = gaussians._scaling.data_ptr

    viewpoint_stack = scene.getTrainCameras()
    n_cameras = len(viewpoint_stack)
    quaternion_new = copy.deepcopy(gaussians._rotation)

    with tqdm(range(n_cameras)) as pbar:
        pbar.set_description("initialize with normal prediction")
        for i in pbar:
            # estimate rotations from normals ---------------------------
            viewpoint_cam = viewpoint_stack[i]
            norm_pred = viewpoint_cam.original_normal
            _, H, W = norm_pred.shape

            norm_pred_world = cam_normal_to_world_normal(norm_pred, viewpoint_cam.R)
            rot_from_norm = normal_to_rot(
                norm_pred_world.permute(1, 2, 0).reshape(-1, 3))  # world2newworld      # n_pix, 3, 3

            quat_from_norm = matrix_to_quaternion(rot_from_norm)
            quat_from_norm = standardize_quaternion(quat_from_norm)

            # match 2d / 3d normal -----------------------
            scaling_modifier = 1.0
            # Set up rasterization configuration
            tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_cam.image_height),
                image_width=int(viewpoint_cam.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=background,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_cam.world_view_transform,
                projmatrix=viewpoint_cam.full_proj_transform,
                sh_degree=gaussians.active_sh_degree,
                campos=viewpoint_cam.camera_center,
                prefiltered=False,
                debug=pipe.debug
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            visibility_mark = rasterizer.markVisible(gaussians.get_xyz)

            visible_xyz = (gaussians.get_xyz[visibility_mark]).unsqueeze(dim=-1)
            R = torch.from_numpy(viewpoint_cam.R).transpose(-1, -2).to(device=visible_xyz.device).type_as(
                visible_xyz)  # world 2 cam
            T = torch.from_numpy(viewpoint_cam.T).to(device=visible_xyz.device).type_as(visible_xyz)  # world 2 cam
            K = torch.from_numpy(viewpoint_cam.K).to(device=visible_xyz.device).type_as(visible_xyz)
            visible_xyz_cam = ((R @ visible_xyz) + T[None, :, None])
            pix = (K @ visible_xyz_cam).squeeze()
            pix /= pix[:, -1:]

            # -1 ~ 1
            pix[:, 0] = (pix[:, 0] * 2 - W) / W
            pix[:, 1] = (pix[:, 1] * 2 - H) / H

            quat_init = torch.nn.functional.grid_sample(quat_from_norm.permute(1, 0).reshape(1, -1, H, W),
                                                        pix[None, None, :, :-1], mode='nearest', align_corners=True)
            quat_init = standardize_quaternion(quat_init)

            mask_zero_quat = (quat_init.abs().squeeze().sum(dim=-2) < 1e-9)
            quaternion_new[visibility_mark] = (quat_init.squeeze().permute(1, 0)) * ~mask_zero_quat[:, None] + \
                                              quaternion_new[visibility_mark] * mask_zero_quat[:, None]  # exclude zeros

    quaternion_new = standardize_quaternion(quaternion_new)
    quaternion_new /= (torch.linalg.norm(quaternion_new, axis=-1, keepdims=True) + 1e-9)

    gaussians._rotation.data[:, 0] = copy.deepcopy(quaternion_new[:, 0].type_as(gaussians._rotation))
    gaussians._rotation.data[:, 1] = copy.deepcopy(quaternion_new[:, 1].type_as(gaussians._rotation))
    gaussians._rotation.data[:, 2] = copy.deepcopy(quaternion_new[:, 2].type_as(gaussians._rotation))
    gaussians._rotation.data[:, 3] = copy.deepcopy(quaternion_new[:, 3].type_as(gaussians._rotation))

    gaussians._scaling[:, 0] = copy.deepcopy(torch.log(torch.tensor([1e-5]).type_as(gaussians._scaling)))
    gaussians._scaling[:, 1] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))
    gaussians._scaling[:, 2] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))

    assert gaussians._rotation.data_ptr == ptr_gs_rot
    assert gaussians._scaling.data_ptr == ptr_gs_scale

    return gaussians


def sort_quaternion_candidates(quaternion_full):
    n_pnt_full = quaternion_full.shape[0]
    n_pnt_batch = 512

    for i in range(math.ceil(n_pnt_full / 512)):
        quaternion_full_subset = quaternion_full[i * n_pnt_batch: min((i + 1) * n_pnt_batch, n_pnt_full)]
        quaternion_sim = quaternion_full_subset @ quaternion_full_subset.permute(0, 2, 1)
        idx_best_normal = torch.argsort(quaternion_sim.sum(dim=2), dim=1, descending=True)
        quaternion_full[i * n_pnt_batch: min((i + 1) * n_pnt_batch, n_pnt_full)] = torch.gather(quaternion_full_subset,
                                                                                                index=idx_best_normal[:,
                                                                                                      :, None].repeat(1,
                                                                                                                      1,
                                                                                                                      4),
                                                                                                dim=1)
    return quaternion_full


def refresh_quaternion_new(quaternion_new, idx_accum_quat):
    max_memory = idx_accum_quat.max().item()
    new_memory = int(max_memory * 0.7)

    # check mem full
    mask_full = (idx_accum_quat >= new_memory)

    quaternion_full = quaternion_new[mask_full]
    quaternion_full = sort_quaternion_candidates(quaternion_full)
    quaternion_new[mask_full] = quaternion_full
    idx_accum_quat[mask_full] = new_memory

    return quaternion_new, idx_accum_quat


def best_quaternion_new(quaternion_new, idx_accum_quat):
    max_memory = quaternion_new.shape[1]
    for n_accum in idx_accum_quat.unique():
        if n_accum == 0:
            continue
        # aaa = (idx_accum_quat == n_accum)
        # indices = torch.nonzero(aaa, as_tuple=True)[0]
        mask = (idx_accum_quat == n_accum)[:, None].repeat(1, max_memory)
        # print("n_accum:", n_accum, " indices:", indices)
        mask[:, n_accum:] = False
        quaternion_accum = quaternion_new[mask].reshape(-1, n_accum, 4)
        quaternion_accum = sort_quaternion_candidates(quaternion_accum)
        quaternion_new[mask] = quaternion_accum.reshape(-1, 4)

    return quaternion_new[:, 0, :]


@torch.no_grad()
def initialize_gaussians_with_window_normals(gaussians, scene, pipe, background):
    print(__name__)

    # validataion
    ptr_gs_rot = gaussians._rotation.data_ptr
    ptr_gs_scale = gaussians._scaling.data_ptr

    viewpoint_stack = scene.getTrainCameras()
    n_cameras = len(viewpoint_stack)

    max_memory = 130
    quaternion_new = copy.deepcopy(gaussians._rotation).reshape(-1, 1, 4).repeat(1, max_memory,
                                                                                 1)  # consider max 100 normal at once
    n_pnt = quaternion_new.shape[0]
    idx_accum_quat = torch.zeros((n_pnt)).to(quaternion_new.device).to(torch.int64)

    condition = (gaussians.get_xyz[:, 0] > 0.3067) & (gaussians.get_xyz[:, 0] < 0.3069) & (
                gaussians.get_xyz[:, 1] > 0.7380) & (gaussians.get_xyz[:, 1] < 0.7382) & (
                            gaussians.get_xyz[:, 2] > -0.5243) & (gaussians.get_xyz[:, 2] < -0.5241)
    B_indices = torch.nonzero(condition, as_tuple=True)[0]

    with tqdm(range(n_cameras)) as pbar:
        pbar.set_description("initialize with normal prediction")
        for i in pbar:
            # print("image name:", viewpoint_stack[i].image_path)
            # estimate rotations from normals ---------------------------
            viewpoint_cam = viewpoint_stack[i]
            norm_pred = viewpoint_cam.original_normal
            _, H, W = norm_pred.shape

            norm_pred_world = cam_normal_to_world_normal(norm_pred, viewpoint_cam.R)
            rot_from_norm = normal_to_rot(
                norm_pred_world.permute(1, 2, 0).reshape(-1, 3))  # world2newworld      # n_pix, 3, 3

            quat_from_norm = matrix_to_quaternion(rot_from_norm)
            quat_from_norm = standardize_quaternion(quat_from_norm)

            # match 2d / 3d normal -----------------------
            scaling_modifier = 1.0
            # Set up rasterization configuration
            tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_cam.image_height),
                image_width=int(viewpoint_cam.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=background,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_cam.world_view_transform,
                projmatrix=viewpoint_cam.full_proj_transform,
                sh_degree=gaussians.active_sh_degree,
                campos=viewpoint_cam.camera_center,
                prefiltered=False,
                debug=pipe.debug
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            visibility_mark = rasterizer.markVisible(gaussians.get_xyz)  # visibility_mark.shape: torch.Size([2370245])

            visible_xyz = (gaussians.get_xyz[visibility_mark]).unsqueeze(dim=-1)
            # condition = (visible_xyz[:,0] > 0.26) & (visible_xyz[:,0] < 0.35) & (visible_xyz[:,1] > 0.69) & (visible_xyz[:,1] < 0.78) & (visible_xyz[:,2] > -0.57) & (visible_xyz[:,2] < -0.47)

            # num = 349
            all_condition = (gaussians.get_xyz[:, 0] > 0.5) & (gaussians.get_xyz[:, 0] < 1.) & (
                        gaussians.get_xyz[:, 1] > .50) & (gaussians.get_xyz[:, 1] < 1.9) & (
                                        gaussians.get_xyz[:, 2] > 0.0) & (gaussians.get_xyz[:, 2] < 3.01)
            all_B_indices = torch.nonzero(all_condition, as_tuple=True)[0]
            condition = (visible_xyz[:, 0] > 0.5) & (visible_xyz[:, 0] < 1.) & (visible_xyz[:, 1] > .50) & (
                        visible_xyz[:, 1] < 1.9) & (visible_xyz[:, 2] > 0.0) & (visible_xyz[:, 2] < 3.01)
            B_indices = torch.nonzero(condition, as_tuple=True)[0]

            # print("visible_xyz_cam[346]:", visible_xyz[346])
            R = torch.from_numpy(viewpoint_cam.R).transpose(-1, -2).to(device=visible_xyz.device).type_as(
                visible_xyz)  # world 2 cam
            T = torch.from_numpy(viewpoint_cam.T).to(device=visible_xyz.device).type_as(visible_xyz)  # world 2 cam
            K = torch.from_numpy(viewpoint_cam.K).to(device=visible_xyz.device).type_as(visible_xyz)
            visible_xyz_cam = ((R @ visible_xyz) + T[None, :, None])
            pix = (K @ visible_xyz_cam).squeeze()
            pix /= pix[:, -1:]

            selected_pt = pix[B_indices]
            # selected_pt = pix
            # print("pix size:", pix.shape, " selected_pt:", selected_pt)
            # Mask for columns where all elements are > 0
            # TODO: 아래 코드 수정 필요
            """
            img_idx = int(viewpoint_stack[i].image_path.split('/')[-1].split('.jpg')[0])
            # if (img_idx > 49 and img_idx < 69) or (img_idx > 110 and img_idx <115):
            o_W = 2 * int(K[0][2])
            o_H = 2 * int(K[1][2])
            selected_pt[:, 0] = (selected_pt[:, 0] / o_W) * W
            selected_pt[:, 1] = (selected_pt[:, 1] / o_H) * H
            # image_np = 255 * viewpoint_cam.original_image.permute(1, 2, 0).cpu().numpy()
            print("initial original")
            # pix_mask = (pix > 0).all(dim=1)  # Keeps columns where all rows are > 0
            # Apply the mask to keep only valid columns
            # pixel_check = pix[pix_mask,:]
            # mask11 = (pix[:,0] > torch.round((selected_pt[0])-0.5)) & (pix[:,0] < torch.round(selected_pt[0]+0.5)) & (pix[:,1] > torch.round((selected_pt[1])-0.5)) & (pix[:,1] < torch.round((selected_pt[1])+0.5))

            img = cv2.imread(viewpoint_stack[i].image_path)
            image = cv2.resize(img, (W, H))
            for hhh in range(0, selected_pt.shape[0]):
                # if int(selected_pt[hhh][1]) > 875 and int(selected_pt[hhh][0]) > 880 and int(selected_pt[hhh][0]) < 1000:
                cv2.circle(image, (int(selected_pt[hhh][0]), int(selected_pt[hhh][1])), 5, (255, 0, 0), -1,
                           cv2.LINE_AA)
                # if  (visible_xyz_cam[hhh][0] > 0.17) & (visible_xyz_cam[hhh][0] < 0.21) & (visible_xyz_cam[hhh][1] > 1.50) & (visible_xyz_cam[hhh][1] < 1.9) & (visible_xyz_cam[hhh][2] > 1.0) & (visible_xyz_cam[hhh][2] < 1.3):
                #     print("visible_xyz_cam: ", visible_xyz_cam[hhh])

            # cv2.circle(image, (1439, 790), 5, (0, 255, 0), -1,cv2.LINE_AA)
            temp_pt = pix
            # tt = temp_pt[4:5,:]
            # print("pix shape:", pix.shape, ", tt shape:", tt.shape)
            temp_pt[:, 0] = (temp_pt[:, 0] / o_W) * W
            temp_pt[:, 1] = (temp_pt[:, 1] / o_H) * H
            temp_index = (temp_pt[:, 0] > 1438.0) & (temp_pt[:, 0] < 1440.0) & (temp_pt[:, 1] < 791)
            temp_indices = torch.nonzero(temp_index, as_tuple=True)[0]

            # cv2.circle(img, (int(pixel_check[0][0]), int(pixel_check[0][1])), 5, (255, 0, 0), -1,cv2.LINE_AA)
            # cv2.circle(img, (int(pixel_check[340][0]), int(pixel_check[340][1])), 5, (0, 255, 0), -1,cv2.LINE_AA)
            # cv2.circle(img, (int(selected_pt[0]), int(selected_pt[1])), 5, (0, 255, 0), -1,cv2.LINE_AA)
            # cv2.imwrite('./result/temp_0/' + str(img_idx) + '.jpg', image) # 원본이 저장됨
            # B11 = pix[mask11,:]
            # B_indices = torch.nonzero(mask11, as_tuple=True)[0]

            # -1 ~ 1
            pix[:, 0] = (pix[:, 0] * 2 - W) / W
            pix[:, 1] = (pix[:, 1] * 2 - H) / H

            quat_init = torch.nn.functional.grid_sample(quat_from_norm.permute(1, 0).reshape(1, -1, H, W),
                                                        pix[None, None, :, :-1], mode='nearest', align_corners=True)
            quat_init = standardize_quaternion(quat_init)  # torch.Size([1, 4, 1, n_vis])

            mask_zero_quat = (quat_init.abs().squeeze().sum(dim=-2) < 1e-9)  # torch.Size([n_vis])
            quat_init_valid = quat_init[:, :, :, ~mask_zero_quat].squeeze().permute(1,
                                                                                    0)  # -> quat_init_valid.shape: torch.Size([n_vis_val,4])
            mask3d_visible_valid = visibility_mark.clone()
            mask3d_visible_valid[visibility_mark] *= ~mask_zero_quat  # torch.Size([n_pnt])

            quaternion_new[mask3d_visible_valid] = torch.scatter(quaternion_new[mask3d_visible_valid], dim=1,
                                                                 index=idx_accum_quat[mask3d_visible_valid][:, None,
                                                                       None].repeat(1, 1, 4),
                                                                 src=quat_init_valid[:, None, :])

            if mask3d_visible_valid[all_B_indices[0]] and img_idx == 349:
                print("img_id:", img_idx, " idx_accum_quat:", idx_accum_quat, " quat:",
                      quaternion_new[all_B_indices[0], :, :])

            # if mask3d_visible_valid[11393]:
            #     print("img_id:", img_idx, " idx_accum_quat[11393]:", idx_accum_quat[11393], " quat:", quaternion_new[11393,:,:])

            # idx_accum_quat[visibility_mark] +=1
            idx_accum_quat[mask3d_visible_valid] += 1

            # if idx_accum_quat[11393] > 0:
            #     print("img_id:", img_idx)

            if idx_accum_quat.max().item() == max_memory:
                quaternion_new, idx_accum_quat = refresh_quaternion_new(quaternion_new, idx_accum_quat)

            
            """
    print("index:", all_B_indices[0], " => quaternion_new => ", quaternion_new[all_B_indices[0], 0, :])

    quaternion_new = best_quaternion_new(quaternion_new, idx_accum_quat)
    quaternion_new = standardize_quaternion(quaternion_new)
    quaternion_new /= (torch.linalg.norm(quaternion_new, axis=-1, keepdims=True) + 1e-9)

    print("index:", all_B_indices[0], " =>  final quaternion_new => ", quaternion_new[all_B_indices[0], :])

    gaussians._rotation.data[:, 0] = copy.deepcopy(quaternion_new[:, 0].type_as(gaussians._rotation))
    gaussians._rotation.data[:, 1] = copy.deepcopy(quaternion_new[:, 1].type_as(gaussians._rotation))
    gaussians._rotation.data[:, 2] = copy.deepcopy(quaternion_new[:, 2].type_as(gaussians._rotation))
    gaussians._rotation.data[:, 3] = copy.deepcopy(quaternion_new[:, 3].type_as(gaussians._rotation))

    none_val_mask = (quaternion_new[:, 0] == 1)
    # gaussians._scaling[~none_val_mask, 0] = copy.deepcopy(torch.log(torch.tensor([1e-5]).type_as(gaussians._scaling)))
    # gaussians._scaling[~none_val_mask, 1] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))
    # gaussians._scaling[~none_val_mask, 2] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))
    # # aa = torch.nonzero(none_val_mask, as_tuple=True)[0]
    # # distance base
    # neg_val_mask = (gaussians._scaling[:, 0] < 0)
    # # # keep small
    # neg_selected = neg_val_mask & ~none_val_mask
    # gaussians._scaling[neg_selected, 0] = copy.deepcopy(torch.log(torch.tensor([1e-5]).type_as(gaussians._scaling)))
    # gaussians._scaling[neg_selected, 1] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))
    # gaussians._scaling[neg_selected, 2] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))

    pos_val_mask = (gaussians._scaling[:, 0] >= 0)
    # # keep small
    pos_selected = pos_val_mask & ~none_val_mask
    gaussians._scaling[pos_selected, 0] = copy.deepcopy(torch.log(torch.tensor([1e-5]).type_as(gaussians._scaling)))
    gaussians._scaling[pos_selected, 1] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))
    gaussians._scaling[pos_selected, 2] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))
    gaussians._scaling[~pos_selected, 0] = copy.deepcopy(torch.log(torch.tensor([1e-5]).type_as(gaussians._scaling)))
    gaussians._scaling[~pos_selected, 1] = copy.deepcopy(gaussians._scaling[~pos_selected, 1])
    gaussians._scaling[~pos_selected, 2] = copy.deepcopy(gaussians._scaling[~pos_selected, 2])

    # pos_selected = ~neg_val_mask & ~none_val_mask
    # gaussians._scaling[~pos_selected, 0] = copy.deepcopy(torch.log(torch.tensor([1e-5]).type_as(gaussians._scaling)))
    # gaussians._scaling[~pos_selected, 1] = copy.deepcopy(gaussians._scaling[~pos_selected,1])
    # gaussians._scaling[~pos_selected, 2] = copy.deepcopy(gaussians._scaling[~pos_selected,2])

    # neg_val_mask = (quaternion_new[:,0] < 0)
    # # # gaussians._scaling[neg_val_mask, 0] = copy.deepcopy(gaussians._scaling[neg_val_mask,0]*10)
    # gaussians._scaling[neg_val_mask, 0] = copy.deepcopy(torch.log(torch.tensor([1e-5]).type_as(gaussians._scaling)))
    # gaussians._scaling[neg_val_mask, 1] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))
    # gaussians._scaling[neg_val_mask, 2] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))

    # # # gaussians._scaling[~neg_val_mask, 0] = copy.deepcopy(gaussians._scaling[~neg_val_mask,0]*0.01)
    # gaussians._scaling[~neg_val_mask, 0] = copy.deepcopy(torch.log(torch.tensor([1e-5]).type_as(gaussians._scaling)))
    # gaussians._scaling[~neg_val_mask, 1] = copy.deepcopy(gaussians._scaling[:,1])
    # gaussians._scaling[~neg_val_mask, 2] = copy.deepcopy(gaussians._scaling[:,2])

    # # one_val_mask = (quaternion_new[:,0] == 1)
    # # # one_val_mask_indices = torch.nonzero(one_val_mask, as_tuple=True)[0]
    # # gaussians._scaling[~one_val_mask, 0] = copy.deepcopy(torch.log(torch.tensor([1e-5]).type_as(gaussians._scaling)))
    # # gaussians._scaling[~one_val_mask, 1] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))
    # # gaussians._scaling[~one_val_mask, 2] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))

    # gaussians._scaling[:, 0] = copy.deepcopy(torch.log(torch.tensor([1e-5]).type_as(gaussians._scaling)))
    # gaussians._scaling[:, 1] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))
    # gaussians._scaling[:, 2] = copy.deepcopy(torch.log(torch.tensor([1e-1]).type_as(gaussians._scaling)))

    assert gaussians._rotation.data_ptr == ptr_gs_rot
    assert gaussians._scaling.data_ptr == ptr_gs_scale

    return gaussians


def check_gaussians_with_projection(gaussians, viewpoint_cam, B_indices, iter, pipe, background):
    # quaternion_new = copy.deepcopy(gaussians._rotation).reshape(-1,1,4).repeat(1,max_memory,1) # consider max 100 normal at once
    # n_pnt = quaternion_new.shape[0]
    # idx_accum_quat = torch.zeros((n_pnt)).to(quaternion_new.device).to(torch.int64)

    # print("image name:", viewpoint_stack[i].image_path)
    # estimate rotations from normals ---------------------------
    norm_pred = viewpoint_cam.original_normal
    _, H, W = norm_pred.shape
    norm_pred_world = cam_normal_to_world_normal(norm_pred, viewpoint_cam.R)
    # rot_from_norm = normal_to_rot(norm_pred_world.permute(1,2,0).reshape(-1,3))  # world2newworld      # n_pix, 3, 3
    # quat_from_norm = matrix_to_quaternion(rot_from_norm)
    # quat_from_norm = standardize_quaternion(quat_from_norm)
    # match 2d / 3d normal -----------------------
    scaling_modifier = 1.0
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_cam.image_height),
        image_width=int(viewpoint_cam.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_cam.world_view_transform,
        projmatrix=viewpoint_cam.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_cam.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    visibility_mark = rasterizer.markVisible(
        gaussians.get_xyz[B_indices])  # visibility_mark.shape: torch.Size([2370245])

    visible_xyz = (gaussians.get_xyz[B_indices][visibility_mark]).unsqueeze(dim=-1)
    # print("visible_xyz_cam[346]:", visible_xyz[346])
    R = torch.from_numpy(viewpoint_cam.R).transpose(-1, -2).to(device=visible_xyz.device).type_as(
        visible_xyz)  # world 2 cam
    T = torch.from_numpy(viewpoint_cam.T).to(device=visible_xyz.device).type_as(visible_xyz)  # world 2 cam
    K = torch.from_numpy(viewpoint_cam.K).to(device=visible_xyz.device).type_as(visible_xyz)
    visible_xyz_cam = ((R @ visible_xyz) + T[None, :, None])
    pix = (K @ visible_xyz_cam).squeeze(dim=-1)
    pix /= pix[:, -1:].clone()

    o_W = 2 * int(K[0][2])
    o_H = 2 * int(K[1][2])
    pix[:, 0] = (pix[:, 0] / o_W) * W
    pix[:, 1] = (pix[:, 1] / o_H) * H

    # print("pix size:", pix.shape, " selected_pt:", selected_pt)
    # Mask for columns where all elements are > 0

    img_idx = int(viewpoint_cam.image_path.split('/')[-1].split('.jpg')[0])
    # pix_mask = (pix > 0).all(dim=1)  # Keeps columns where all rows are > 0
    # Apply the mask to keep only valid columns
    # pixel_check = pix[pix_mask,:]

    img = cv2.imread(viewpoint_cam.image_path)
    image = cv2.resize(img, (W, H))
    # image_np = viewpoint_cam.original_image.permute(1, 2, 0).cpu().numpy()

    for hhh in range(0, pix.shape[0]):
        cv2.circle(image, (int(pix[hhh][0]), int(pix[hhh][1])), 5, (255, 0, 0), -1, cv2.LINE_AA)
        if pix[hhh][1] >= W or pix[hhh][0] >= H:
            continue
        # norm_pred_world = cam_normal_to_world_normal(norm_pred, viewpoint_cam.R)
        tmp_rot_from_norm = normal_to_rot(
            norm_pred_world[:, int(pix[hhh][1]), int(pix[hhh][0])])  # world2newworld      # n_pix, 3, 3
        tmp_quat_from_norm = matrix_to_quaternion(tmp_rot_from_norm)
        tmp_quat_from_norm = standardize_quaternion(tmp_quat_from_norm)
        print("pix:[", int(pix[hhh][0]), ",", int(pix[hhh][1]), "] => original quat:",
              tmp_quat_from_norm.detach().cpu().numpy())

    # cv2.circle(img, (int(pixel_check[0][0]), int(pixel_check[0][1])), 5, (255, 0, 0), -1,cv2.LINE_AA)
    # cv2.circle(img, (int(pixel_check[340][0]), int(pixel_check[340][1])), 5, (0, 255, 0), -1,cv2.LINE_AA)
    cv2.imwrite('./result/temp_1/' + str(img_idx) + '_' + str(iter) + '.jpg', image)
    # B11 = pix[mask11,:]
    # B_indices = torch.nonzero(mask11, as_tuple=True)[0]
    return pix


def initialize_gaussians_projection_into_images(gaussians, scene, pipe, background):
    import numpy as np

    print(__name__)

    viewpoint_stack = scene.getTrainCameras()
    n_cameras = len(viewpoint_stack)

    with tqdm(range(n_cameras)) as pbar:
        pbar.set_description("start pcd projection")
        for i in pbar:
            # print("image name:", viewpoint_stack[i].image_path)
            # estimate rotations from normals ---------------------------
            viewpoint_cam = viewpoint_stack[i]

            # match 2d / 3d normal -----------------------
            scaling_modifier = 1.0
            # Set up rasterization configuration
            tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_cam.image_height),
                image_width=int(viewpoint_cam.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=background,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_cam.world_view_transform,
                projmatrix=viewpoint_cam.full_proj_transform,
                sh_degree=gaussians.active_sh_degree,
                campos=viewpoint_cam.camera_center,
                prefiltered=False,
                debug=pipe.debug
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            visibility_mark = rasterizer.markVisible(gaussians.get_xyz)  # visibility_mark.shape: torch.Size([2370245])

            visible_xyz = (gaussians.get_xyz[visibility_mark]).unsqueeze(dim=-1)

            # print("visible_xyz_cam[346]:", visible_xyz[346])
            R = torch.from_numpy(viewpoint_cam.R).transpose(-1, -2).to(device=visible_xyz.device).type_as(
                visible_xyz)  # world 2 cam
            T = torch.from_numpy(viewpoint_cam.T).to(device=visible_xyz.device).type_as(visible_xyz)  # world 2 cam
            K = torch.from_numpy(viewpoint_cam.K).to(device=visible_xyz.device).type_as(visible_xyz)
            visible_xyz_cam = ((R @ visible_xyz) + T[None, :, None])
            pix = (K @ visible_xyz_cam).squeeze()

            depths = pix[:, -1:].clone()
            # normalized_depths = (depths - depths.min()) / (depths.max() - depths.min())  # Normalize to [0, 1]
            normalized_depths = (((depths - depths.min()) / (depths.max() - depths.min())) * 255).to(torch.uint8)
            normalized_depths_np = normalized_depths.cpu().numpy().reshape(-1, 1)

            colormap = cv2.applyColorMap(normalized_depths_np, cv2.COLORMAP_JET)

            pix /= pix[:, -1:].clone()

            # Mask for columns where all elements are > 0
            # TODO: 이름 바꿔야 함
            """
            img_idx = int(viewpoint_stack[i].image_path.split('/')[-1].split('.jpg')[0])
            img = cv2.imread(viewpoint_stack[i].image_path)
            H, W, _ = img.shape

            image = cv2.resize(img, (W, H))
            o_W = 2 * int(K[0][2])
            o_H = 2 * int(K[1][2])
            pix[:, 0] = (pix[:, 0] / o_W) * W
            pix[:, 1] = (pix[:, 1] / o_H) * H

            for hhh in range(0, pix.shape[0]):
                # color = (int(255 * normalized_depths[hhh]), 0, int(255 * (1 - normalized_depths[hhh])))  # Blue for near, Red for far
                # color = tuple(colormap[hhh,0,:])  # Get the color for the current depth
                color = tuple(map(int, colormap[hhh, 0, :]))  # Convert to tuple of integers

                # print("hhh:", hhh, " color:", color)
                # cv2.circle(image, (int(pix[hhh][0]), int(pix[hhh][1])), 1, (255, 0, 0), -1,cv2.LINE_AA)
                cv2.circle(image, (int(pix[hhh][0]), int(pix[hhh][1])), 1, color=color, thickness=-1)

            cv2.imwrite('./result/pcd_projection/' + str(img_idx) + '.jpg', image)
            """

        print("pcd projection done!!")









