# Normal-consistency loss, following VEGS (Hwang et al., ECCV 2024,
# https://github.com/deepshwang/vegs).

from utils.graphics_utils import cam_normal_to_world_normal, quaternion_to_matrix


def loss_normal_guidance(viewpoint_cam, cov_quat, cov_scale, axis_weight=0.8):
    """L_normal = axis_weight * L_axis + (1 - axis_weight) * L_scale.

    cov_quat / cov_scale are the per-pixel alpha-blended rotation (4, H, W) and
    scale (3, H, W) of the Gaussians rendered by the modified rasterizer. Each
    covariance axis is penalized by its alignment with the predicted surface
    normal, so that the normal becomes the Gaussian's thinnest axis.
    """
    norm_pred = viewpoint_cam.original_normal

    cov_scale = cov_scale.permute(1, 2, 0).reshape(-1, 1, 3).contiguous()   # n_pix, 1, 3
    cov_quat = cov_quat.permute(1, 2, 0).reshape(-1, 4).contiguous()
    cov_rot = quaternion_to_matrix(cov_quat)                                 # n_pix, 3, 3
    cov_rs = cov_rot.detach() * cov_scale                                    # n_pix, 3, 3

    # Off-the-shelf (camera-space) normal to world space
    norm_pred_world = cam_normal_to_world_normal(norm_pred, viewpoint_cam.R)
    norm_pred_world = norm_pred_world.permute(1, 2, 0).reshape(-1, 3).contiguous()
    norm_pred_world = norm_pred_world[:, :, None].repeat(1, 1, 3)            # n_pix, 3, 3

    loss_axis = (cov_rot * norm_pred_world).sum(dim=-2).abs().mean()
    loss_scale = (cov_rs * norm_pred_world).sum(dim=-2).abs().mean()
    return axis_weight * loss_axis + (1 - axis_weight) * loss_scale
