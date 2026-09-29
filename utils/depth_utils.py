
import torch
from scene.dataset_readers import fetchPly
import numpy as np
from PIL import Image
import cv2
import torch.nn.functional as F
import os

resolution = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_depth(scene):
    if scene.depth_model_type=="zoe":
        # model_zoe = torch.hub.load("./ZoeDepth", "ZoeD_NK", source="local", pretrained=True).to('cuda')
        # source_depth = model_zoe.infer_pil(image.convert("RGB"))
        repo = "isl-org/ZoeDepth"
        model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
        depth_model = model_zoe_n.to(device)
    elif scene.depth_model_type=="metric3d":
        model_metric3d = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
        depth_model = model_metric3d.to(device)


    scene.depth_model = depth_model
    return scene
    
def get_dense_depth(scene, cam_info):

    # pcd = fetchPly(ply_path)
    image_name,ext = os.path.splitext(os.path.basename(cam_info.image_path))
    completion_depth_name = 'depth_completion_' + scene.depth_model_type
    completion_depth_path = cam_info.image_path.replace('images',completion_depth_name).replace(ext, '.npy')

    if os.path.exists(completion_depth_path):
        depthmap = np.load(completion_depth_path)
        depth_weight = np.zeros((depthmap.shape[0]//resolution,depthmap.shape[1]//resolution)) 
        depth_weight[depthmap != 0] = depthmap[depthmap != 0]
        depth_weight = depth_weight/depth_weight.max()
    else:
        image = Image.open(cam_info.image_path)

        image = image.resize((cam_info.image_width, cam_info.image_height))


        depthmap, depth_weight = initialize_depth(cam_info, scene.point_cloud)

        source_depth = predict_depth(scene.depth_model, scene.depth_model_type, np.asarray(image))

        target=depthmap.copy()           
        target=((target != 0) * 255).astype(np.uint8)

        depthmap, depthloss = optimize_depth(source=source_depth.cuda(), target=depthmap, 
                                mask=depthmap>0.0, depth_weight=depth_weight)

    # import cv2
    # from render import depth_colorize_with_mask
    # source, refined = depth_colorize_with_mask(source_depth.cpu().numpy()[None,:,:],dmindmax=(0.0,5.0)).squeeze(), \
    #                     depth_colorize_with_mask(depthmap[None,:,:], dmindmax=(20.0, 130.0)).squeeze() 
    

    # cv2.imwrite(f"debug/{cam_info.image_name}_source.png", (source[:,:,::-1]*255).astype(np.uint8))
    # cv2.imwrite(f"debug/{cam_info.image_name}_refined.png", (refined[:,:,::-1]*255).astype(np.uint8))
    # cv2.imwrite(f"debug/{cam_info.image_name}_target.png", target)

    # #### cvpr
    gt_norm_depth = F.interpolate(torch.tensor(depthmap)[None,None], size=cam_info.resolution[::-1])[0,0] if depthmap is not None else None
    depth_weight = F.interpolate(torch.tensor(depth_weight)[None,None], size=cam_info.resolution[::-1])[0,0] if depth_weight is not None else None

    original_depth = gt_norm_depth.to(cam_info.data_device) if depthmap is not None else None
    original_depth_weight = depth_weight.to(cam_info.data_device) if depth_weight is not None else None
    ####

    # return depthmap, depth_weight, depthloss
    return original_depth

def initialize_depth(cam_info, pcd):

    # K = cam_info.K
    R = cam_info.R
    T = cam_info.T
    width = cam_info.image_width
    height = cam_info.image_height


    #### depth
    depthmap, depth_weight = None, None
    resolution = 1

    depthmap, depth_weight = np.zeros((height//resolution,width//resolution)), np.zeros((height//resolution,width//resolution))
    K = np.array([[cam_info.K[0][0], 0, cam_info.K[0][2]//resolution/2],[0,cam_info.K[1][1],cam_info.K[1][2]//resolution/2],[0,0,1]])
    cam_coord = np.matmul(K, np.matmul(R.transpose(), pcd.points.transpose()) + T.reshape(3,1)) ### for coordinate definition, see getWorld2View2() function
    valid_idx = np.where(np.logical_and.reduce((cam_coord[2]>0, cam_coord[0]/cam_coord[2]>=0, cam_coord[0]/cam_coord[2]<=width//resolution-1, cam_coord[1]/cam_coord[2]>=0, cam_coord[1]/cam_coord[2]<=height//resolution-1)))[0]
    pts_depths = cam_coord[-1:, valid_idx]
    cam_coord = cam_coord[:2, valid_idx]/cam_coord[-1:, valid_idx]
    depthmap[np.round(cam_coord[1]).astype(np.int32).clip(0,height//resolution-1), np.round(cam_coord[0]).astype(np.int32).clip(0,width//resolution-1)] = pts_depths
    # depth_weight[np.round(cam_coord[1]).astype(np.int32).clip(0,height//resolution-1), np.round(cam_coord[0]).astype(np.int32).clip(0,width//resolution-1)] = 1/pcd.errors[valid_idx] if pcd.errors is not None else 1
    depth_weight[np.round(cam_coord[1]).astype(np.int32).clip(0,height//resolution-1), np.round(cam_coord[0]).astype(np.int32).clip(0,width//resolution-1)] = 1
    depth_weight = depth_weight/depth_weight.max()

    return depthmap, depth_weight


def predict_depth(depth_model, depth_model_type, img):
    if depth_model_type=="zoe":
        source_depth = depth_model.infer_pil(Image.fromarray(img.astype(np.uint8)),
                                            output_type='tensor')
    elif depth_model_type=="metric3d":        
        # # keep ratio resize
        input_size = (616, 1064) # for vit model
        # input_size = (544, 1216) # for convnext model
        h, w = img.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        ### normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(np.copy(img).transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :].cuda()

        # print(img.shape)
        # print(rgb.shape)
        # image = torch.from_numpy(img/255.).unsqueeze(0)
        pred_depth,_,_ = depth_model.inference({'input':rgb})
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
        
        # upsample to original size
        source_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], img.shape[:2], mode='bilinear').squeeze()

    return source_depth

def optimize_depth(source, target, mask, depth_weight, prune_ratio=0.001):
    """
    Arguments
    =========
    source: np.array(h,w)
    target: np.array(h,w)
    mask: np.array(h,w):
        array of [True if valid pointcloud is visible.]
    depth_weight: np.array(h,w):
        weight array at loss.
    Returns
    =======
    refined_source: np.array(h,w)
        literally "refined" source.
    loss: float
    """
    # source = torch.from_numpy(source).cuda()
    target = torch.from_numpy(target).cuda()
    mask = torch.from_numpy(mask).cuda()
    depth_weight = torch.from_numpy(depth_weight).cuda()

    # Prune some depths considered "outlier"     
    with torch.no_grad():
        target_depth_sorted = target[target>1e-7].sort().values
        min_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*prune_ratio)]
        max_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*(1.0-prune_ratio))]

        mask2 = target > min_prune_threshold
        mask3 = target < max_prune_threshold
        mask = torch.logical_and( torch.logical_and(mask, mask2), mask3)

    source_masked = source[mask]
    target_masked = target[mask]
    depth_weight_masked = depth_weight[mask]
    # tmin, tmax = target_masked.min(), target_masked.max()

    # # Normalize
    # target_masked = target_masked - tmin 
    # target_masked = target_masked / (tmax-tmin)

    scale = torch.ones(1).cuda().requires_grad_(True)
    shift = (torch.ones(1) * 0.5).cuda().requires_grad_(True)

    optimizer = torch.optim.Adam(params=[scale, shift], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))
    loss = torch.ones(1).cuda() * 1e5

    iteration = 1
    loss_prev = 1e6
    loss_ema = 0.0
    
    while abs(loss_ema - loss_prev) > 1e-5:
        source_hat = scale*source_masked + shift
        loss = torch.mean(((target_masked - source_hat)**2)*depth_weight_masked)

        # penalize depths not in [0,1]
        loss_hinge1 = loss_hinge2 = 0.0
        if (source_hat<=0.0).any():
            loss_hinge1 = 2.0*((source_hat[source_hat<=0.0])**2).mean()
        # if (source_hat>=1.0).any():
        #     loss_hinge2 = 0.3*((source_hat[source_hat>=1.0])**2).mean() 
        
        loss = loss + loss_hinge1 + loss_hinge2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        iteration+=1
        if iteration % 1000 == 0:
            # print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
            loss_prev = loss.item()
        loss_ema = loss.item() * 0.2 + loss_ema * 0.8

    loss = loss.item()
    # print(f"loss ={loss:10.5f}")

    with torch.no_grad():
        refined_source = (scale*source + shift) 
    torch.cuda.empty_cache()
    return refined_source.cpu().numpy(), loss