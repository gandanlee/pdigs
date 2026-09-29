# Normal priors with Metric3D

The normal priors (`<scene>/normals/<image_name>.npy`) were produced with
[Metric3D](https://github.com/YvanYin/Metric3D) v2, ViT-Large (`metric_depth_vit_large_800k.pth`),
through its test script:

```bash
# inside a Metric3D checkout, with the weight in ./weight/
python mono/tools/test_scale_cano.py mono/configs/HourglassDecoder/vit.raft5.large.py \
    --load-from ./weight/metric_depth_vit_large_800k.pth \
    --test_data_path $DATA/<sfm>/<scene>/images --launcher None
```

Stock Metric3D only saves visualizations. We added one `np.save` to
`save_normal_val_imgs` in `mono/utils/visualization.py`, so that the predicted normal of every image is
also written as a `(3, H, W)` float array in a `normals/` folder next to the outputs:

```python
# mono/utils/visualization.py, inside save_normal_val_imgs(...), after pred is permuted to (H, W, 3)
normal_dir = os.path.join(save_dir, filename.split('/')[0], 'normals')
os.makedirs(normal_dir, exist_ok=True)
np.save(os.path.join(normal_dir, filename.split('/')[-1].replace('normal_', '')[:-4] + '.npy'),
        pred.cpu().numpy().transpose(2, 0, 1))
```

Copy that `normals/` folder into the scene directory. Every image in `images/` needs a
`normals/<image_name>.npy`, and the normals are in the camera frame.
`scene/cameras.py` resizes them to the training resolution when it loads them.

The three SfM variants of a scene (`colmap/`, `sp-sg/`, `defree_sfm/`) share the same images, so they
can share one set of normals.

Metric3D is released under CC0 1.0.
