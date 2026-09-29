# Trained models

The models behind Table 1 are 3D Gaussian Splatting outputs (`point_cloud/iteration_30000/point_cloud.ply`
plus `cfg_args` and `cameras.json`).

The Tanks and Temples models (Train, Horse; 3DGS and ours; three SfM pipelines each, 12 in total, 2.9 GB) are on
Hugging Face Hub at **[gandan-lee/pdigs](https://huggingface.co/gandan-lee/pdigs)**:

```bash
pip install huggingface_hub
hf download gandan-lee/pdigs --local-dir models      # models/<method>/<sfm>/<scene>/point_cloud.ply
```

The Parking lots and Street-view models are not released, because they would expose the unreleased
in-house imagery.

Sizes of all Table 1 model folders on the training server:

| Scene | SfM | 3DGS | Ours |
|:--|:--|:-:|:-:|
| Train | COLMAP | 1.5 GB | 1.5 GB |
| Train | SP-SG | 1.3 GB | 1.4 GB |
| Train | LoFTR | 1.3 GB | 1.8 GB |
| Horse | COLMAP | 0.9 GB | 1.0 GB |
| Horse | SP-SG | 0.8 GB | 0.9 GB |
| Horse | LoFTR | 0.9 GB | 1.2 GB |
| Parking lots | SP-SG | 1.3 GB | 1.2 GB |
| Parking lots | LoFTR | 1.2 GB | 1.3 GB |
| Street-view | SP-SG | 1.0 GB | 0.4 GB |
| Street-view | LoFTR | 0.9 GB | 1.0 GB |

Sizes include the test/train renderings stored next to each model.
