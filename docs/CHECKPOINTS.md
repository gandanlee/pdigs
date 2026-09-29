# Trained models

The models behind Table 1 are 3D Gaussian Splatting outputs (`point_cloud/iteration_30000/point_cloud.ply`
plus `cfg_args` and `cameras.json`), about 0.4–2 GB each and 25 GB in total. They are not released.

Models of the Parking lots and Street-view scenes would also expose the unreleased in-house imagery,
so only the Tanks and Temples models (Train, Horse) are candidates for release.

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
