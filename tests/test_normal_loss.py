"""CPU checks of the normal-consistency loss (no CUDA rasterizer needed).

    python -m pytest tests
"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss.normal_guidance import loss_normal_guidance  # noqa: E402


def _cam(normal, R=np.eye(3)):
    return SimpleNamespace(original_normal=normal, R=R)


def _identity_quat(H, W):
    q = torch.zeros(4, H, W)
    q[0] = 1.0  # (w, x, y, z)
    return q


def test_axis_aligned_normal():
    # Identity rotation: the covariance axes are x, y, z. A +z normal is orthogonal to x and y and
    # parallel to z; both terms average |axis . n| over the three axes, so L_axis = 1/3 and
    # L_scale = s_z / 3.
    H, W = 4, 5
    normal = torch.zeros(3, H, W)
    normal[2] = 1.0
    scale = torch.tensor([0.5, 0.3, 0.2])[:, None, None].repeat(1, H, W)
    loss = loss_normal_guidance(_cam(normal), _identity_quat(H, W), scale, axis_weight=0.8)
    assert math.isclose(loss.item(), 0.8 / 3 + 0.2 * 0.2 / 3, rel_tol=1e-5)


def test_axis_weight_endpoints():
    torch.manual_seed(0)
    H, W = 6, 7
    normal = torch.nn.functional.normalize(torch.randn(3, H, W), dim=0)
    quat, scale = torch.randn(4, H, W), torch.rand(3, H, W)
    l_axis = loss_normal_guidance(_cam(normal), quat, scale, axis_weight=1.0)
    l_scale = loss_normal_guidance(_cam(normal), quat, scale, axis_weight=0.0)
    l_mix = loss_normal_guidance(_cam(normal), quat, scale, axis_weight=0.8)
    assert torch.allclose(l_mix, 0.8 * l_axis + 0.2 * l_scale, atol=1e-6)


def test_gradient_flows_to_rotation_and_scale():
    torch.manual_seed(0)
    H, W = 3, 3
    normal = torch.nn.functional.normalize(torch.randn(3, H, W), dim=0)
    quat = torch.randn(4, H, W, requires_grad=True)
    scale = torch.rand(3, H, W, requires_grad=True)
    loss_normal_guidance(_cam(normal), quat, scale).backward()
    assert quat.grad is not None and quat.grad.abs().sum() > 0
    assert scale.grad is not None and scale.grad.abs().sum() > 0
