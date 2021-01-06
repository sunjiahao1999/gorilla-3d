# Copyright (c) Gorilla-Lab. All rights reserved.
import math
from typing import Optional, Union
from numpy.lib.arraysetops import isin

import torch
import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import transforms3d.euler as euler
from plyfile import PlyData
from plyfile import PlyElement

def elastic(xyz, gran, mag):
    """Elastic distortion (from point group)

    Args:
        xyz (np.ndarray): input point cloud
        gran (float): distortion param
        mag (float): distortion scalar

    Returns:
        xyz: point cloud with elastic distortion
    """
    blur0 = np.ones((3, 1, 1)).astype("float32") / 3
    blur1 = np.ones((1, 3, 1)).astype("float32") / 3
    blur2 = np.ones((1, 1, 3)).astype("float32") / 3

    bb = np.abs(xyz).max(0).astype(np.int32)//gran + 3
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype("float32") for _ in range(3)]
    noise = [ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
    noise = [ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
    noise = [ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
    noise = [ndimage.filters.convolve(n, blur0, mode="constant", cval=0) for n in noise]
    noise = [ndimage.filters.convolve(n, blur1, mode="constant", cval=0) for n in noise]
    noise = [ndimage.filters.convolve(n, blur2, mode="constant", cval=0) for n in noise]
    ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
    interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
    def g(xyz_):
        return np.hstack([i(xyz_)[:,None] for i in interp])
    return xyz + g(xyz) * mag


def pc_jitter(xyz, std=0.1):
    jitter_mat = np.eye(3)
    jitter_mat += np.random.randn(3, 3) * std
    xyz = xyz @ jitter_mat
    return xyz

def pc_flipper(xyz, dim="x"):
    dims = ["x", "y", "z"]
    assert dim in dims
    flip_dim = dims.index(dim)
    xyz[:, flip_dim] = -xyz[:, flip_dim]
    return xyz

def pc_rotator(xyz):
    theta = np.random.rand() * 2 * math.pi
    rot_mat = euler.euler2mat(0, 0, theta, "syxz")
    xyz = xyz @ rot_mat.T
    return xyz


def pc_aug(xyz, jitter=False, flip=False, rot=False):
    """point cloud augmentation(from point group)

    Args:
        x (np.ndarray): input point cloud
        jitter (bool, optional): [description]. Defaults to False.
        flip (bool, optional): [description]. Defaults to False.
        rot (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    if jitter:
        xyz = pc_jitter(xyz)
    if flip:
        flag = np.random.randint(0, 2)
        if flag:
            xyz = pc_flipper(xyz)
    if rot:
        xyz = pc_rotator(xyz)

    return xyz


def square_distance(src, dst=None):
    r"""Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [N, C]
            dst: target points, [M, C]
        Output:
            dist: per-point square distance, [N, M]
    """
    if dst is None:
        dst = src
    N = src.shape[0]
    M = dst.shape[0]
    dst_t = dst.T # [C, M]
    dist  = -2 * (src @ dst_t) # [N, M]
    dist += (src ** 2).sum(-1).reshape(N, 1)
    dist += (dst ** 2).sum(-1).reshape(1, M)
    return dist


def save_pc(points: Union[np.ndarray, torch.Tensor],
            colors: Optional[Union[np.ndarray, torch.Tensor]]=None,
            filename: str="./temp.ply") -> None:
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if colors is not None:
        if isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()
    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        if colors is not None and colors.max() > 1:
            colors = colors / 255
            pc.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filename, pc)
    except:
        if colors is not None and colors.max() < 1:
            colors = (colors * 255).astype(np.int32)
            vc = np.concatenate([points, colors], axis=1)
            vc = [tuple(vc[i]) for i in range(vc.shape[0])]
            vc = np.array(vc, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                    ("r", "i4"), ("g", "i4"), ("b", "i4")])
        else:
            vc = points
            vc = [tuple(vc[i]) for i in range(vc.shape[0])]
            vc = np.array(vc, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        el = PlyElement.describe(vc, "vertex")
        PlyData([el]).write(filename)

