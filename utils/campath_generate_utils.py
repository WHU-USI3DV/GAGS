import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

# from utils.stepfun import sample_np, sample
from scipy.interpolate import splprep, splev
import scipy


def integrate_weights(w):
    """Compute the cumulative sum of w, assuming all weight vectors sum to 1.

    The output's size on the last dimension is one greater than that of the input,
    because we're computing the integral corresponding to the endpoints of a step
    function, not the integral of the interior/bin values.

    Args:
      w: Tensor, which will be integrated along the last axis. This is assumed to
        sum to 1 along the last axis, and this function will (silently) break if
        that is not the case.

    Returns:
      cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
    """
    cw = torch.cumsum(w[..., :-1], dim=-1).clamp_max(1)
    shape = cw.shape[:-1] + (1,)
    # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
    cw0 = torch.cat(
        [torch.zeros(shape, device=cw.device), cw, torch.ones(shape, device=cw.device)],
        dim=-1,
    )
    return cw0


def invert_cdf(u, t, w_logits):
    """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
    # Compute the PDF and CDF for each weight vector.
    w = torch.softmax(w_logits, dim=-1)
    cw = integrate_weights(w)
    # Interpolate into the inverse CDF.
    t_new = math.sorted_interp(u, cw, t)
    return t_new


def sample(
    rand, t, w_logits, num_samples, single_jitter=False, deterministic_center=False
):
    """Piecewise-Constant PDF sampling from a step function.

    Args:
      rand: random number generator (or None for `linspace` sampling).
      t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
      w_logits: [..., num_bins], logits corresponding to bin weights
      num_samples: int, the number of samples.
      single_jitter: bool, if True, jitter every sample along each ray by the same
        amount in the inverse CDF. Otherwise, jitter each sample independently.
      deterministic_center: bool, if False, when `rand` is None return samples that
        linspace the entire PDF. If True, skip the front and back of the linspace
        so that the centers of each PDF interval are returned.

    Returns:
      t_samples: [batch_size, num_samples].
    """
    eps = torch.finfo(t.dtype).eps
    # eps = 1e-3

    device = t.device

    # Draw uniform samples.
    if not rand:
        if deterministic_center:
            pad = 1 / (2 * num_samples)
            u = torch.linspace(pad, 1.0 - pad - eps, num_samples, device=device)
        else:
            u = torch.linspace(0, 1.0 - eps, num_samples, device=device)
        u = torch.broadcast_to(u, t.shape[:-1] + (num_samples,))
    else:
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u_max = eps + (1 - eps) / num_samples
        max_jitter = (1 - u_max) / (num_samples - 1) - eps
        d = 1 if single_jitter else num_samples
        u = (
            torch.linspace(0, 1 - u_max, num_samples, device=device)
            + torch.rand(t.shape[:-1] + (d,), device=device) * max_jitter
        )

    return invert_cdf(u, t, w_logits)


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(lookdir, up, position, subtract_position=False):
    """Construct lookat view matrix."""
    vec2 = normalize((lookdir - position) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def generate_interpolated_path(
    views,
    n_interp,
    spline_degree=5,
    smoothness=0.03,
    rot_weight=0.1,
    lock_up=False,
    fixed_up_vector=None,
    lookahead_i=None,
    frames_per_colmap=None,
    const_speed=False,
    n_buffer=None,
    periodic=False,
    n_interp_as_total=False,
):
    """Creates a smooth spline path between input keyframe camera poses.

    Spline is calculated with poses in format (position, lookat-point, up-point).
    Args:
      poses: (n, 3, 4) array of input pose keyframes.
      n_interp: returned path will have n_interp * (n - 1) total poses.
      spline_degree: polynomial degree of B-spline.
      smoothness: parameter for spline smoothing, 0 forces exact interpolation.
      rot_weight: relative weighting of rotation/translation in spline solve.
      lock_up: if True, forced to use given Up and allow Lookat to vary.
      fixed_up_vector: replace the interpolated `up` with a fixed vector.
      lookahead_i: force the look direction to look at the pose `i` frames ahead.
      frames_per_colmap: conversion factor for the desired average velocity.
      const_speed: renormalize spline to have constant delta between each pose.
      n_buffer: Number of buffer frames to insert at the start and end of the
        path. Helps keep the ends of a spline path straight.
      periodic: make the spline path periodic (perfect loop).
      n_interp_as_total: use n_interp as total number of poses in path rather than
        the number of poses to interpolate between each input.

    Returns:
      Array of new camera poses with shape (n_interp * (n - 1), 3, 4), or
      (n_interp, 3, 4) if n_interp_as_total is set.
    """
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        poses = []
        for i in range(len(points)):
            pos, lookat_point, up_point = points[i]
            if lookahead_i is not None:
                if i + lookahead_i < len(points):
                    lookat = pos - points[i + lookahead_i][0]
            else:
                lookat = pos - lookat_point
            up = (up_point - pos) if fixed_up_vector is None else fixed_up_vector
            poses.append(viewmatrix(lookat, up, pos))
        return np.array(poses)

    def insert_buffer_poses(poses, n_buffer):
        """Insert extra poses at the start and end of the path."""

        def average_distance(points):
            distances = np.linalg.norm(points[1:] - points[0:-1], axis=-1)
            return np.mean(distances)

        def shift(pose, dz):
            result = np.copy(pose)
            z = result[:3, 2]
            z /= np.linalg.norm(z)
            # Move along forward-backward axis. -z is forward.
            result[:3, 3] += z * dz
            return result

        dz = average_distance(poses[:, :3, 3])
        prefix = np.stack([shift(poses[0], (i + 1) * dz) for i in range(n_buffer)])
        prefix = prefix[::-1]  # reverse order
        suffix = np.stack([shift(poses[-1], -(i + 1) * dz) for i in range(n_buffer)])
        result = np.concatenate([prefix, poses, suffix])
        return result

    def remove_buffer_poses(poses, u, n_frames, u_keyframes, n_buffer):
        u_keyframes = u_keyframes[n_buffer:-n_buffer]
        mask = (u >= u_keyframes[0]) & (u <= u_keyframes[-1])
        poses = poses[mask]
        u = u[mask]
        n_frames = len(poses)
        return poses, u, n_frames, u_keyframes

    def interp(points, u, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1)) # 对每个view的(position, lookat, up)同时进行插值
        k = min(k, sh[0] - 1)
        tck, u_keyframes = scipy.interpolate.splprep(pts.T, k=k, s=s, per=periodic)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (len(u), sh[1], sh[2]))
        return new_points, u_keyframes

    if n_buffer is not None:
        poses = insert_buffer_poses(poses, n_buffer)
    points = poses_to_points(poses, dist=rot_weight)
    if n_interp_as_total:
        n_frames = n_interp + 1  # Add extra since final pose is discarded.
    else:
        n_frames = n_interp * (points.shape[0] - 1)
    u = np.linspace(0, 1, n_frames, endpoint=True)
    new_points, u_keyframes = interp(points, u=u, k=spline_degree, s=smoothness)
    poses = points_to_poses(new_points)
    if n_buffer is not None:
        poses, u, n_frames, u_keyframes = remove_buffer_poses(
            poses, u, n_frames, u_keyframes, n_buffer
        )
        # poses, transform = transform_poses_pca(poses)
    if frames_per_colmap is not None:
        # Recalculate the number of frames to achieve desired average velocity.
        positions = poses[:, :3, -1]
        lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        total_length_colmap = lengths.sum()
        print("old n_frames:", n_frames)
        print("total_length_colmap:", total_length_colmap)
        n_frames = int(total_length_colmap * frames_per_colmap)
        print("new n_frames:", n_frames)
        u = np.linspace(
            np.min(u_keyframes), np.max(u_keyframes), n_frames, endpoint=True
        )
        new_points, _ = interp(points, u=u, k=spline_degree, s=smoothness)
        poses = points_to_poses(new_points)

    if const_speed:
        # Resample timesteps so that the velocity is nearly constant.
        positions = poses[:, :3, -1]
        lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        u = sample(None, u, np.log(lengths), n_frames + 1) # 根据log(d(i,i+1))重新采样时间步长
        new_points, _ = interp(points, u=u, k=spline_degree, s=smoothness)
        poses = points_to_poses(new_points)

    #   return poses[:-1], u[:-1], u_keyframes
    return poses[:-1]


def simple_interpolation(views, num_points, spline_degree=3):
    """Perform linear interpolation between given views."""
    poses = []
    for view in views:
        tmp_pos = view.T # 3
        poses.append(tmp_pos)
    poses = np.stack(poses, 0) # (n, 3)
    
    x, y, z = poses[:, 0], poses[:, 1], poses[:, 2]

    # 生成样条曲线
    tck, u = splprep([x, y, z], s=0, k=spline_degree)

    # 生成参数 u_new，从0到1，表示在样条曲线上均匀取样
    u_new = np.linspace(0, 1, num_points)

    # 使用splev计算样条曲线上的num_points个坐标
    new_points = splev(u_new, tck)

    # 将生成的点重新组合为三维坐标
    interpolated_coords = np.vstack(new_points).T

    return interpolated_coords
