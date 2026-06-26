"""
LBS 姿态动画（选做）
独立脚本，不影响 run_lbs_lab.py 的必做内容。

功能：
  - 固定 shape 参数
  - 让左肘关节从 0 逐渐旋转到目标角度
  - 生成逐帧 PNG + 合成 GIF
  - 用颜色标出 lbs_weights 中该关节的影响区域，直观看到权重随骨骼运动被平滑带动

运行示例：
  D:\python3.13.12\python.exe run_lbs_animation.py \
      --model-dir ./models \
      --out-dir ./outputs/animation \
      --joint-id 18 \
      --frames 24 \
      --max-angle 2.0

参数说明：
  --joint-id   要旋转的关节编号（默认 18 = 左肘）
  --frames     总帧数（默认 24）
  --max-angle  最大旋转角度，单位弧度（默认 2.0，约 115 度）
  --fps        GIF 帧率（默认 12）
"""

import os
import sys
import types
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import smplx
from smplx.lbs import (
    blend_shapes,
    vertices2joints,
    batch_rodrigues,
    batch_rigid_transform,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# chumpy 兼容 shim（与 run_lbs_lab.py 相同）
# ─────────────────────────────────────────────
class _ChumpyArrayShim:
    def __setstate__(self, state):
        self.__dict__.update(state)

    def _array(self):
        if hasattr(self, "r"):
            return self.r
        if hasattr(self, "x"):
            return self.x
        raise AttributeError("Cannot recover array data from chumpy pickle object")

    def __array__(self, dtype=None):
        return np.asarray(self._array(), dtype=dtype)

    @property
    def shape(self):
        return np.asarray(self).shape

    def __len__(self):
        return len(np.asarray(self))

    def __getitem__(self, item):
        return np.asarray(self)[item]


def install_chumpy_pickle_shim():
    if "chumpy.ch" in sys.modules:
        return
    chumpy_module = types.ModuleType("chumpy")
    chumpy_ch_module = types.ModuleType("chumpy.ch")
    _ChumpyArrayShim.__name__ = "Ch"
    _ChumpyArrayShim.__qualname__ = "Ch"
    _ChumpyArrayShim.__module__ = "chumpy.ch"
    chumpy_ch_module.Ch = _ChumpyArrayShim
    chumpy_module.ch = chumpy_ch_module
    sys.modules["chumpy"] = chumpy_module
    sys.modules["chumpy.ch"] = chumpy_ch_module


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def set_axes_equal(ax, vertices: np.ndarray):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = 0.5 * np.max(maxs - mins + 1e-8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def smpl_to_plot_coords(points: np.ndarray):
    """SMPL 坐标系 → 绘图坐标系（交换 Y/Z 使人物直立）"""
    return points[:, [0, 2, 1]]


def shade_face_colors(vertices: np.ndarray, faces: np.ndarray, face_colors: np.ndarray):
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0]
    )
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals /= norm_len
    light_dir = np.array([-0.25, -0.55, 0.80], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    intensity = 0.35 + 0.65 * np.clip(normals @ light_dir, 0.0, 1.0)
    shaded = face_colors.copy()
    shaded[:, :3] *= intensity[:, None]
    return shaded


def get_face_colors_from_vertex_scalar(vertex_scalar, faces, cmap_name="viridis"):
    scalar = vertex_scalar.astype(np.float64)
    scalar = (scalar - scalar.min()) / (scalar.max() - scalar.min() + 1e-8)
    face_scalar = scalar[faces].mean(axis=1)
    cmap = plt.get_cmap(cmap_name)
    return cmap(face_scalar)


def prepare_posedirs(posedirs: torch.Tensor, expected_pose_dim: int):
    if posedirs.dim() != 2:
        posedirs = posedirs.reshape(posedirs.shape[0], -1)
    if posedirs.shape[0] == expected_pose_dim:
        return posedirs
    if posedirs.shape[1] == expected_pose_dim:
        return posedirs.T
    raise RuntimeError(
        f"posedirs 形状与 pose_feature 不匹配: {tuple(posedirs.shape)}, "
        f"expected_pose_dim={expected_pose_dim}"
    )


# ─────────────────────────────────────────────
# 核心：手写 LBS 前向
# ─────────────────────────────────────────────
def manual_lbs(model, betas, global_orient, body_pose):
    device = betas.device
    dtype = betas.dtype

    v_template = model.v_template
    if v_template.dim() == 2:
        v_template = v_template.unsqueeze(0)

    shapedirs = model.shapedirs[:, :, :betas.shape[1]]
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    J = vertices2joints(model.J_regressor, v_shaped)

    full_pose = torch.cat([global_orient, body_pose], dim=1)
    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)

    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view(1, -1)

    posedirs = prepare_posedirs(model.posedirs, expected_pose_dim=pose_feature.shape[1])
    pose_offsets = torch.matmul(pose_feature, posedirs).view(1, -1, 3)
    v_posed = v_shaped + pose_offsets

    J_transformed, A = batch_rigid_transform(rot_mats, J, model.parents, dtype=dtype)

    num_joints = J.shape[1]
    W = model.lbs_weights.unsqueeze(0).expand(1, -1, -1)
    T = torch.matmul(W, A.view(1, num_joints, 16)).view(1, -1, 4, 4)

    homogen_coord = torch.ones((1, v_posed.shape[1], 1), dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


# ─────────────────────────────────────────────
# 绘制单帧
# ─────────────────────────────────────────────
def draw_frame(
    ax,
    verts_np: np.ndarray,
    faces: np.ndarray,
    joints_np: np.ndarray,
    weight_scalar: np.ndarray,
    title: str,
):
    pv = smpl_to_plot_coords(verts_np)
    pj = smpl_to_plot_coords(joints_np)

    face_colors = get_face_colors_from_vertex_scalar(weight_scalar, faces, cmap_name="plasma")
    face_colors = shade_face_colors(pv, faces, face_colors)

    mesh = Poly3DCollection(
        pv[faces],
        facecolors=face_colors,
        linewidths=0.02,
        edgecolors=(0, 0, 0, 0.04),
    )
    ax.add_collection3d(mesh)

    ax.scatter(
        pj[:, 0], pj[:, 1], pj[:, 2],
        c="white", s=10, depthshade=False,
        edgecolors="black", linewidths=0.3,
    )

    set_axes_equal(ax, pv)
    ax.set_proj_type("persp", focal_length=0.85)
    ax.view_init(elev=12, azim=108)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
def main(args):
    device = torch.device("cpu")
    dtype = torch.float32

    model_dir = args.model_dir if os.path.isabs(args.model_dir) \
        else os.path.join(SCRIPT_DIR, args.model_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) \
        else os.path.join(SCRIPT_DIR, args.out_dir)

    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # 加载模型
    install_chumpy_pickle_shim()
    model = smplx.create(
        model_path=model_dir,
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        num_betas=args.num_betas,
    ).to(device)

    faces = np.asarray(model.faces, dtype=np.int32)

    # 固定 shape 参数
    betas = torch.zeros((1, args.num_betas), dtype=dtype, device=device)
    betas[0, 0] = 2.0
    betas[0, 1] = -1.2

    # 基础姿态（双臂略微外展，与必做保持一致）
    global_orient = torch.zeros((1, 3), dtype=dtype, device=device)
    base_body_pose = torch.zeros((1, 23 * 3), dtype=dtype, device=device)

    # 要旋转的关节（默认 18 = 左肘，对应 body_pose 中第 17 个关节，索引从 1 开始）
    # body_pose 的关节编号比 lbs_weights 关节编号少 1（因为 global_orient 占了 0）
    joint_id = int(args.joint_id)          # lbs_weights / SMPL 关节编号
    pose_idx = joint_id - 1               # body_pose 数组中的偏移
    axis = np.array([0.0, -1.0, 0.0])     # 绕 Y 轴旋转（弯肘方向）

    # lbs_weights 中该关节对每个顶点的影响权重（用于染色）
    weight_scalar = to_numpy(model.lbs_weights[:, joint_id])

    num_frames = int(args.frames)
    max_angle = float(args.max_angle)

    frame_paths = []

    print(f"生成 {num_frames} 帧动画，关节 {joint_id}，最大旋转角度 {max_angle:.2f} rad ...")

    for i in range(num_frames):
        # 0 → max_angle → 0  平滑来回（余弦插值）
        t = i / max(num_frames - 1, 1)
        angle = max_angle * 0.5 * (1.0 - np.cos(2.0 * np.pi * t))

        body_pose = base_body_pose.clone()
        aa = axis * angle                              # 轴角向量
        start = pose_idx * 3
        body_pose[0, start:start + 3] = torch.tensor(aa, dtype=dtype, device=device)

        with torch.no_grad():
            verts, J_transformed = manual_lbs(model, betas, global_orient, body_pose)

        verts_np = to_numpy(verts[0])
        joints_np = to_numpy(J_transformed[0])

        fig = plt.figure(figsize=(5, 6))
        ax = fig.add_subplot(111, projection="3d")
        draw_frame(
            ax, verts_np, faces, joints_np, weight_scalar,
            title=f"Joint {joint_id}  angle={angle:.2f} rad  (frame {i+1}/{num_frames})",
        )
        fig.tight_layout()

        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        fig.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

        print(f"  帧 {i+1:3d}/{num_frames}  angle={angle:.3f}")

    # 合成 GIF
    gif_path = os.path.join(out_dir, "lbs_animation.gif")
    try:
        from PIL import Image
        images = [Image.open(p) for p in frame_paths]
        duration_ms = int(1000 / args.fps)
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            loop=0,
            duration=duration_ms,
        )
        print(f"\nGIF 已保存：{gif_path}")
    except ImportError:
        print("\n[提示] 未找到 Pillow，跳过 GIF 合成。")
        print("       安装方法：D:\\python3.13.12\\python.exe -m pip install Pillow")
        print(f"       逐帧 PNG 已保存在：{frames_dir}")

    print(f"\n完成！输出目录：{out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL LBS 姿态动画（选做）")
    parser.add_argument("--model-dir", type=str, default="./models")
    parser.add_argument("--out-dir",   type=str, default="./outputs/animation")
    parser.add_argument("--joint-id",  type=int, default=18,  help="要旋转的关节编号（默认 18=左肘）")
    parser.add_argument("--frames",    type=int, default=24,  help="总帧数")
    parser.add_argument("--max-angle", type=float, default=2.0, help="最大旋转角度（弧度）")
    parser.add_argument("--fps",       type=int, default=12,  help="GIF 帧率")
    parser.add_argument("--num-betas", type=int, default=10)
    args = parser.parse_args()
    main(args)
