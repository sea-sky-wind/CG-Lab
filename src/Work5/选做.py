import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import pytorch3d

from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    BlendParams,
    SoftSilhouetteShader,
    SoftPhongShader,
    PointLights,
    TexturesVertex
)

from pytorch3d.loss import (
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print(f"当前设备: {device}")
print(f"PyTorch3D 版本: {pytorch3d.__version__}")
print("=" * 60)

output_dir = "texture_results"

mesh_dir = os.path.join(output_dir, "meshes")
image_dir = os.path.join(output_dir, "images")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(mesh_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

obj_path = "cow.obj"

if not os.path.exists(obj_path):
    raise FileNotFoundError(
        "未找到 cow.obj，请确保其与 py 文件在同一目录！"
    )

print("[INFO] 正在加载目标模型...")

verts, faces, _ = load_obj(obj_path)

verts = verts.to(device)
faces_idx = faces.verts_idx.to(device)


verts = verts - verts.mean(0)
verts = verts / verts.abs().max()


verts_rgb = (verts - verts.min()) / (
    verts.max() - verts.min()
)

target_textures = TexturesVertex(
    verts_features=verts_rgb.unsqueeze(0)
)

target_mesh = Meshes(
    verts=[verts],
    faces=[faces_idx],
    textures=target_textures
)

print("[INFO] 目标 Mesh 加载完成")

num_views = 10

elev = torch.linspace(-20, 20, num_views)
azim = torch.linspace(-180, 180, num_views)

R, T = look_at_view_transform(
    dist=2.7,
    elev=elev,
    azim=azim
)

cameras = FoVPerspectiveCameras(
    device=device,
    R=R,
    T=T
)

lights = PointLights(
    device=device,
    location=[[0.0, 0.0, -3.0]]
)

sigma = 1e-4

raster_settings = RasterizationSettings(
    image_size=128,
    blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
    faces_per_pixel=50
)

blend_params = BlendParams(
    sigma=sigma,
    gamma=1e-4
)

silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(
        blend_params=blend_params
    )
)

phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights
    )
)

print("[INFO] 正在生成 Ground Truth...")

with torch.no_grad():

    target_images = phong_renderer(
        target_mesh.extend(num_views)
    )

    target_silhouette = silhouette_renderer(
        target_mesh.extend(num_views)
    )[..., 3]

print("[INFO] Ground Truth 渲染完成")

print("[INFO] 初始化源球体...")

src_mesh = ico_sphere(3, device)

deform_verts = torch.zeros_like(
    src_mesh.verts_packed(),
    requires_grad=True
)


vertex_colors = torch.full_like(
    src_mesh.verts_packed(),
    0.5,
    requires_grad=True
)

optimizer = torch.optim.Adam(
    [deform_verts, vertex_colors],
    lr=0.01
)

w_rgb = 1.0
w_silhouette = 1.0
w_laplacian = 0.5
w_edge = 0.2
w_normal = 0.01

epochs = 300

loss_history = []

print("[INFO] 开始联合优化...")

for epoch in tqdm(range(epochs)):

    optimizer.zero_grad()

    new_mesh = src_mesh.offset_verts(
        deform_verts
    )

    textures = TexturesVertex(
        verts_features=torch.sigmoid(
            vertex_colors
        ).unsqueeze(0)
    )

    new_mesh.textures = textures

    pred_images = phong_renderer(
        new_mesh.extend(num_views)
    )

    pred_silhouette = silhouette_renderer(
        new_mesh.extend(num_views)
    )[..., 3]

    loss_rgb = (
        (
            pred_images[..., :3]
            - target_images[..., :3]
        ) ** 2
    ).mean()

    loss_silhouette = (
        (
            pred_silhouette
            - target_silhouette
        ) ** 2
    ).mean()


    loss_laplacian = mesh_laplacian_smoothing(
        new_mesh
    )

    loss_edge = mesh_edge_loss(
        new_mesh
    )

    loss_normal = mesh_normal_consistency(
        new_mesh
    )

    loss = (
        w_rgb * loss_rgb
        + w_silhouette * loss_silhouette
        + w_laplacian * loss_laplacian
        + w_edge * loss_edge
        + w_normal * loss_normal
    )

    loss.backward()

    optimizer.step()

    loss_history.append(loss.item())


    if epoch % 20 == 0 or epoch == epochs - 1:

        print("\n" + "=" * 60)
        print(f"Epoch {epoch}/{epochs}")
        print(f"Total Loss      : {loss.item():.6f}")
        print(f"RGB Loss        : {loss_rgb.item():.6f}")
        print(f"Silhouette Loss : {loss_silhouette.item():.6f}")
        print("=" * 60)

        current_verts = new_mesh.verts_list()[0]
        current_faces = new_mesh.faces_list()[0]

        mesh_path = os.path.join(
            mesh_dir,
            f"mesh_epoch_{epoch:03d}.obj"
        )

        save_obj(
            mesh_path,
            current_verts,
            current_faces
        )

        print(f"[SAVE] Mesh 已保存: {mesh_path}")

        fig, ax = plt.subplots(
            2,
            2,
            figsize=(10, 10)
        )

        # Target RGB
        ax[0, 0].imshow(
            target_images[0, ..., :3]
            .cpu()
            .numpy()
        )

        ax[0, 0].set_title(
            "Target RGB"
        )

        ax[0, 0].axis("off")

        # Pred RGB
        ax[0, 1].imshow(
            pred_images[0, ..., :3]
            .detach()
            .cpu()
            .numpy()
        )

        ax[0, 1].set_title(
            f"Pred RGB (Epoch {epoch})"
        )

        ax[0, 1].axis("off")

        # Target Silhouette
        ax[1, 0].imshow(
            target_silhouette[0]
            .cpu()
            .numpy(),
            cmap="gray"
        )

        ax[1, 0].set_title(
            "Target Silhouette"
        )

        ax[1, 0].axis("off")

        # Pred Silhouette
        ax[1, 1].imshow(
            pred_silhouette[0]
            .detach()
            .cpu()
            .numpy(),
            cmap="gray"
        )

        ax[1, 1].set_title(
            "Pred Silhouette"
        )

        ax[1, 1].axis("off")

        plt.tight_layout()

        image_path = os.path.join(
            image_dir,
            f"epoch_{epoch:03d}.png"
        )

        plt.savefig(image_path)

        plt.close()

        print(f"[SAVE] 图像已保存: {image_path}")

print("\n[INFO] 正在保存最终模型...")

final_mesh = src_mesh.offset_verts(
    deform_verts
)

final_mesh.textures = TexturesVertex(
    verts_features=torch.sigmoid(
        vertex_colors
    ).unsqueeze(0)
)

final_verts = final_mesh.verts_list()[0]
final_faces = final_mesh.faces_list()[0]

final_mesh_path = os.path.join(
    output_dir,
    "final_textured_cow.obj"
)

save_obj(
    final_mesh_path,
    final_verts,
    final_faces
)

print(f"[SAVE] 最终模型已保存: {final_mesh_path}")

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Texture Optimization Loss"
)

loss_curve_path = os.path.join(
    output_dir,
    "loss_curve.png"
)

plt.savefig(loss_curve_path)

plt.close()

print(f"[SAVE] Loss 曲线已保存: {loss_curve_path}")

print("\n")
print("=" * 60)
print("联合 Shape + Texture Optimization 完成！")
print("=" * 60)