"""Refactorizado para uso en Colab y por lotes.

Define `InstantMeshProcessor` que carga modelos una sola vez y expone
`process_folder` / `process_images` para procesar listas de rutas.
Mantiene compatibilidad CLI al ejecutar como script.
"""

from __future__ import annotations

import os
import argparse
import time
import gc
from typing import List, Optional

import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl, save_glb
from src.utils.infer_util import remove_background, resize_foreground, save_video


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    frames = []
    for i in range(0, render_cameras.shape[1], chunk_size):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            # Some models expose different synthesizer APIs; this keeps previous behavior
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)

    frames = torch.cat(frames, dim=1)[0]
    return frames


class InstantMeshProcessor:
    def __init__(self, config_path: str, device: Optional[torch.device] = None, cache_dir: str = './ckpts',
                 dtype=torch.float16, enable_xformers: bool = True):
        self.config_path = os.path.abspath(config_path)
        self.config = OmegaConf.load(self.config_path)
        self.config_name = os.path.basename(self.config_path).replace('.yaml', '')
        self.model_config = self.config.model_config
        self.infer_config = self.config.infer_config

        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dtype = dtype
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.pipeline = None
        self.model = None
        self.rembg_session = None
        self.enable_xformers = enable_xformers

        seed_everything(42)

    def load_models(self):
        # Diffusion pipeline (Zero123)
        print('Loading diffusion pipeline...')
        self.pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline="zero123plus",
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing='trailing'
        )

        # try to enable xformers if present
        # try to enable xformers if present; do not attempt to install/build here
        xformers_available = False
        if self.enable_xformers:
            try:
                import xformers  # noqa: F401
                xformers_available = True
            except Exception:
                xformers_available = False

        if xformers_available:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print('xformers enabled for memory-efficient attention')
            except Exception as e:
                print('xformers import present but enable failed:', e)
                # fallback: set sdpa hint if possible
                try:
                    if hasattr(self.pipeline, 'unet') and hasattr(self.pipeline.unet.config, 'attention_impl'):
                        self.pipeline.unet.config.attention_impl = 'sdpa'
                except Exception:
                    pass
        else:
            print('xformers not available; continuing without it')

        # load custom UNet weights if provided in config
        unet_path = getattr(self.infer_config, 'unet_path', None)
        if unet_path and os.path.exists(unet_path):
            print('Loading custom UNet weights...')
            state_dict = torch.load(unet_path, map_location='cpu')
            self.pipeline.unet.load_state_dict(state_dict, strict=True)

        self.pipeline = self.pipeline.to(self.device)

        # Reconstruction model
        print('Loading reconstruction model...')
        self.model = instantiate_from_config(self.model_config)
        model_path = getattr(self.infer_config, 'model_path', None)
        if model_path and os.path.exists(model_path):
            ckpt = model_path
        else:
            ckpt = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{self.config_name.replace('-', '_')}.ckpt", repo_type="model", cache_dir=self.cache_dir)

        state_dict = torch.load(ckpt, map_location='cpu')['state_dict']
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
        self.model.load_state_dict(state_dict, strict=True)

        self.model = self.model.to(self.device)
        if self.config_name.startswith('instant-mesh'):
            try:
                self.model.init_flexicubes_geometry(self.device, fovy=30.0)
            except Exception:
                pass
        self.model = self.model.eval()

        # initialize rembg session
        self.rembg_session = rembg.new_session()

    def _prepare_image(self, image_path: str, remove_bg: bool = True):
        img = Image.open(image_path).convert('RGBA')
        if remove_bg:
            img = remove_background(img, self.rembg_session)
            img = resize_foreground(img, 0.85)
        return img

    def generate_multiview(self, input_image: Image.Image, steps: int = 75, generator: Optional[torch.Generator] = None):
        # sampling
        out = self.pipeline(input_image, num_inference_steps=steps, generator=generator)
        return out.images[0]

    def reconstruct_and_export(self, images_numpy: np.ndarray, output_basename: str, output_folder: str,
                               export_texmap: bool = False, save_video_flag: bool = False, distance: float = 4.5, view: int = 6):
        os.makedirs(output_folder, exist_ok=True)
        mesh_obj_path = os.path.join(output_folder, f"{output_basename}.obj")
        mesh_glb_path = os.path.join(output_folder, f"{output_basename}.glb")

        images = torch.from_numpy(images_numpy).permute(2, 0, 1).contiguous().float()
        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(self.device)

        images = images.unsqueeze(0).to(self.device)
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

        if view == 4:
            indices = torch.tensor([0, 2, 4, 5]).long().to(self.device)
            images = images[:, indices]
            input_cameras = input_cameras[:, indices]

        # detect optional libraries required for texture/UV export
        have_xatlas = True
        have_nvdiffrast = True
        try:
            import xatlas  # noqa: F401
        except Exception:
            have_xatlas = False
        try:
            import nvdiffrast  # noqa: F401
        except Exception:
            have_nvdiffrast = False

        if export_texmap and not (have_xatlas and have_nvdiffrast):
            print('Warning: texture export requested but xatlas/nvdiffrast not available — falling back to vertex-color export')
            export_texmap = False

        with torch.no_grad():
            planes = self.model.forward_planes(images, input_cameras)

            mesh_out = self.model.extract_mesh(
                planes,
                use_texture_map=export_texmap,
                **self.infer_config,
            )

            # Handle textured and non-textured outputs robustly
            if export_texmap:
                try:
                    vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                    save_obj_with_mtl(
                        vertices.data.cpu().numpy(),
                        uvs.data.cpu().numpy(),
                        faces.data.cpu().numpy(),
                        mesh_tex_idx.data.cpu().numpy(),
                        tex_map.permute(1, 2, 0).data.cpu().numpy(),
                        mesh_obj_path,
                    )
                    try:
                        save_glb(vertices.data.cpu().numpy(), faces.data.cpu().numpy(), None, mesh_glb_path)
                    except Exception:
                        pass
                except Exception as e:
                    print('Failed to export textured mesh, falling back to vertex colors. Error:', e)
                    export_texmap = False

            if not export_texmap:
                # vertex-color export (works without xatlas/nvdiffrast)
                # mesh_out might be (vertices, faces, vertex_colors) or include extra fields — pick first three
                try:
                    vertices, faces, vertex_colors = mesh_out if len(mesh_out) >= 3 else (mesh_out[0], mesh_out[1], None)
                except Exception:
                    # Fallback if mesh_out has unexpected format
                    vertices, faces, vertex_colors = mesh_out

                try:
                    if vertex_colors is not None:
                        save_obj(vertices.data.cpu().numpy(), faces.data.cpu().numpy(), vertex_colors.data.cpu().numpy(), mesh_obj_path)
                        save_glb(vertices.data.cpu().numpy(), faces.data.cpu().numpy(), vertex_colors.data.cpu().numpy(), mesh_glb_path)
                    else:
                        # If no vertex colors, save geometry only
                        save_obj(vertices.data.cpu().numpy(), faces.data.cpu().numpy(), None, mesh_obj_path)
                        save_glb(vertices.data.cpu().numpy(), faces.data.cpu().numpy(), None, mesh_glb_path)
                except Exception as e:
                    print('Mesh save failed:', e)

            if save_video_flag:
                render_cameras = get_render_cameras(batch_size=1, M=120, radius=distance, elevation=20.0, is_flexicubes=self.config_name.startswith('instant-mesh')).to(self.device)
                chunk_size = 20 if self.config_name.startswith('instant-mesh') else 1
                frames = render_frames(self.model, planes, render_cameras=render_cameras, render_size=self.infer_config.render_resolution, chunk_size=chunk_size, is_flexicubes=self.config_name.startswith('instant-mesh'))
                video_path = os.path.join(output_folder, f"{output_basename}.mp4")
                save_video(frames, video_path, fps=30)

        return mesh_obj_path, mesh_glb_path

    def process_images(self, image_paths: List[str], output_dir: str, diffusion_steps: int = 75, remove_bg: bool = True,
                       export_texmap: bool = False, save_video_flag: bool = False, distance: float = 4.5, view: int = 6):
        if self.pipeline is None or self.model is None:
            self.load_models()

        os.makedirs(output_dir, exist_ok=True)
        results = []
        generator = torch.Generator(device=self.device)

        for idx, image_path in enumerate(tqdm(image_paths, desc='Processing images')):
            base = os.path.basename(image_path).split('.')[0]
            print(f'[{idx+1}/{len(image_paths)}] Processing {base}...')

            input_image = self._prepare_image(image_path, remove_bg=remove_bg)
            z123_image = self.generate_multiview(input_image, steps=diffusion_steps, generator=generator)

            save_img_path = os.path.join(output_dir, f'{base}.png')
            z123_image.save(save_img_path)

            images_np = np.asarray(z123_image, dtype=np.float32) / 255.0
            obj_path, glb_path = self.reconstruct_and_export(images_np, base, output_dir, export_texmap, save_video_flag, distance, view)

            # Strict memory cleanup per iteration
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

            results.append({'name': base, 'image': save_img_path, 'obj': obj_path, 'glb': glb_path})

        return results


def _gather_input_files(input_path: str):
    if os.path.isdir(input_path):
        files = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
    else:
        files = [input_path]
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('input_path', type=str, help='Path to input image or directory.')
    parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
    parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
    parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
    parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
    parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
    parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
    parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
    args = parser.parse_args()

    seed_everything(args.seed)

    proc = InstantMeshProcessor(args.config)
    files = _gather_input_files(args.input_path)
    proc.load_models()
    results = proc.process_images(files, args.output_path, diffusion_steps=args.diffusion_steps, remove_bg=not args.no_rembg, export_texmap=args.export_texmap, save_video_flag=args.save_video, distance=args.distance, view=args.view)

    print('Done. Results:')
    for r in results:
        print(r)


if __name__ == '__main__':
    main()
