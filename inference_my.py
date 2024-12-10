import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)
import argparse
import numpy as np
import torch
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDIMScheduler, MarigoldNormalsPipeline
from omegaconf import OmegaConf
from PIL import Image
import os
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_magicman import MagicManPipeline
from src.utils.util import get_camera
from src.utils.util import (
    preprocess_image,
    save_image_seq,
)
import sys

sys.path.append("./thirdparties/econ")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, default="configs/inference/inference-plus.yaml"
    )
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--input_path", type=str, default="examples/001.jpg")
    parser.add_argument("--output_path", type=str, default="examples/001")
    args = parser.parse_args()

    return args


def init_module(args, config):
    device = args.device
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    # VAE
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    # image encoder
    image_encoder = None

    # reference unet
    reference_unet = UNet2DConditionModel.from_pretrained_2d(
        config.pretrained_unet_path,
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs,
            resolve=True,
        ),
    ).to(dtype=weight_dtype, device=device)

    # denoising unet
    if config.unet_additional_kwargs.use_motion_module:
        mm_path = config.pretrained_motion_module_path
    else:
        mm_path = ""
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_unet_path,
        mm_path,
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs,
            resolve=True,
        ),
    ).to(dtype=weight_dtype, device=device)

    # pose guider for normal maps & semantic segmentation maps
    semantic_guider = PoseGuider(**config.pose_guider_kwargs).to(device="cuda")
    normal_guider = PoseGuider(**config.pose_guider_kwargs).to(device="cuda")

    # scheduler
    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    # random generator
    generator = torch.manual_seed(args.seed)

    # load pretrained weights
    ckpt_path = config.ckpt_path
    denoising_unet.load_state_dict(
        torch.load(os.path.join(ckpt_path, f"denoising_unet.pth"), map_location="cpu"),
    )
    reference_unet.load_state_dict(
        torch.load(os.path.join(ckpt_path, f"reference_unet.pth"), map_location="cpu"),
    )
    semantic_guider.load_state_dict(
        torch.load(
            os.path.join(ckpt_path, f"semantic_guider.pth"),
            map_location="cpu",
        ),
    )
    normal_guider.load_state_dict(
        torch.load(
            os.path.join(ckpt_path, f"normal_guider.pth"),
            map_location="cpu",
        ),
    )

    return (
        vae,
        image_encoder,
        reference_unet,
        denoising_unet,
        semantic_guider,
        normal_guider,
        scheduler,
        generator,
    )


def init_pipeline(
    vae,
    image_encoder,
    reference_unet,
    denoising_unet,
    semantic_guider,
    normal_guider,
    scheduler,
    unet_attention_mode,
    weight_dtype,
    device,
):
    pipe = MagicManPipeline(
        vae=vae,
        image_encoder=image_encoder,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        semantic_guider=semantic_guider,
        normal_guider=normal_guider,
        scheduler=scheduler,
        unet_attention_mode=unet_attention_mode,
    )
    pipe = pipe.to(device, dtype=weight_dtype)

    vae.eval()
    if image_encoder is not None:
        image_encoder.eval()
    reference_unet.eval()
    denoising_unet.eval()
    semantic_guider.eval()
    normal_guider.eval()

    return pipe


def init_camera(num_views):
    clip_interval = 360 // num_views
    azim_list = []
    elev_list = []
    camera_list = []
    for i in range(num_views):
        azim = -float(i * clip_interval)
        elev = 0.0
        azim_list.append(azim)
        elev_list.append(elev)
    for azim, elev in zip(azim_list, elev_list):
        camera = get_camera(elev, azim)
        camera_list.append(camera)
    cameras = np.stack(camera_list, axis=0)  # (f, 4, 4)
    ref_camera = get_camera(0.0, 0.0)  # (4, 4)
    return azim_list, elev_list, cameras, ref_camera


def init_ref_normal(rgb_pil, mask_pil, method="marigold", device="cuda:0"):
    if method == "marigold":
        pipe = MarigoldNormalsPipeline.from_pretrained(
            "prs-eth/marigold-normals-v0-1", variant="fp16", torch_dtype=torch.float16
        ).to(device)
        normal_np = pipe(rgb_pil, num_inference_steps=25).prediction
        mask_np = np.array(mask_pil)[None, :, :, None]

        def normalize_normal_map(normal_np):
            norms = np.linalg.norm(normal_np, axis=-1, keepdims=True)
            normal_np = normal_np / norms
            normal_np = (normal_np + 1.0) / 2.0
            return normal_np

        # normalize & mask bg
        normal_np = normalize_normal_map(normal_np)
        normal_np = normal_np * (mask_np > 0)
        normal_pil = Image.fromarray((normal_np[0] * 255).astype(np.uint8)).convert(
            "RGB"
        )

        del pipe
        torch.cuda.empty_cache()

        return normal_pil

    else:
        raise NotImplementedError


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    device = args.device
    width, height = args.W, args.H
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    # module initialization
    (
        vae,
        image_encoder,
        reference_unet,
        denoising_unet,
        semantic_guider,
        normal_guider,
        scheduler,
        generator,
    ) = init_module(args, config)

    # pipeline initialization
    pipe = init_pipeline(
        vae=vae,
        image_encoder=image_encoder,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        semantic_guider=semantic_guider,
        normal_guider=normal_guider,
        scheduler=scheduler,
        unet_attention_mode=config.unet_attention_mode,
        weight_dtype=weight_dtype,
        device=device,
    )

    # camera initialization
    num_views = config.num_views
    azim_list, elev_list, cameras, ref_camera = init_camera(num_views)

    input_path = args.input_path
    output_path = args.output_path
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    ##0## reference image preparation
    ref_rgb_pil = Image.open(input_path).convert("RGB")
    ref_rgb_pil, ref_mask_pil = preprocess_image(
        ref_rgb_pil
    )  # remove background & resize & center
    ref_normal_pil = init_ref_normal(
        ref_rgb_pil, ref_mask_pil, method="marigold", device=device
    )  # initilize reference normal map
    ref_rgb_pil.save(os.path.join(output_path, f"ref_rgb.png"))
    ref_mask_pil.save(os.path.join(output_path, f"ref_mask.png"))
    ref_normal_pil.save(os.path.join(output_path, f"ref_normal.png"))

    ##2## Initialize NVS w/o SMPL-X
    output = pipe(
        # ref rgb/normal
        ref_rgb_pil,
        ref_normal_pil,
        # cond semantic/normal
        None,
        None,
        cameras,
        ref_camera,
        width,
        height,
        num_views,
        num_inference_steps=config.intermediate_denoising_steps,
        guidance_scale=config.cfg_scale,
        smplx_guidance_scale=0.0,
        guidance_rescale=config.guidance_rescale,
        generator=generator,
    )  # (b=1, c, f, h, w)
    rgb_video = output.rgb_videos

    save_image_seq(rgb_video, os.path.join(output_path, f"images"))


if __name__ == "__main__":
    main()
