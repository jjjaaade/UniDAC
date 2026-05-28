import argparse
import json
import math
import os
import os.path as osp
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from unidac.dataloaders.dataset import resize_for_input
from unidac.models.unidac import UniDAC
from unidac.utils.erp_geometry import (cam_to_erp_patch_fast,
                                       erp_patch_to_cam_fast)

NORMALIZATION_STATS = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniDAC custom single-image inference")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--intrinsics-json", type=str, required=True)
    parser.add_argument("--model-file", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--erp-height", type=int, default=None, help="Override canonical ERP height"
    )
    parser.add_argument(
        "--input-height", type=int, default=None, help="Override model input height"
    )
    parser.add_argument(
        "--input-width", type=int, default=None, help="Override model input width"
    )
    parser.add_argument(
        "--undistort-alpha",
        type=float,
        default=0.0,
        help="OpenCV alpha for getOptimalNewCameraMatrix; 0 keeps valid pixels, 1 keeps FoV",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    return parser.parse_args()


def load_rgb_image(image_file: str) -> np.ndarray:
    return np.asarray(Image.open(image_file).convert("RGB"))


def load_intrinsics(
    intrinsics_json: str, image_shape: Tuple[int, int]
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    with open(intrinsics_json, "r", encoding="utf-8") as handle:
        intrinsics = json.load(handle)

    if intrinsics.get("camera_model") != "PINHOLE":
        raise ValueError(
            f"demo_unidac_custom.py currently supports only PINHOLE intrinsics, got {intrinsics.get('camera_model')}"
        )

    image_h, image_w = image_shape
    json_w = intrinsics.get("width")
    json_h = intrinsics.get("height")
    if (
        json_w is not None
        and json_h is not None
        and (json_w != image_w or json_h != image_h)
    ):
        raise ValueError(
            f"Image size {image_w}x{image_h} does not match intrinsics JSON size {json_w}x{json_h}"
        )

    camera_matrix = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dist_coeffs = np.array(
        [
            intrinsics.get("k1", 0.0),
            intrinsics.get("k2", 0.0),
            intrinsics.get("p1", 0.0),
            intrinsics.get("p2", 0.0),
            intrinsics.get("k3", 0.0),
        ],
        dtype=np.float32,
    )
    return intrinsics, camera_matrix, dist_coeffs


def load_config(config_file: str, input_size: Tuple[int, int] | None) -> Dict:
    with open(config_file, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    if input_size is not None:
        config["data"]["fwd_sz"] = list(input_size)
        config["model"]["pixel_encoder"]["img_size"] = list(input_size)

    config["model"]["pixel_encoder"]["pretrained"] = None
    return config


def build_model(config: Dict, model_file: str, device: torch.device) -> UniDAC:
    model = UniDAC.build(config)
    model.load_pretrained(model_file)
    model = model.to(device)
    model.eval()
    return model


def undistort_image(
    image_rgb: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    image_h, image_w = image_rgb.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (image_w, image_h),
        alpha,
        (image_w, image_h),
    )
    undistorted = cv2.undistort(
        image_rgb, camera_matrix, dist_coeffs, None, new_camera_matrix
    )
    return undistorted, new_camera_matrix.astype(np.float32)


def compute_erp_crop(
    undistorted_camera_matrix: np.ndarray,
    image_shape: Tuple[int, int],
    erp_height: int,
) -> Tuple[int, int, float, float]:
    image_h, image_w = image_shape
    fx = float(undistorted_camera_matrix[0, 0])
    fy = float(undistorted_camera_matrix[1, 1])
    h_fov = 2.0 * math.atan(image_w / (2.0 * fx))
    v_fov = 2.0 * math.atan(image_h / (2.0 * fy))
    patch_w = max(16, int(round(erp_height * h_fov / math.pi)))
    patch_h = max(16, int(round(erp_height * v_fov / math.pi)))
    return patch_h, patch_w, h_fov, v_fov


def prepare_erp_batch(
    undistorted_image: np.ndarray,
    undistorted_camera_matrix: np.ndarray,
    config: Dict,
    erp_height: int,
) -> Tuple[Dict, Dict, np.ndarray]:
    image_h, image_w = undistorted_image.shape[:2]
    erp_width = erp_height * 2
    patch_h, patch_w, h_fov, v_fov = compute_erp_crop(
        undistorted_camera_matrix, (image_h, image_w), erp_height
    )

    cam_params = {
        "dataset": "custom",
        "camera_model": "PINHOLE",
        "fx": float(undistorted_camera_matrix[0, 0]),
        "fy": float(undistorted_camera_matrix[1, 1]),
        "cx": float(undistorted_camera_matrix[0, 2]),
        "cy": float(undistorted_camera_matrix[1, 2]),
    }

    image_float = undistorted_image.astype(np.float32) / 255.0
    dummy_depth = np.zeros((image_h, image_w, 1), dtype=np.float32)
    dummy_mask = np.zeros((image_h, image_w, 1), dtype=np.float32)

    erp_image, erp_depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
        image_float,
        dummy_depth,
        dummy_mask,
        theta=0.0,
        phi=0.0,
        patch_h=patch_h,
        patch_w=patch_w,
        erp_h=erp_height,
        erp_w=erp_width,
        cam_params=cam_params,
        roll=0.0,
        scale_fac=None,
    )

    fwd_sz = tuple(config["data"]["fwd_sz"])
    erp_input, erp_depth, pad, pred_scale_factor, attn_mask, lat_grid, long_grid = (
        resize_for_input(
            (erp_image * 255.0).astype(np.uint8),
            erp_depth,
            fwd_sz,
            None,
            [erp_image.shape[0], erp_image.shape[1]],
            1.0,
            padding_rgb=[0, 0, 0],
            mask=erp_mask,
            lat_grid=latitude,
            long_grid=longitude,
        )
    )

    batch = {
        "image": TF.normalize(TF.to_tensor(erp_input), **NORMALIZATION_STATS).unsqueeze(
            0
        ),
        "gt": TF.to_tensor(erp_depth).unsqueeze(0),
        "mask": TF.to_tensor((erp_depth > 0.01).astype(np.uint8)).unsqueeze(0),
        "attn_mask": TF.to_tensor((attn_mask > 0).astype(np.float32)).unsqueeze(0),
        "lat_range": torch.tensor(
            [[float(np.min(latitude)), float(np.max(latitude))]], dtype=torch.float32
        ),
        "long_range": torch.tensor(
            [[float(np.min(longitude)), float(np.max(longitude))]], dtype=torch.float32
        ),
        "lat_grid": torch.tensor(lat_grid, dtype=torch.float32).unsqueeze(0),
        "long_grid": torch.tensor(long_grid, dtype=torch.float32).unsqueeze(0),
        "info": {
            "pred_scale_factor": float(pred_scale_factor),
        },
    }

    metadata = {
        "cam_params": cam_params,
        "erp_height": erp_height,
        "erp_width": erp_width,
        "patch_height": patch_h,
        "patch_width": patch_w,
        "pred_scale_factor": float(pred_scale_factor),
        "h_fov_deg": math.degrees(h_fov),
        "v_fov_deg": math.degrees(v_fov),
        "pad": [int(v) for v in pad],
    }
    return batch, metadata, erp_input


def run_inference(model: UniDAC, batch: Dict, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        preds, _, _ = model(
            batch["image"].to(device),
            batch["lat_range"].to(device),
            batch["long_range"].to(device),
            attn_mask=batch["attn_mask"].to(device),
            lat_grid=batch["lat_grid"].to(device),
        )
    preds *= batch["info"]["pred_scale_factor"]
    return preds.cpu()


def project_erp_depth_to_undistorted(
    batch: Dict,
    preds: torch.Tensor,
    metadata: Dict,
    output_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_h, out_w = output_shape
    pred_scale = metadata["pred_scale_factor"]
    erp_height_scaled = metadata["erp_height"] * pred_scale
    erp_width_scaled = metadata["erp_width"] * pred_scale

    img_out, depth_out, _, active_mask = erp_patch_to_cam_fast(
        batch["image"][0],
        preds[0],
        batch["attn_mask"][0],
        theta=0.0,
        phi=0.0,
        out_h=out_h,
        out_w=out_w,
        erp_h=erp_height_scaled,
        erp_w=erp_width_scaled,
        cam_params=metadata["cam_params"],
    )

    rgb_tensor = img_out.squeeze(0).cpu()
    depth_np = depth_out.squeeze().cpu().numpy().astype(np.float32)
    mask_np = active_mask.squeeze().cpu().numpy().astype(np.float32)
    return denormalize_tensor_image(rgb_tensor), depth_np, mask_np


def project_depth_to_original_view(
    undistorted_depth: np.ndarray,
    undistorted_mask: np.ndarray,
    original_shape: Tuple[int, int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    undistorted_camera_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    out_h, out_w = original_shape
    grid_x, grid_y = np.meshgrid(
        np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32)
    )
    pixel_coords = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2)
    undistorted_coords = cv2.undistortPoints(
        pixel_coords, camera_matrix, dist_coeffs, P=undistorted_camera_matrix
    )
    map_x = undistorted_coords[:, 0, 0].reshape(out_h, out_w).astype(np.float32)
    map_y = undistorted_coords[:, 0, 1].reshape(out_h, out_w).astype(np.float32)

    depth_original = cv2.remap(
        undistorted_depth,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_original = cv2.remap(
        undistorted_mask.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    depth_original *= (valid_original > 0.5).astype(np.float32)
    return depth_original.astype(np.float32), valid_original.astype(np.float32)


def denormalize_tensor_image(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(NORMALIZATION_STATS["mean"], dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(NORMALIZATION_STATS["std"], dtype=image.dtype).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0.0, 1.0)
    return (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def depth_to_uint16(depth: np.ndarray) -> np.ndarray:
    return np.clip(depth * 1000.0, 0.0, np.iinfo(np.uint16).max).astype(np.uint16)


def colorize_depth(depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    depth_vis = depth.copy()
    if mask is not None:
        depth_vis = np.where(mask > 0, depth_vis, 0.0)
    positive = depth_vis[depth_vis > 0]
    vmax = float(np.percentile(positive, 95)) if positive.size else 1.0
    depth_norm = np.clip(depth_vis / max(vmax, 1e-6), 0.0, 1.0)
    depth_uint8 = (depth_norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    if mask is not None:
        color[mask <= 0] = 0
    return color


def make_overlay(
    image_rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    depth_color = colorize_depth(depth, mask)
    overlay = image_rgb.copy()
    valid = mask > 0
    if np.any(valid):
        blended = cv2.addWeighted(image_rgb, 0.55, depth_color, 0.45, 0.0)
        overlay[valid] = blended[valid]
    return overlay


def save_outputs(
    out_dir: str,
    original_image: np.ndarray,
    undistorted_image: np.ndarray,
    erp_input: np.ndarray,
    erp_depth: np.ndarray,
    undistorted_depth: np.ndarray,
    undistorted_mask: np.ndarray,
    original_depth: np.ndarray,
    original_mask: np.ndarray,
    overlay: np.ndarray,
    metadata: Dict,
    intrinsics: Dict,
    undistorted_camera_matrix: np.ndarray,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(original_image).save(osp.join(out_dir, "original_rgb.png"))
    Image.fromarray(undistorted_image).save(osp.join(out_dir, "undistorted_rgb.png"))
    Image.fromarray(erp_input).save(osp.join(out_dir, "erp_model_input.png"))
    Image.fromarray(colorize_depth(erp_depth)).save(
        osp.join(out_dir, "erp_depth_pred.png")
    )
    Image.fromarray(colorize_depth(undistorted_depth, undistorted_mask)).save(
        osp.join(out_dir, "undistorted_depth_pred.png")
    )
    Image.fromarray(colorize_depth(original_depth, original_mask)).save(
        osp.join(out_dir, "original_view_depth_pred.png")
    )
    Image.fromarray(overlay).save(osp.join(out_dir, "original_view_depth_overlay.png"))

    cv2.imwrite(osp.join(out_dir, "erp_depth_pred_mm.png"), depth_to_uint16(erp_depth))
    cv2.imwrite(
        osp.join(out_dir, "undistorted_depth_pred_mm.png"),
        depth_to_uint16(undistorted_depth),
    )
    cv2.imwrite(
        osp.join(out_dir, "original_view_depth_pred_mm.png"),
        depth_to_uint16(original_depth),
    )
    np.save(osp.join(out_dir, "erp_depth_pred.npy"), erp_depth)
    np.save(osp.join(out_dir, "undistorted_depth_pred.npy"), undistorted_depth)
    np.save(osp.join(out_dir, "original_view_depth_pred.npy"), original_depth)

    computed = {
        "input_image": osp.abspath(osp.join(out_dir, "original_rgb.png")),
        "undistorted_image": osp.abspath(osp.join(out_dir, "undistorted_rgb.png")),
        "erp_model_input": osp.abspath(osp.join(out_dir, "erp_model_input.png")),
        "erp_depth_pred": osp.abspath(osp.join(out_dir, "erp_depth_pred.npy")),
        "undistorted_depth_pred": osp.abspath(
            osp.join(out_dir, "undistorted_depth_pred.npy")
        ),
        "original_view_depth_pred": osp.abspath(
            osp.join(out_dir, "original_view_depth_pred.npy")
        ),
        "overlay": osp.abspath(osp.join(out_dir, "original_view_depth_overlay.png")),
        "input_intrinsics": intrinsics,
        "undistorted_intrinsics": {
            "fx": float(undistorted_camera_matrix[0, 0]),
            "fy": float(undistorted_camera_matrix[1, 1]),
            "cx": float(undistorted_camera_matrix[0, 2]),
            "cy": float(undistorted_camera_matrix[1, 2]),
        },
        **metadata,
    }
    with open(osp.join(out_dir, "run_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(computed, handle, indent=2)


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    image_file = osp.abspath(args.image_file)
    intrinsics_json = osp.abspath(args.intrinsics_json)
    model_file = osp.abspath(args.model_file)
    config_file = osp.abspath(args.config_file)
    out_dir = osp.abspath(args.out_dir)

    original_image = load_rgb_image(image_file)
    intrinsics, camera_matrix, dist_coeffs = load_intrinsics(
        intrinsics_json, original_image.shape[:2]
    )

    input_size = None
    if args.input_height is not None or args.input_width is not None:
        if args.input_height is None or args.input_width is None:
            raise ValueError(
                "--input-height and --input-width must be provided together"
            )
        input_size = (args.input_height, args.input_width)

    config = load_config(config_file, input_size=input_size)
    model = build_model(config, model_file, device)

    erp_height = int(args.erp_height or config["data"]["cano_sz"][0])
    undistorted_image, undistorted_camera_matrix = undistort_image(
        original_image,
        camera_matrix,
        dist_coeffs,
        args.undistort_alpha,
    )
    batch, metadata, erp_input = prepare_erp_batch(
        undistorted_image,
        undistorted_camera_matrix,
        config,
        erp_height,
    )
    preds = run_inference(model, batch, device)

    undistorted_rgb_from_erp, undistorted_depth, undistorted_mask = (
        project_erp_depth_to_undistorted(
            batch,
            preds,
            metadata,
            output_shape=original_image.shape[:2],
        )
    )
    original_depth, original_mask = project_depth_to_original_view(
        undistorted_depth,
        undistorted_mask,
        original_shape=original_image.shape[:2],
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        undistorted_camera_matrix=undistorted_camera_matrix,
    )
    overlay = make_overlay(original_image, original_depth, original_mask)

    save_outputs(
        out_dir=out_dir,
        original_image=original_image,
        undistorted_image=undistorted_image,
        erp_input=erp_input,
        erp_depth=preds[0, 0].numpy().astype(np.float32),
        undistorted_depth=undistorted_depth,
        undistorted_mask=undistorted_mask,
        original_depth=original_depth,
        original_mask=original_mask,
        overlay=overlay,
        metadata=metadata,
        intrinsics=intrinsics,
        undistorted_camera_matrix=undistorted_camera_matrix,
    )

    print(f"Saved outputs to {out_dir}")
    print(
        f"Computed undistorted intrinsics: fx={undistorted_camera_matrix[0, 0]:.4f}, fy={undistorted_camera_matrix[1, 1]:.4f}, cx={undistorted_camera_matrix[0, 2]:.4f}, cy={undistorted_camera_matrix[1, 2]:.4f}"
    )
    print(
        f"ERP crop FoV: h={metadata['h_fov_deg']:.2f} deg, v={metadata['v_fov_deg']:.2f} deg"
    )
    print(
        f"ERP patch size before resize: {metadata['patch_width']}x{metadata['patch_height']}"
    )
    print(f"ERP model input size: {erp_input.shape[1]}x{erp_input.shape[0]}")


if __name__ == "__main__":
    main(parse_args())
