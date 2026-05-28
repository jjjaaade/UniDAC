#!/usr/bin/env python3
"""
Standalone custom inference demo for UniDAC.

Processes a single image (--image-file) or every image under a directory
(--input-dir) using user-supplied camera intrinsics (--intrinsics-json).
No AnyCalib / DINOv3 checkpoint dependency.

Three artifacts are saved per image to --out-dir:

  <stem>_undistorted.jpg   -- undistorted camera-view reference image
                          (pinhole projection of the fisheye input;
                           NOT the ERP image that the model sees)
  <stem>_depth_cam.jpg     -- predicted depth converted back from ERP to
                          the original camera/image format (with colorbar
                          showing physical scale in metres)
  <stem>_overlay.jpg       -- depth colormap overlaid on the undistorted
                          image for alignment inspection

Usage examples:
  # Single image
  python demo/demo_unidac_wild.py \\
      --image-file  demo/input/scannetpp_rgb.jpg \\
      --intrinsics-json demo/input/custom_intrinsics.json \\
      --model-file  checkpoints/unidac.pt \\
      --out-dir     demo/output_custom

  # Directory of images
  python demo/demo_unidac_wild.py \\
      --input-dir   /path/to/images/ \\
      --intrinsics-json demo/input/custom_intrinsics.json \\
      --model-file  checkpoints/unidac.pt \\
      --out-dir     demo/output_custom \\
      --depth-max   40.0
"""

import os
import json
import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from PIL import Image

from unidac.models.unidac import UniDAC
from unidac.utils.erp_geometry import erp_patch_to_cam_fast, cam_to_erp_patch_fast
from unidac.dataloaders.dataset import resize_for_input

# Common image extensions to consider when scanning a directory
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# ImageNet normalisation constants (must match model training)
_IMAGENET_MEAN = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
_IMAGENET_STD  = np.array([58.395,  57.12,   57.375],  dtype=np.float32)

# Default config that defines model architecture and cano_sz
_DEFAULT_CONFIG = 'configs/test/dac_dinov3l+dpt_indoor_test_scannetpp.json'


# ---------------------------------------------------------------------------
# Intrinsics helpers
# ---------------------------------------------------------------------------

def load_intrinsics(json_path: str) -> dict:
    """
    Load camera intrinsics from a JSON file.

    Required keys for all models:
        camera_model  (str)  -- 'OPENCV_FISHEYE' or 'PINHOLE'
        fl_x, fl_y   (float) -- focal lengths in pixels
        cx, cy        (float) -- principal point in pixels

    Additional keys required for OPENCV_FISHEYE:
        k1, k2, k3, k4  (float) -- radial distortion coefficients

    Optional keys (override CLI defaults if present):
        crop_wFoV   (float) -- horizontal FOV for ERP crop in degrees
        fwd_sz      [H, W]  -- model input patch size in pixels
    """
    with open(json_path, 'r') as f:
        intr = json.load(f)

    required = ['camera_model', 'fl_x', 'fl_y', 'cx', 'cy']
    for key in required:
        if key not in intr:
            raise ValueError(f"Intrinsics JSON is missing required key: '{key}'")

    if intr['camera_model'] == 'OPENCV_FISHEYE':
        for key in ['k1', 'k2', 'k3', 'k4']:
            if key not in intr:
                raise ValueError(
                    f"OPENCV_FISHEYE intrinsics JSON is missing distortion key: '{key}'")

    return intr


def build_cam_params(intr: dict) -> dict:
    """Convert loaded intrinsics dict to the cam_params format expected by erp_geometry."""
    cam_params = {
        'dataset':       'custom',
        'camera_model':  intr['camera_model'],
        # Both fl_x/fl_y (used by OPENCV_FISHEYE path) and fx/fy (used by
        # pinhole path) are populated so the dict works with either code path.
        'fl_x':  intr['fl_x'],
        'fl_y':  intr['fl_y'],
        'fx':    intr['fl_x'],
        'fy':    intr['fl_y'],
        'cx':    intr['cx'],
        'cy':    intr['cy'],
    }
    if intr['camera_model'] == 'OPENCV_FISHEYE':
        cam_params.update({
            'k1': intr['k1'],
            'k2': intr['k2'],
            'k3': intr['k3'],
            'k4': intr['k4'],
        })
    return cam_params


# ---------------------------------------------------------------------------
# Camera-space back-projection helpers
# ---------------------------------------------------------------------------

def compute_undistorted_K(cam_params: dict, img_h: int, img_w: int) -> np.ndarray:
    """
    Compute a new pinhole camera matrix K_new that describes the undistorted
    output image space for an OPENCV_FISHEYE camera.  Returns K_new (3x3).
    For a PINHOLE camera K_new equals K itself.
    """
    K = np.array([[cam_params['fl_x'], 0,                cam_params['cx']],
                  [0,                  cam_params['fl_y'], cam_params['cy']],
                  [0,                  0,                  1             ]],
                 dtype=np.float64)

    if cam_params['camera_model'] == 'OPENCV_FISHEYE':
        D = np.array([cam_params['k1'], cam_params['k2'],
                      cam_params['k3'], cam_params['k4']], dtype=np.float64)
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (img_w, img_h), np.eye(3), balance=0.0)
    else:
        K_new = K.copy()

    return K_new


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def decode_erp_image(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Decode a normalised [C, H, W] float tensor (ImageNet stats) to a
    uint8 RGB numpy array [H, W, 3].
    """
    img_np = img_tensor.squeeze().cpu().float().numpy()          # [C, H, W]
    img_np = img_np * _IMAGENET_STD[:, None, None] + _IMAGENET_MEAN[:, None, None]
    return img_np.transpose(1, 2, 0).clip(0, 255).astype(np.uint8)  # [H, W, 3]


def save_depth_with_colorbar(
        depth_np: np.ndarray,
        mask_np: np.ndarray,
        filepath: str,
        depth_max: float,
) -> None:
    """
    Save a depth map as an image with a colorbar showing physical scale (m).
    Pixels where mask_np == 0 are zeroed out before plotting.
    """
    depth_vis = depth_np.copy()
    depth_vis[mask_np == 0] = 0.0

    cmap = cm.magma_r
    norm = mcolors.Normalize(vmin=0, vmax=depth_max)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(depth_vis, cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (m)', fontsize=11)
    ax.set_title('Predicted depth — camera space', fontsize=12)
    ax.axis('off')
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)


def save_depth_overlay(
        rgb_np: np.ndarray,
        depth_np: np.ndarray,
        mask_np: np.ndarray,
        filepath: str,
        depth_max: float,
        alpha: float = 0.55,
) -> None:
    """
    Save an overlay of a colourised depth map on top of the undistorted
    reference image.  Only pixels inside the camera FOV (mask_np > 0) are
    blended; the rest keeps the original RGB.  A colorbar is included so
    the physical scale is readable directly from the overlay image.
    """
    cmap = cm.magma_r
    norm = mcolors.Normalize(vmin=0, vmax=depth_max)

    depth_vis = depth_np.copy()
    depth_vis[mask_np == 0] = 0.0
    depth_color = (cmap(norm(depth_vis))[:, :, :3] * 255).astype(np.uint8)

    valid = mask_np > 0
    overlay = rgb_np.copy()
    overlay[valid] = (
        (1.0 - alpha) * rgb_np[valid].astype(np.float32)
        + alpha       * depth_color[valid].astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    # Add colorbar via a ScalarMappable so the depth scale is visible
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.imshow(overlay)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (m)', fontsize=11)
    ax.set_title('Depth overlay on undistorted image (alignment check)', fontsize=12)
    ax.axis('off')
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core per-image pipeline
# ---------------------------------------------------------------------------

def process_image(
        model: torch.nn.Module,
        device: torch.device,
        image_path: str,
        cam_params: dict,
        cano_sz: list,
        fwd_sz: list,
        crop_wFoV: float,
        depth_max: float,
        out_dir: str,
) -> None:
    """Run the full inference pipeline for a single image and write outputs."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    print(f"  [1/4] Loading image: {os.path.basename(image_path)}")
    image = np.asarray(Image.open(image_path).convert('RGB'))
    org_h, org_w = image.shape[:2]
    print(f"        Image size: {org_w}x{org_h}")

    # No GT depth is available for custom images — use a zero placeholder.
    depth_placeholder = np.zeros((org_h, org_w), dtype=np.float32)

    # ------------------------------------------------------------------
    # Step 1: fisheye image → ERP patch
    # ------------------------------------------------------------------
    print(f"  [2/4] Projecting to ERP patch (crop_wFoV={crop_wFoV}°) ...")
    phi  = np.array(0, dtype=np.float32)
    roll = np.array(0, dtype=np.float32)
    theta = 0.0

    image_f = image.astype(np.float32) / 255.0
    depth_e = depth_placeholder[:, :, np.newaxis]
    mask_valid = (depth_e > 0.01).astype(np.float32)

    crop_width  = int(cano_sz[0] * crop_wFoV / 180.0)
    crop_height = int(crop_width * fwd_sz[0] / fwd_sz[1])

    erp_img, erp_depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
        image_f, depth_e, mask_valid,
        theta, phi, crop_height, crop_width,
        cano_sz[0], cano_sz[0] * 2,
        cam_params, roll, scale_fac=None,
    )
    lat_range  = torch.tensor([float(latitude.min()),  float(latitude.max())])
    long_range = torch.tensor([float(longitude.min()), float(longitude.max())])

    erp_img_u8 = (erp_img * 255.0).astype(np.uint8)
    (erp_img_u8, erp_depth_r,
     _pad, pred_scale_factor,
     attn_mask, lat_grid, long_grid) = resize_for_input(
        erp_img_u8, erp_depth, fwd_sz, None,
        [erp_img.shape[0], erp_img.shape[1]], 1.0,
        padding_rgb=[0, 0, 0],
        mask=erp_mask, lat_grid=latitude, long_grid=longitude,
    )
    print(f"        ERP patch size (after resize): {erp_img_u8.shape[1]}x{erp_img_u8.shape[0]}")

    # ------------------------------------------------------------------
    # Step 2: model inference on the ERP patch
    # ------------------------------------------------------------------
    print(f"  [3/4] Running model inference ...")
    norm_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    img_tensor  = TF.normalize(TF.to_tensor(erp_img_u8), **norm_stats)
    gt_tensor   = TF.to_tensor(erp_depth_r)
    mask_tensor = TF.to_tensor((erp_depth_r > 0.01).astype(np.uint8))
    attn_tensor = TF.to_tensor((attn_mask > 0).astype(np.float32))

    batch = {
        'image':      img_tensor.unsqueeze(0),
        'gt':         gt_tensor.unsqueeze(0),
        'mask':       mask_tensor.unsqueeze(0),
        'attn_mask':  attn_tensor.unsqueeze(0),
        'lat_range':  lat_range.unsqueeze(0),
        'long_range': long_range.unsqueeze(0),
        'lat_grid':   torch.tensor(lat_grid).unsqueeze(0),
        'long_grid':  torch.tensor(long_grid).unsqueeze(0),
    }

    with torch.no_grad():
        preds, _, _ = model(
            batch['image'].to(device),
            batch['lat_range'].to(device),
            batch['long_range'].to(device),
            attn_mask=batch['attn_mask'].to(device),
            lat_grid=batch['lat_grid'].to(device),
        )
    # Apply the scale factor so depth values are in metric metres
    preds = preds * pred_scale_factor
    print(f"        Depth range in ERP space: "
          f"[{preds.min().item():.3f}, {preds.max().item():.3f}] m")

    # ------------------------------------------------------------------
    # Step 3: back-project ERP predictions → undistorted camera space
    # ------------------------------------------------------------------
    print(f"  [4/4] Back-projecting depth to camera space ...")

    # The virtual ERP sphere height after the resize-to-fwd_sz step
    erp_h = float(cano_sz[0]) * float(pred_scale_factor)
    erp_w = erp_h * 2.0

    # Derive a pinhole K_new that matches the undistorted output image plane.
    # For OPENCV_FISHEYE this removes the radial distortion; for PINHOLE it is
    # identical to the original K.
    K_new = compute_undistorted_K(cam_params, org_h, org_w)
    cam_params_pinhole = {
        'dataset': 'custom_undistorted',
        'fx': float(K_new[0, 0]),
        'fy': float(K_new[1, 1]),
        'cx': float(K_new[0, 2]),
        'cy': float(K_new[1, 2]),
    }

    # erp_patch_to_cam_fast expects:
    #   img_erp        : [3, H, W]  normalised tensor
    #   depth_erp      : [1, H, W]  depth tensor
    #   mask_valid_erp : [1, H, W]  valid-region mask (attn_mask here)
    img_out, depth_out, _mask_valid_out, mask_active_out = erp_patch_to_cam_fast(
        batch['image'][0].to(device),
        preds[0].to(device),
        batch['attn_mask'][0].to(device),
        theta=0.0, phi=0.0,
        out_h=org_h, out_w=org_w,
        erp_h=erp_h, erp_w=erp_w,
        cam_params=cam_params_pinhole,
    )

    # Decode outputs to numpy arrays
    undist_rgb  = decode_erp_image(img_out)              # [H, W, 3] uint8
    depth_cam   = depth_out[0, 0].cpu().numpy()          # [H, W] float32  (metric m)
    # mask_active_out shape: [1, H, W] — True where the pinhole view is valid
    mask_active = (mask_active_out[0].cpu().numpy() > 0).astype(np.uint8)  # [H, W]

    valid_depths = depth_cam[mask_active > 0]
    if valid_depths.size > 0:
        print(f"        Depth range in camera space: "
              f"[{valid_depths.min():.3f}, {valid_depths.max():.3f}] m")

    # ------------------------------------------------------------------
    # Step 4: save the three key outputs
    # ------------------------------------------------------------------

    # (a) Undistorted camera-view reference image (NOT the ERP image)
    path_undist = os.path.join(out_dir, f'{stem}_undistorted.jpg')
    Image.fromarray(undist_rgb).save(path_undist)
    print(f"        [output a] Undistorted reference image   -> {path_undist}")

    # (b) Predicted depth converted back to camera/image format, with colorbar
    path_depth = os.path.join(out_dir, f'{stem}_depth_cam.jpg')
    save_depth_with_colorbar(depth_cam, mask_active, path_depth, depth_max)
    print(f"        [output b] Camera-space depth with colorbar -> {path_depth}")

    # (c) Depth overlay on undistorted image (alignment inspection)
    path_overlay = os.path.join(out_dir, f'{stem}_overlay.jpg')
    save_depth_overlay(undist_rgb, depth_cam, mask_active, path_overlay, depth_max)
    print(f"        [output c] Depth overlay on undistorted image -> {path_overlay}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(model_file: str, config_file: str, device: torch.device):
    """Load UniDAC model from checkpoint and config."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    model = UniDAC.build(config)
    model.load_pretrained(model_file)
    model = model.to(device)
    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def collect_images(args: argparse.Namespace) -> list:
    """Return a sorted list of image paths based on CLI arguments."""
    if args.image_file:
        return [args.image_file]
    paths = []
    for fname in sorted(os.listdir(args.input_dir)):
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
            paths.append(os.path.join(args.input_dir, fname))
    return paths


def main(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Camera intrinsics ---
    print(f"\nLoading intrinsics from: {args.intrinsics_json}")
    intr = load_intrinsics(args.intrinsics_json)
    cam_params = build_cam_params(intr)
    print(f"  camera_model = {intr['camera_model']}")
    print(f"  fl_x={intr['fl_x']:.2f}  fl_y={intr['fl_y']:.2f}  "
          f"cx={intr['cx']:.2f}  cy={intr['cy']:.2f}")

    # CLI args can be overridden by values in the intrinsics JSON
    crop_wFoV = float(intr.get('crop_wFoV', args.crop_wfov))
    fwd_sz    = list(intr.get('fwd_sz',    args.fwd_sz))

    # --- Model ---
    print(f"\nLoading model from: {args.model_file}")
    model, config = build_model(args.model_file, args.config, device)
    cano_sz = config['data']['cano_sz']
    print(f"  cano_sz={cano_sz}  fwd_sz={fwd_sz}  crop_wFoV={crop_wFoV}°  depth_max={args.depth_max} m")

    # --- Image list ---
    image_paths = collect_images(args)
    if not image_paths:
        print('\nNo images found. '
              'Use --image-file <path> or --input-dir <directory>.')
        return

    print(f"\nProcessing {len(image_paths)} image(s)  ->  output dir: {args.out_dir}\n")

    for i, img_path in enumerate(image_paths):
        print(f"[{i + 1}/{len(image_paths)}] {img_path}")
        try:
            process_image(
                model, device, img_path, cam_params,
                cano_sz, fwd_sz, crop_wFoV, args.depth_max, args.out_dir,
            )
        except Exception as exc:
            print(f"  ERROR processing {img_path}: {exc}")
        print()

    print('All images processed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='UniDAC custom inference demo (user-supplied intrinsics)',
        conflict_handler='resolve',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input (mutually exclusive) ---
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument(
        '--image-file', type=str,
        help='Path to a single input image.',
    )
    input_grp.add_argument(
        '--input-dir', type=str,
        help='Path to a directory; all images with common extensions are processed.',
    )

    # --- Required ---
    parser.add_argument(
        '--intrinsics-json', type=str, required=True,
        help='Path to a JSON file containing camera intrinsics '
             '(shared for all images in the batch).',
    )

    # --- Model / output ---
    parser.add_argument('--model-file', type=str, default='checkpoints/unidac.pt',
                        help='Path to the UniDAC model checkpoint (.pt).')
    parser.add_argument('--config', type=str, default=_DEFAULT_CONFIG,
                        help='Path to the model config JSON.')
    parser.add_argument('--out-dir', type=str, default='demo/output_custom',
                        help='Directory to write output artifacts.')

    # --- Inference parameters ---
    parser.add_argument('--depth-max', type=float, default=10.0,
                        help='Maximum depth (m) for colour-scale of visualisations.')
    parser.add_argument('--crop-wfov', type=float, default=180.0,
                        help='Horizontal FOV (degrees) for ERP crop. '
                             'Overridden by "crop_wFoV" in the intrinsics JSON if present.')
    parser.add_argument('--fwd-sz', type=int, nargs=2, default=[512, 704],
                        metavar=('H', 'W'),
                        help='Model input patch size [H W]. '
                             'Overridden by "fwd_sz" in the intrinsics JSON if present.')

    args = parser.parse_args()
    main(args)