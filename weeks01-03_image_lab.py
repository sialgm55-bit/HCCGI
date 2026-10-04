"""
Weeks 1-3 Lab: Image Pixels, Filters, and Edges
Author: G.M (HCCGI Image Lab)
Course Repository Script for Automated Grading and Image Analysis
"""

import os
from typing import Dict, List, Any
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for headless execution
import matplotlib.pyplot as plt


def inspect_image(image_path: str) -> Dict[str, Any]:
    """
    Load the image and return its measured image-data properties.

    Parameters:
        image_path (str): Path to the original input image file.

    Returns:
        dict: Image properties including width, height, channels, shape,
              pixel_count, estimated_bytes, and color_order.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Original image not found at path: {image_path}")

    # Load image using OpenCV (loads as BGR format by default)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from path: {image_path}")

    height, width, channels = img.shape
    shape_list = [height, width, channels]
    pixel_count = width * height
    estimated_bytes = height * width * channels * 1  # 8 bits = 1 byte per channel
    color_order = "BGR"

    report = {
        "width": width,
        "height": height,
        "channels": channels,
        "shape": shape_list,
        "pixel_count": pixel_count,
        "estimated_bytes": estimated_bytes,
        "color_order": color_order
    }

    print("--- Task 1: Image Inspection Report ---")
    for key, val in report.items():
        print(f"  {key}: {val}")
    print("--------------------------------------")

    return report


def create_pixel_views(image_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Create the labeled channel, grayscale, and downsampled views.

    Parameters:
        image_path (str): Path to original image.
        output_dir (str): Directory where output montage will be saved.

    Returns:
        dict: Dictionary containing original_size, downsampled_size, and output_path.
    """
    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Unable to read image at {image_path}")

    height, width, channels = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Separate color channels for visual isolation
    b_channel, g_channel, r_channel = cv2.split(img_bgr)

    # 3-channel color isolated representations
    red_view = np.zeros_like(img_rgb)
    red_view[:, :, 0] = r_channel

    green_view = np.zeros_like(img_rgb)
    green_view[:, :, 1] = g_channel

    blue_view = np.zeros_like(img_rgb)
    blue_view[:, :, 2] = b_channel

    # Grayscale conversion
    gray_view = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Downsampling (half width and half height)
    downsampled_w = width // 2
    downsampled_h = height // 2
    downsampled_img = cv2.resize(img_rgb, (downsampled_w, downsampled_h), interpolation=cv2.INTER_AREA)

    # Create montage layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Task 2: Color Channels, Grayscale, and Downsampled Views", fontsize=16, fontweight='bold')

    axes[0, 0].imshow(red_view)
    axes[0, 0].set_title("Red Channel View")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(green_view)
    axes[0, 1].set_title("Green Channel View")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(blue_view)
    axes[0, 2].set_title("Blue Channel View")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(gray_view, cmap='gray')
    axes[1, 0].set_title("Grayscale View")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(downsampled_img)
    axes[1, 1].set_title(f"Downsampled View ({downsampled_w}x{downsampled_h})")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(img_rgb)
    axes[1, 2].set_title(f"Original RGB View ({width}x{height})")
    axes[1, 2].axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "pixel_views.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        "original_size": [width, height],
        "downsampled_size": [downsampled_w, downsampled_h],
        "output_path": output_path
    }


def create_adjustments(
    image_path: str,
    output_dir: str,
    brightness_delta: int = 40,
    contrast_factor: float = 1.5,
    threshold: int = 127,
) -> Dict[str, Any]:
    """
    Create labeled brightness, contrast, and threshold results.

    Parameters:
        image_path (str): Path to original image.
        output_dir (str): Target output directory.
        brightness_delta (int): Delta to add for brightness adjustment.
        contrast_factor (float): Multiplier for contrast adjustment.
        threshold (int): Intensity cutoff for binary thresholding (0-255).

    Returns:
        dict: Parameters used and the output_path of the resulting montage.
    """
    if not (0 <= threshold <= 255):
        raise ValueError(f"Threshold must be in the range 0-255, got {threshold}")

    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Unable to read image at {image_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Brightness Adjustment (Additive with clipping)
    brighter = np.clip(gray.astype(np.int16) + brightness_delta, 0, 255).astype(np.uint8)

    # 2. Contrast Adjustment (Multiplicative with clipping)
    higher_contrast = np.clip(gray.astype(np.float32) * contrast_factor, 0, 255).astype(np.uint8)

    # 3. Binary Thresholding (Greater than threshold becomes white 255, else black 0)
    _, thresh_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Create montage layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Task 3: Brightness, Contrast, and Thresholding Adjustments", fontsize=16, fontweight='bold')

    axes[0, 0].imshow(gray, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("Original Grayscale Base")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(brighter, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(f"Brighter (Delta = +{brightness_delta})")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(higher_contrast, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(f"Higher Contrast (Factor = {contrast_factor}x)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(thresh_img, cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title(f"Binary Thresholded (Cutoff = {threshold})")
    axes[1, 1].axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "adjustments.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        "brightness_delta": brightness_delta,
        "contrast_factor": contrast_factor,
        "threshold": threshold,
        "output_path": output_path
    }


def create_blur_and_edges(
    image_path: str,
    output_dir: str,
    kernel_size: int = 5,
) -> Dict[str, Any]:
    """
    Create labeled grayscale, mean-blur, and Sobel-edge results.

    Parameters:
        image_path (str): Path to input image.
        output_dir (str): Directory where montage image will be saved.
        kernel_size (int): Size of averaging blur kernel (must be odd positive int).

    Returns:
        dict: Dict containing kernel_size and output_path.
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Unable to read image at {image_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Mean Blur (Averaging box filter)
    blurred = cv2.blur(gray, (kernel_size, kernel_size))

    # Sobel Edge Detection on Original Grayscale
    sobelx_orig = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely_orig = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_orig_mag = np.hypot(sobelx_orig, sobely_orig)
    sobel_orig_norm = cv2.normalize(sobel_orig_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Sobel Edge Detection on Blurred Grayscale
    sobelx_blur = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely_blur = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_blur_mag = np.hypot(sobelx_blur, sobely_blur)
    sobel_blur_norm = cv2.normalize(sobel_blur_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Create montage layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Task 4: Mean Blur (Kernel={kernel_size}x{kernel_size}) and Sobel Edge Detection", fontsize=16, fontweight='bold')

    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title("1. Original Grayscale")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title(f"2. Mean Blurred (Kernel = {kernel_size}x{kernel_size})")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(sobel_orig_norm, cmap='gray')
    axes[1, 0].set_title("3. Sobel Edges (Original Image)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(sobel_blur_norm, cmap='gray')
    axes[1, 1].set_title("4. Sobel Edges (Blurred Image)")
    axes[1, 1].axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "blur_and_edges.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        "kernel_size": kernel_size,
        "output_path": output_path
    }


def run_lab(image_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Run Tasks 1-4 and return their results together.

    Parameters:
        image_path (str): Path to original image.
        output_dir (str): Path to output directory.

    Returns:
        dict: Nested dictionary containing task1, task2, task3, and task4 outputs.
    """
    task1_result = inspect_image(image_path)
    task2_result = create_pixel_views(image_path, output_dir)
    task3_result = create_adjustments(image_path, output_dir)
    task4_result = create_blur_and_edges(image_path, output_dir)

    return {
        "task1": task1_result,
        "task2": task2_result,
        "task3": task3_result,
        "task4": task4_result
    }


def main() -> None:
    """Run the lab using the required repository paths."""
    image_path = os.path.join("images", "original.jpg")
    output_dir = "outputs"
    
    os.makedirs(output_dir, exist_ok=True)
    results = run_lab(image_path, output_dir)
    print("\nLab execution completed successfully!")
    print(f"All output montages created under '{output_dir}/':")
    print(f"  - {results['task2']['output_path']}")
    print(f"  - {results['task3']['output_path']}")
    print(f"  - {results['task4']['output_path']}")


if __name__ == "__main__":
    main()
