#!/usr/bin/env python3
"""
pyPetana

Tk/OpenCV tool for browsing frames (video or image folders), interactively tuning
preprocessing/selection parameters, exporting frames, and extracting contour /
fractal-dimension metrics across interpolated configurations.

Notes:
- Uses multiprocessing "spawn" start method for safety with OpenCV/Tk on macOS/Windows.
- Worker initializer limits OpenCV + BLAS/OpenMP thread usage to reduce oversubscription.
"""

# Importing this first to ensure new processes are spawns
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

# Loading this second to ensure OpenCV only uses 1 thread/process
import cv2
import concurrent.futures
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import copy
from PIL import Image, ImageTk
import numpy as np
import time
import os
import glob
import json

# Cache for last displayed frame (avoids repeated decoding if slider does not move)
last_frame = None
last_frame_index = -1

# module-level globals
_WORKER = False
_TPL = None


def _worker_init() -> None:
    """
    ProcessPool worker initializer.

    This is called once per worker process. It marks the process as a worker and
    sets thread limits so each worker uses a single thread for:
      - OpenCV internal threading (cv2.setNumThreads(1))
      - NumPy/BLAS/OpenMP pools (via threadpoolctl)

    Returns:
        None
    """
    global _WORKER, _TPL
    _WORKER = True

    # OpenCV threads
    import cv2

    cv2.setNumThreads(1)

    # NumPy/BLAS/OpenMP threads (threadpoolctl)
    from threadpoolctl import threadpool_limits

    _TPL = threadpool_limits(limits=1)
    _TPL.__enter__()


def _available_logical_cpus() -> int:
    """
    Return the number of logical CPUs available to this process.

    Prefer OS CPU affinity (Linux) if available; fall back to os.cpu_count().

    Returns:
        int: Number of logical CPUs (>= 1).
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError):
        return os.cpu_count() or 8


def _media_base_dir(media_path: str) -> str:
    """
    Return the directory to treat as the "project base" for paths relative to media.

    If media_path is a directory, that directory is the base.
    If media_path is a file, the base is its parent directory.
    If parent directory is empty (e.g., "video.mp4"), return ".".

    Args:
        media_path (str): Path to a media file or image directory.

    Returns:
        str: Base directory path.
    """
    media_path = os.path.normpath(media_path)

    # If it exists and is a directory, use it directly (image-folder case)
    if os.path.isdir(media_path):
        base = media_path
    else:
        # Otherwise treat it as a file path (video case)
        base = os.path.dirname(media_path)

    # dirname("video.mp4") -> "" ; make it "."
    return base or "."


def json_path_relative_to_media(media_path: str, json_path: str) -> str:
    """
    Convert an absolute JSON path into a path relative to the media base directory.

    On Windows, if json_path and media_path are on different drives, os.path.relpath()
    raises ValueError; in that case return a normalized absolute path.

    Args:
        media_path (str): Path to a media file or image folder.
        json_path (str): Path to a JSON file (typically absolute).

    Returns:
        str: Relative JSON path (or absolute path if relpath cannot be computed).
    """
    base = _media_base_dir(media_path)

    # If paths are on different drives on Windows, relpath can raise ValueError
    try:
        return os.path.relpath(os.path.normpath(json_path), start=base)
    except ValueError:
        # Fall back to a normalized absolute path
        return os.path.abspath(json_path)


def resolve_json_path_from_media(media_path: str, json_path_rel: str) -> str:
    """
    Resolve a JSON path (stored relative to media) into an absolute/normalized path.

    If json_path_rel is already absolute, os.path.join(base, json_path_rel) will
    ignore base as usual.

    Args:
        media_path (str): Path to a media file or image folder.
        json_path_rel (str): Relative (or absolute) JSON path to resolve.

    Returns:
        str: Normalized resolved path.
    """
    base = _media_base_dir(media_path)
    return os.path.normpath(os.path.join(base, json_path_rel))


def open_media(path):
    """
    Open either:
      - None: synthetic empty image case (black mask),
      - a directory: a folder of images (sorted),
      - a file: a video file.

    Returns a tuple:
      total_frames (int): number of frames/images
      read_frame (Callable[[int], Tuple[bool, np.ndarray|None]]): read_frame(idx) -> (ret, frame)
      cleanup (Callable[[], None]): resource cleanup function (no-op for directories)

    Args:
        path (str|None): Media path. None => synthetic test frames.

    Returns:
        tuple[int|None, callable|None, callable|None]:
            (total_frames, read_frame, cleanup) or (None, None, None) on failure.
    """
    try:
        if path is None:
            # Empty image case: return a fixed black image
            mask = np.zeros((1080, 1920, 3), dtype=np.uint8)
            total_frames = 100

            def read_frame(idx):
                return True, mask

            def cleanup():
                pass

        elif os.path.isdir(path):
            # Folder of images.
            image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(path, ext)))
            image_files = sorted(image_files)
            total_frames = len(image_files)

            def read_frame(idx):
                if idx >= total_frames or idx < 0:
                    return False, None
                frame = cv2.imread(image_files[idx])
                ret = frame is not None
                return ret, frame

            cleanup = lambda: None  # nothing to release for a folder of images.

        elif os.path.isfile(path):
            # Assume it's a video file.
            cap = cv2.VideoCapture(path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            def read_frame(idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                return ret, frame

            cleanup = cap.release  # release the VideoCapture object when done.
        else:
            # Path is neither file nor directory
            pass
    except Exception:
        # Failed to load, so signal it
        return None, None, None

    return total_frames, read_frame, cleanup


def is_contour_inside(contour1, contour2) -> bool:
    """
    Check whether all points of contour1 lie inside contour2.

    Uses cv2.pointPolygonTest for each point.

    Args:
        contour1 (np.ndarray): Candidate inner contour, shape (N, 1, 2).
        contour2 (np.ndarray): Candidate outer contour, shape (M, 1, 2).

    Returns:
        bool: True if every point in contour1 is inside contour2 (strictly inside or on edge),
              False otherwise.
    """
    for i in range(len(contour1)):
        point = (int(contour1[i][0][0]), int(contour1[i][0][1]))
        if cv2.pointPolygonTest(contour2, point, False) < 0:
            return False
    return True


def generate_points(last_point, point, N):
    """
    Generate N evenly spaced integer points along the segment from last_point to point.

    This is used to "densify" a polygon edge so you can test intermediate points against
    another contour, reducing missed intersections for long edges.

    Args:
        last_point (tuple[int,int]): Starting point (x1, y1).
        point (tuple[int,int]): Ending point (x2, y2).
        N (int): Number of intermediate points to generate.

    Returns:
        list[tuple[int,int]]: Generated points (rounded to nearest integer pixels).
    """
    x1, y1 = last_point
    x2, y2 = point

    points = []
    for i in range(1, N + 1):
        t = i / (N + 1)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        points.append((int(round(x)), int(round(y))))

    return points


def is_contour_partially_inside(contour1, contour2, side_width) -> bool:
    """
    Check whether ANY point (including interpolated edge points) of contour1 is inside contour2.

    This differs from is_contour_inside(): it returns True as soon as it finds a point
    of contour1 (or a generated intermediate point along an edge) that is inside contour2.

    Args:
        contour1 (np.ndarray): Contour to test, shape (N, 1, 2).
        contour2 (np.ndarray): Contour to test against, shape (M, 1, 2).
        side_width (float|int): Approximate edge length scale controlling how many
            interpolated points to test (int(side_width/2)+1).

    Returns:
        bool: True if any point (original or generated) is inside contour2.
    """
    last_point = (int(contour1[-1][0][0]), int(contour1[-1][0][1]))
    for i in range(len(contour1)):
        point = (int(contour1[i][0][0]), int(contour1[i][0][1]))
        if cv2.pointPolygonTest(contour2, point, False) > 0:
            return True
        points = generate_points(last_point, point, int(side_width / 2) + 1)
        for new_point in points:
            if cv2.pointPolygonTest(contour2, new_point, False) > 0:
                return True
        last_point = point
    return False


def apply_curve_transformation(
    image, transformation_function, lower_threshold, upper_threshold
):
    """
    Apply a luminance/curve transformation function to an image using float scaling.

    The image is converted to float [0,1], transformed, then clipped and converted
    back to uint8 [0,255].

    Args:
        image (np.ndarray): Input image (uint8).
        transformation_function (callable): Function signature f(float_image, lo, hi) -> float_image.
        lower_threshold (float): Lower threshold in [0,1] (or compatible with transformation_function).
        upper_threshold (float): Upper threshold in [0,1] (or compatible with transformation_function).

    Returns:
        np.ndarray: Transformed uint8 image.
    """
    float_image = image.astype(np.float32) / 255.0
    transformed_image = transformation_function(
        float_image, lower_threshold, upper_threshold
    )
    transformed_image = np.clip(transformed_image, 0, 1) * 255
    transformed_image = transformed_image.astype(np.uint8)
    return transformed_image


def draw_split_line(bgr, cx, cy, angle_deg, length, color=(0, 255, 0), thickness=2):
    """
    Draw a line through (cx, cy) on a BGR image.

    Convention:
      angle_deg = 0   -> vertical line (splits left/right)
      angle_deg = 90  -> horizontal line (splits top/bottom)

    Args:
        bgr (np.ndarray): Input BGR image (uint8).
        cx (int|float): X coordinate of split center.
        cy (int|float): Y coordinate of split center.
        angle_deg (float): Line angle in degrees (0 vertical, 90 horizontal).
        length (float): Half-length scale used to compute endpoints.
        color (tuple[int,int,int]): BGR color for the line.
        thickness (int): Line thickness in pixels.

    Returns:
        np.ndarray: Copy of input image with line (and center marker) drawn.
    """
    out = bgr.copy()

    # Direction vector along the LINE.
    theta = np.deg2rad(angle_deg)
    dx = np.sin(theta)
    dy = np.cos(theta)

    x1 = int(round(cx - dx * length))
    y1 = int(round(cy - dy * length))
    x2 = int(round(cx + dx * length))
    y2 = int(round(cy + dy * length))

    cv2.line(out, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    cv2.circle(out, (int(cx), int(cy)), 4, color, -1, lineType=cv2.LINE_AA)
    return out


def tilt_luminance_from_line(
    bgr, cx, cy, angle_deg, strength, max_contrast=0.25, max_brightness=20.0
):
    """
    Apply a luminance "tilt" across a split line through (cx, cy).

    The image is converted to LAB, and L is modified by a combination of:
      - a contrast gain varying linearly with signed distance to the line
      - a brightness bias varying linearly with signed distance to the line

    Args:
        bgr (np.ndarray): Input BGR image (uint8).
        cx (int|float): Split center x (pixels).
        cy (int|float): Split center y (pixels).
        angle_deg (float): 0..180 degrees; 0 = vertical line per convention.
        strength (float|int): -100..100; positive brightens one side, darkens the other.
        max_contrast (float): At strength=100, gain changes by up to ±max_contrast.
        max_brightness (float): At strength=100, bias changes by up to ±max_brightness (L units).

    Returns:
        np.ndarray: Output BGR image (uint8) with adjusted luminance.
    """
    h, w = bgr.shape[:2]

    # Convert to LAB and work on luminance only
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    Lf = L.astype(np.float32)

    # Build signed-distance field to the line through (cx, cy)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - float(cx)
    dy = yy - float(cy)

    theta = np.deg2rad(angle_deg)
    ux = np.sin(theta)
    uy = np.cos(theta)

    # Normal vector (perpendicular to the line)
    nx = uy
    ny = -ux

    d = dx * nx + dy * ny  # signed distance along normal

    # Normalize so corners map roughly into [-1, 1]
    corners = np.array(
        [
            [0 - cx, 0 - cy],
            [w - 1 - cx, 0 - cy],
            [0 - cx, h - 1 - cy],
            [w - 1 - cx, h - 1 - cy],
        ],
        dtype=np.float32,
    )

    dmax = np.max(np.abs(corners[:, 0] * nx + corners[:, 1] * ny))
    if dmax < 1e-6:
        return bgr

    d_norm = np.clip(d / dmax, -1.0, 1.0)

    # strength in [-1, 1]
    s = float(strength) / 100.0

    # Contrast varies across the split (about mid-gray = 128 in L channel)
    gain = 1.0 + (s * max_contrast) * d_norm
    gain = np.clip(gain, 0.1, 10.0)

    # Brightness offset varies across the split
    bias = (s * max_brightness) * d_norm

    # Apply contrast around 128 + brightness bias
    Lf2 = (Lf - 128.0) * gain + 128.0 + bias
    L2 = np.clip(Lf2, 0, 255).astype(np.uint8)

    out = cv2.merge([L2, A, B])
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def piecewise_linear_transformation(
    pixel_values_in, lower_threshold, upper_threshold, invert=False
):
    """
    Apply a simple linear remap of pixel values between thresholds.

    Values are clipped to [lower_threshold, upper_threshold], then mapped to [0, 1].

    Args:
        pixel_values_in (np.ndarray): Input float image in [0,1] (or compatible range).
        lower_threshold (float): Lower clip threshold.
        upper_threshold (float): Upper clip threshold.
        invert (bool): Reserved for future use (currently ignored).

    Returns:
        np.ndarray: Float image mapped to [0,1].
    """
    clipped_values = np.clip(pixel_values_in, lower_threshold, upper_threshold)
    scale = 1.0 / (upper_threshold - lower_threshold)
    transformed = (clipped_values - lower_threshold) * scale
    return transformed


def crop_to_content(img):
    """
    Crop an image to the bounding box of non-background content.

    Background is estimated as the mode/most common value among the 4 corners.
    For grayscale: background is an intensity value.
    For color: background is the most common corner BGR triplet.

    Args:
        img (np.ndarray): Input grayscale (H,W) or BGR (H,W,3) image.

    Returns:
        np.ndarray: Cropped image. If no content found, returns empty slice img[0:0, 0:0].
    """
    h, w = img.shape[:2]

    # background = mode of 4 corners
    corners = np.array([img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]])
    if img.ndim == 2:
        bg = int(np.bincount(corners.astype(np.int64)).argmax())
        fg = (img != bg).astype(np.uint8) * 255
    else:
        bg = corners[
            np.argmax(np.unique(corners.reshape(4, -1), axis=0, return_counts=True)[1])
        ]
        fg = (np.any(img != bg, axis=2)).astype(np.uint8) * 255

    pts = cv2.findNonZero(fg)
    if pts is None:
        return img[0:0, 0:0]  # no content

    x, y, ww, hh = cv2.boundingRect(pts)
    return img[y : y + hh, x : x + ww]


def draw_eroded_contour_band(
    image, contours, color, contour_thickness=2, kernel=None, pad=3
):
    """
    Draw a contour "band" of thickness contour_thickness inside filled contours.

    Implementation:
      - Fill contour into a binary ROI mask.
      - Erode the mask by contour_thickness iterations.
      - Band = mask AND (NOT eroded_mask) -> an inner boundary band.
      - Colorize band pixels in the input image ROI.

    Args:
        image (np.ndarray): Image to draw into (modified in-place in ROIs).
        contours (list[np.ndarray]): Contours to draw.
        color (tuple[int,int,int]): BGR color for band pixels.
        contour_thickness (int): Erosion iterations controlling band thickness.
        kernel (np.ndarray|None): Structuring element for erosion (default 3x3 ellipse).
        pad (int): Extra pixels around bounding box to avoid erosion edge clipping.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            (band, image)
            band is the band mask from the last contour processed.
    """
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    h, w = image.shape[:2]
    iters = int(contour_thickness)

    # Precompute extra padding needed so erosion doesn't clip at ROI edge.
    margin = pad + iters

    band = None
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)

        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(w, x + cw + margin)
        y1 = min(h, y + ch + margin)

        roi_h = y1 - y0
        roi_w = x1 - x0

        # Shift contour into ROI coords
        c = contour.copy()
        c[:, 0, 0] -= x0
        c[:, 0, 1] -= y0

        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 1, thickness=cv2.FILLED)

        er = cv2.erode(mask, kernel, iterations=iters)
        band = mask & (1 - er)

        m = band.astype(bool)
        image[y0:y1, x0:x1][m] = color

    return band, image


def crop_mask_to_content(mask, *masks):
    """
    Crop a 2D mask (and any additional masks) to the bounding box of nonzero pixels.

    Args:
        mask (np.ndarray): Primary 2D mask (0/1 or 0/255).
        *masks (np.ndarray): Additional masks to crop with the same bounding box.

    Returns:
        tuple[np.ndarray, ...]:
            Cropped (mask, *masks). If primary mask is empty, returns empty slice of primary mask.
    """
    m = (mask > 0).astype(np.uint8) * 255
    pts = cv2.findNonZero(m)
    if pts is None:
        return mask[0:0, 0:0]  # empty

    x, y, w, h = cv2.boundingRect(pts)
    cropped = (mask[y : y + h, x : x + w],)
    cropped += tuple(m2[y : y + h, x : x + w] for m2 in masks)
    return cropped


def generate_fd_out(
    fd_mask,
    fd_block_sums,
    fd_box_size,
    fd_Wp,
    fd_Hp,
    fd_Wpad,
    fd_Hpad,
    block_color=np.array([230, 150, 100], dtype=np.float32),
    alpha=0.5,
):
    """
    Create a visualization image for fractal-dimension box occupancy.

    Produces a white background BGR image where:
      - mask pixels are red
      - occupied boxes are alpha-blended with block_color
      - box outlines are drawn in black for occupied boxes
      - output is padded so outlines are consistent.

    Args:
        fd_mask (np.ndarray): 2D binary mask (0/1).
        fd_block_sums (np.ndarray): Block sums (nonzero indicates occupied block).
        fd_box_size (int): Block size in pixels.
        fd_Wp (int): Padded mask width.
        fd_Hp (int): Padded mask height.
        fd_Wpad (int): Global pad width to reach max box multiple.
        fd_Hpad (int): Global pad height to reach max box multiple.
        block_color (np.ndarray): Color for occupied blocks (BGR float).
        alpha (float): Blend factor for occupied blocks.

    Returns:
        np.ndarray: BGR uint8 visualization image.
    """
    fd_occ = fd_block_sums > 0
    nby, nbx = fd_occ.shape

    fd_mask_u8 = fd_mask.astype(np.uint8)  # 0/1
    fd_mask_base = np.full((*fd_mask_u8.shape, 3), 255, dtype=np.uint8)  # white
    fd_mask_base[fd_mask_u8.astype(bool)] = (0, 0, 255)  # red in BGR

    fd_mask_occ = cv2.resize(
        fd_occ.astype(np.uint8), (fd_Wp, fd_Hp), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    fd_mask_out = fd_mask_base.astype(np.float32)
    fd_mask_out[fd_mask_occ] = (
        fd_mask_out[fd_mask_occ] * (1.0 - alpha) + block_color * alpha
    )
    fd_mask_out_clipped = np.clip(fd_mask_out, 0, 255).astype(np.uint8)

    # pad by one on all sides to make sure box outlines are always 2 thick
    fd_mask_out_clipped = np.pad(
        fd_mask_out_clipped,
        ((1, 1), (1, 1), (0, 0)),
        mode="constant",
        constant_values=255,
    )

    for by in range(nby):
        y0 = by * fd_box_size + 1
        y1 = y0 + fd_box_size - 1
        for bx in range(nbx):
            if not fd_occ[by, bx]:
                continue
            x0 = bx * fd_box_size + 1
            x1 = x0 + fd_box_size - 1
            cv2.rectangle(
                fd_mask_out_clipped,
                (x0, y0),
                (x1, y1),
                (0, 0, 0),
                1,
                lineType=cv2.LINE_8,
            )

            # outer outline (expanded by 1)
            cv2.rectangle(
                fd_mask_out_clipped,
                (x0 - 1, y0 - 1),
                (x1 + 1, y1 + 1),
                (0, 0, 0),
                1,
                lineType=cv2.LINE_8,
            )

    # now we pad to the largest box size multiple plus 1
    fd_pad_top = fd_Hpad // 2
    fd_pad_bottom = fd_Hpad - fd_pad_top
    fd_pad_left = fd_Wpad // 2
    fd_pad_right = fd_Wpad - fd_pad_left
    fd_mask_out_clipped = np.pad(
        fd_mask_out_clipped,
        (
            (fd_pad_top + 1, fd_pad_bottom + 1),
            (fd_pad_left + 1, fd_pad_right + 1),
            (0, 0),
        ),
        mode="constant",
        constant_values=255,
    )
    return fd_mask_out_clipped


def extract_data(
    blur,
    thresh_contours_filtered,
    box_sizes_series: str,
    frame_index: int,
    contour_thickness: int = 1,
    include_holes=False,
    extract_fd_frames="",
    extract_fd_fits="",
    **_extras,
):
    """
    Compute geometric + fractal-dimension metrics for the selected contour(s).

    Assumes thresh_contours_filtered[0] is the "main" contour and any additional
    contours represent interior holes (when include_holes=True).

    Metrics:
      - area, perimeter of main contour
      - optional area_holes, perimeter_holes
      - circularity (4πA / P²), and circularity with holes adjustment
      - box-counting fractal dimension for:
          filled mask
          boundary band mask
          optionally holes-filled mask and holes boundary mask
      - RMS fit quality for each dimension

    Optional outputs:
      - If extract_fd_frames dir exists: write box-count visualization images.
      - If extract_fd_fits dir exists: write fit data tables.

    Args:
        blur (np.ndarray): Preprocessed grayscale-ish image (used for mask sizing).
        thresh_contours_filtered (list[np.ndarray]): Contours with [0] = main contour.
        box_sizes_series (str): Comma-separated base box sizes; also expanded by powers of 2.
        frame_index (int): Frame index for naming outputs.
        contour_thickness (int): Thickness (in erosion iterations) for boundary band masks.
        include_holes (bool): Whether to treat inner contours as holes.
        extract_fd_frames (str): Directory path; if exists, FD visualization PNGs written.
        extract_fd_fits (str): Directory path; if exists, FD fit .dat files written.
        **_extras: Ignored extra kwargs for compatibility with **params patterns.

    Returns:
        dict: Results dictionary with scalar metrics (and optionally hole metrics).
    """
    # calculate the area and perimeter of the center contour, including holes
    area = cv2.contourArea(thresh_contours_filtered[0])
    perimeter = cv2.arcLength(thresh_contours_filtered[0], True)

    area_holes = 0
    perimeter_holes = 0
    for inner_contour in thresh_contours_filtered[1:]:
        area_holes += cv2.contourArea(inner_contour)
        perimeter_holes += cv2.arcLength(inner_contour, True)

    circularity = 4 * np.pi * area / perimeter**2
    circularity_holes = (
        4 * np.pi * (area - area_holes) / (perimeter + perimeter_holes) ** 2
    )

    # Outer-filled mask (no holes punched)
    mask = np.zeros(blur.shape, dtype=np.uint8)
    cv2.drawContours(mask, thresh_contours_filtered[:1], -1, 1, thickness=cv2.FILLED)

    # Inner boundary from erosion (inside the region)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    er = cv2.erode(mask, erode_kernel, iterations=contour_thickness)
    mask_boundary = (mask & (1 - er)).astype(np.uint8)

    if include_holes:
        # Filled mask with holes removed
        mask_holes = mask.copy()
        cv2.drawContours(
            mask_holes, thresh_contours_filtered[1:], -1, 0, thickness=cv2.FILLED
        )

        # Inner boundary including hole boundaries
        er_h = cv2.erode(mask_holes, erode_kernel, iterations=contour_thickness)
        mask_holes_boundary = (mask_holes & (1 - er_h)).astype(np.uint8)
        mask, mask_boundary, mask_holes, mask_holes_boundary = crop_mask_to_content(
            mask, mask_boundary, mask_holes, mask_holes_boundary
        )
    else:
        mask, mask_boundary = crop_mask_to_content(mask, mask_boundary)

    H, W = mask.shape
    max_box = min(H, W) // 5

    # Expand base sizes by powers of 2, keep <= max_box
    powers = 2 ** np.arange(0, 32)
    box_sizes = sorted(
        v
        for b in box_sizes_series.strip().split(",")
        for v in (int(b) * powers)
        if v <= max_box
    )

    Hpad = max([(-H) % box_size for box_size in box_sizes])
    Wpad = max([(-W) % box_size for box_size in box_sizes])

    n = 4 * len(box_sizes)
    logbs = np.empty(n, dtype=np.float64)
    logN = np.empty(n, dtype=np.float64)
    logN_boundary = np.empty(n, dtype=np.float64)
    logN_holes = np.empty(n, dtype=np.float64)
    logN_holes_boundary = np.empty(n, dtype=np.float64)
    np_log = np.log

    for i, box_size in enumerate(box_sizes):
        h_pad = (-H) % box_size
        w_pad = (-W) % box_size

        # Evaluate 4 padding placements to reduce alignment bias
        for j, [pad_top, pad_left] in enumerate(
            [
                [h_pad // 2, w_pad // 2],
                [h_pad, w_pad // 2],
                [h_pad, w_pad],
                [h_pad // 2, w_pad],
            ]
        ):
            pad_bottom = h_pad - pad_top
            pad_right = w_pad - pad_left

            padded_mask = np.pad(
                mask,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )
            padded_mask_boundary = np.pad(
                mask_boundary,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )
            if include_holes:
                padded_mask_holes = np.pad(
                    mask_holes,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="constant",
                    constant_values=0,
                )
                padded_mask_holes_boundary = np.pad(
                    mask_holes_boundary,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="constant",
                    constant_values=0,
                )

            Hp, Wp = padded_mask.shape

            # reshape into blocks
            blocks = padded_mask.reshape(
                Hp // box_size, box_size, Wp // box_size, box_size
            )
            blocks_boundary = padded_mask_boundary.reshape(
                Hp // box_size, box_size, Wp // box_size, box_size
            )

            block_sums = blocks.sum(axis=(1, 3))
            block_sums_boundary = blocks_boundary.sum(axis=(1, 3))

            count_partial = np.count_nonzero(block_sums)
            count_partial_boundary = np.count_nonzero(block_sums_boundary)

            logbs[i * 4 + j] = np_log(box_size)
            logN[i * 4 + j] = np_log(count_partial)
            logN_boundary[i * 4 + j] = np_log(count_partial_boundary)

            if include_holes:
                blocks_holes = padded_mask_holes.reshape(
                    Hp // box_size, box_size, Wp // box_size, box_size
                )
                blocks_holes_boundary = padded_mask_holes_boundary.reshape(
                    Hp // box_size, box_size, Wp // box_size, box_size
                )

                block_sums_holes = blocks_holes.sum(axis=(1, 3))
                block_sums_holes_boundary = blocks_holes_boundary.sum(axis=(1, 3))

                count_partial_holes = np.count_nonzero(block_sums_holes)
                count_partial_holes_boundary = np.count_nonzero(
                    block_sums_holes_boundary
                )

                logN_holes[i * 4 + j] = np_log(count_partial_holes)
                logN_holes_boundary[i * 4 + j] = np_log(count_partial_holes_boundary)

            # Optional frame dumps
            if os.path.exists(extract_fd_frames):
                padded_out = generate_fd_out(
                    padded_mask,
                    block_sums,
                    box_size,
                    Wp,
                    Hp,
                    Wpad - w_pad,
                    Hpad - h_pad,
                )
                padded_out_boundary = generate_fd_out(
                    padded_mask_boundary,
                    block_sums_boundary,
                    box_size,
                    Wp,
                    Hp,
                    Wpad - w_pad,
                    Hpad - h_pad,
                )
                cv2.imwrite(
                    os.path.join(
                        extract_fd_frames, f"fd_f{frame_index:05}_{i:03}_{j:03}.png"
                    ),
                    padded_out,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )
                cv2.imwrite(
                    os.path.join(
                        extract_fd_frames, f"fd_b{frame_index:05}_{i:03}_{j:03}.png"
                    ),
                    padded_out_boundary,
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )
                if include_holes:
                    padded_out_holes = generate_fd_out(
                        padded_mask_holes,
                        block_sums_holes,
                        box_size,
                        Wp,
                        Hp,
                        Wpad - w_pad,
                        Hpad - h_pad,
                    )
                    padded_out_holes_boundary = generate_fd_out(
                        padded_mask_holes_boundary,
                        block_sums_holes_boundary,
                        box_size,
                        Wp,
                        Hp,
                        Wpad - w_pad,
                        Hpad - h_pad,
                    )
                    cv2.imwrite(
                        os.path.join(
                            extract_fd_frames, f"fd_hf{frame_index:05}_{i:03}.png"
                        ),
                        padded_out_holes,
                        [cv2.IMWRITE_PNG_COMPRESSION, 9],
                    )
                    cv2.imwrite(
                        os.path.join(
                            extract_fd_frames, f"fd_hb{frame_index:05}_{i:03}.png"
                        ),
                        padded_out_holes_boundary,
                        [cv2.IMWRITE_PNG_COMPRESSION, 9],
                    )

    # Fits: note slope is negative in logN vs log(box) with this convention
    coeffs, residuals, rank, _, _ = np.polyfit(logbs, logN, 1, full=True)
    slope, intercept = coeffs
    fractal_dimension = -slope

    fractal_dimension_rms = 0
    if rank < 2:
        fractal_dimension_rms = np.nan
    elif residuals.size > 0:
        fractal_dimension_rms = np.sqrt(residuals[0] / len(logbs))

    coeffs_boundary, residuals_boundary, rank_boundary, _, _ = np.polyfit(
        logbs, logN_boundary, 1, full=True
    )
    slope_boundary, intercept_boundary = coeffs_boundary
    fractal_dimension_boundary = -slope_boundary

    fractal_dimension_boundary_rms = 0
    if rank_boundary < 2:
        fractal_dimension_boundary_rms = np.nan
    elif residuals_boundary.size > 0:
        fractal_dimension_boundary_rms = np.sqrt(residuals_boundary[0] / len(logbs))

    if include_holes:
        coeffs_holes, residuals_holes, rank_holes, _, _ = np.polyfit(
            logbs, logN_holes, 1, full=True
        )
        slope_holes, intercept_holes = coeffs_holes
        fractal_dimension_holes = -slope_holes

        fractal_dimension_holes_rms = 0
        if rank_holes < 2:
            fractal_dimension_holes_rms = np.nan
        elif residuals_holes.size > 0:
            fractal_dimension_holes_rms = np.sqrt(residuals_holes[0] / len(logbs))

        coeffs_holes_boundary, residuals_holes_boundary, rank_holes_boundary, _, _ = (
            np.polyfit(logbs, logN_holes_boundary, 1, full=True)
        )
        slope_holes_boundary, intercept_holes_boundary = coeffs_holes_boundary
        fractal_dimension_holes_boundary = -slope_holes_boundary

        fractal_dimension_holes_boundary_rms = 0
        if rank_holes_boundary < 2:
            fractal_dimension_holes_boundary_rms = np.nan
        elif residuals_holes_boundary.size > 0:
            fractal_dimension_holes_boundary_rms = np.sqrt(
                residuals_holes_boundary[0] / len(logbs)
            )

    # Optional fit table output
    if os.path.exists(extract_fd_fits):
        box_sizes_4x = np.repeat(np.asarray(box_sizes), 4)
        fd_fits = {
            "box_sizes": box_sizes_4x,
            "logbs": logbs,
            "logN": logN,
            "logN_boundary": logN_boundary,
        }
        if include_holes:
            fd_fits["logN_holes"] = logN_holes
            fd_fits["logN_holes_boundary"] = logN_holes_boundary

        dtype = [(key, "i4") if key == "box_sizes" else (key, "f8") for key in fd_fits]
        array = np.column_stack([fd_fits[key] for key in fd_fits])
        structured_array = np.array([tuple(row) for row in array], dtype=dtype)

        keys = [key for key, _ in dtype]
        header = f"{keys[0]:>{max(len(keys[0]), 12)}}  " + "  ".join(
            f"{k:>{max(len(k), 13)}}" for k in keys[1:]
        )
        fmt = [
            f"%{max(len(key), 13)}d" if t == "i4" else f"%{max(len(key) + 1, 14)}e"
            for key, t in dtype
        ]
        meta = {
            "contour_thickness": contour_thickness,
            "box_sizes_series": box_sizes_series,
        }
        meta_lines = [f" meta_json={json.dumps(meta, separators=(',', ':'))}"]
        np.savetxt(
            os.path.join(extract_fd_fits, f"fd_fit_d{frame_index:05}.dat"),
            structured_array,
            fmt=fmt,
            header="\n".join([header] + meta_lines),
            comments="#",
        )

    results = {
        "frame_id": frame_index,
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "fractal_dimension": fractal_dimension,
        "fractal_dimension_rms": fractal_dimension_rms,
        "fractal_dimension_boundary": fractal_dimension_boundary,
        "fractal_dimension_boundary_rms": fractal_dimension_boundary_rms,
    }
    if include_holes:
        results["area_holes"] = (area_holes,)
        results["perimeter_holes"] = (perimeter_holes,)
        results["circularity_holes"] = (circularity_holes,)
        results["fractal_dimension_holes"] = (fractal_dimension_holes,)
        results["fractal_dimension_holes_rms"] = (fractal_dimension_holes_rms,)
        results["fractal_dimension_holes_boundary"] = (
            fractal_dimension_holes_boundary,
        )
        results["fractal_dimension_holes_boundary_rms"] = (
            fractal_dimension_holes_boundary_rms,
        )
    return results


def export_frame(
    image_frame,
    image_frame_cropped,
    thresh,
    thresh_contours,
    thresh_contours_filtered,
    cropped_x: int = 0,
    cropped_y: int = 0,
    tilt_strength: float = 0,
    tilt_center_x: float = 50,
    tilt_center_y: float = 50,
    tilt_rotation: float = 0,
    contour_thickness: int = 1,
    include_holes: bool = False,
    export_tilt: bool = False,
    export_tilt_bar: bool = False,
    export_masked: bool = False,
    export_alpha: bool = False,
    export_contours: bool = False,
    export_selects: bool = False,
    **_extras,
):
    """
    Render the 3 preview/export images for a processed frame:
      1) select_image    - threshold image with contour overlays
      2) resulting_image - cropped image masked by the selected contour (holes optional)
      3) output_image    - base export image with optional modifications (tilt, contours, alpha, etc.)

    Args:
        image_frame (np.ndarray): Original full frame (BGR).
        image_frame_cropped (np.ndarray): Cropped frame (BGR) matching contour coordinate space.
        thresh (np.ndarray): Thresholded grayscale image used for contour detection.
        thresh_contours (list[np.ndarray]): All contours found (sorted).
        thresh_contours_filtered (list[np.ndarray]): [0]=main contour, [1:]=holes.
        cropped_x (int): X offset of crop relative to original image.
        cropped_y (int): Y offset of crop relative to original image.
        tilt_strength (float): If non-zero, draw/compute tilt options.
        tilt_center_x (float): Tilt center in percent (0..100) of original width.
        tilt_center_y (float): Tilt center in percent (0..100) of original height.
        tilt_rotation (float): Tilt split line angle (degrees).
        contour_thickness (int): Inner-band thickness (erosion iterations) for contour overlays.
        include_holes (bool): Whether to treat inner contours as holes.
        export_tilt (bool): Apply tilt luminance effect to output_image (if tilt_strength != 0).
        export_tilt_bar (bool): Draw the tilt split line on output_image.
        export_masked (bool): Mask output_image by selected contour/holes.
        export_alpha (bool): Convert output_image to BGRA with background alpha.
        export_contours (bool): Draw contour overlays on output_image.
        export_selects (bool): Export select/threshold visualization as output_image.
        **_extras: Ignored extra kwargs for compatibility with **params patterns.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (select_image, resulting_image, output_image)
    """
    # all contours drawn in blue, center contour in red and interior-to-center contours in green
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if len(thresh_contours) > 0:
        _, select_image = draw_eroded_contour_band(
            cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
            thresh_contours,
            (255, 0, 0),
            contour_thickness=contour_thickness,
            kernel=erode_kernel,
        )
        _, select_image = draw_eroded_contour_band(
            select_image,
            thresh_contours_filtered[:1],
            (0, 0, 255),
            contour_thickness=contour_thickness,
            kernel=erode_kernel,
        )
        if include_holes and len(thresh_contours_filtered[1:]) > 0:
            _, select_image = draw_eroded_contour_band(
                select_image,
                thresh_contours_filtered[1:],
                (0, 255, 0),
                contour_thickness=contour_thickness,
                kernel=erode_kernel,
            )

        # Mask away anything outside the main contour, and optionally punch holes
        mask = np.zeros(image_frame_cropped.shape[:2], dtype=np.uint8)
        cv2.drawContours(
            mask, thresh_contours_filtered[:1], -1, 255, thickness=cv2.FILLED
        )
        if include_holes:
            cv2.drawContours(
                mask, thresh_contours_filtered[1:], -1, 0, thickness=cv2.FILLED
            )
        resulting_image = cv2.bitwise_and(
            image_frame_cropped, image_frame_cropped, mask=mask
        )
    else:
        mask = np.zeros(image_frame_cropped.shape[:2], dtype=np.uint8)
        select_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        resulting_image = image_frame_cropped

    # Draw the tilt line in select_image (visual cue only)
    original_height, original_width = image_frame.shape[:2]
    if tilt_strength != 0:
        select_image = draw_split_line(
            select_image,
            int(0.01 * tilt_center_x * original_width - cropped_x),
            int(0.01 * tilt_center_y * original_height - cropped_y),
            tilt_rotation,
            np.hypot(original_height, original_width),
            color=(127, 0, 127),
            thickness=2,
        )

    # Resolve all the options for the output image
    output_image = copy.deepcopy(image_frame_cropped)

    if export_selects:
        output_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    if export_tilt and tilt_strength != 0:
        output_image = tilt_luminance_from_line(
            output_image,
            int(0.01 * tilt_center_x * original_width),
            int(0.01 * tilt_center_y * original_height),
            tilt_rotation,
            tilt_strength,
            max_contrast=1.00,
            max_brightness=100.0,
        )

    if export_tilt_bar and tilt_strength != 0:
        output_image = draw_split_line(
            output_image,
            int(0.01 * tilt_center_x * original_width - cropped_x),
            int(0.01 * tilt_center_y * original_height - cropped_y),
            tilt_rotation,
            np.hypot(original_height, original_width),
            color=(127, 0, 127),
            thickness=2,
        )

    if export_contours and len(thresh_contours) > 0:
        if export_selects:
            _, output_image = draw_eroded_contour_band(
                select_image,
                thresh_contours,
                (255, 0, 0),
                contour_thickness=contour_thickness,
                kernel=erode_kernel,
            )
        _, output_image = draw_eroded_contour_band(
            output_image,
            thresh_contours_filtered[:1],
            (0, 0, 255),
            contour_thickness=contour_thickness,
            kernel=erode_kernel,
        )
        if include_holes and len(thresh_contours_filtered[1:]) > 0:
            _, output_image = draw_eroded_contour_band(
                output_image,
                thresh_contours_filtered[1:],
                (0, 255, 0),
                contour_thickness=contour_thickness,
                kernel=erode_kernel,
            )

    if export_masked and len(thresh_contours) > 0:
        output_image = cv2.bitwise_and(output_image, output_image, mask=mask)

    if export_alpha:
        output_image = background_alpha_from_corners(output_image)

    return select_image, resulting_image, output_image


def preprocess_frame(
    image_frame,
    frame_index: int,
    crop_square: float,
    crop_circle: float,
    crop_center_x: float,
    crop_center_y: float,
    tilt_strength: float,
    tilt_center_x: float,
    tilt_center_y: float,
    tilt_rotation: float,
    lower_curve: int,
    upper_curve: int,
    blur_kernel: int,
    morph_kernel: int,
    morph_iterations: int,
    lower_threshold: int,
    upper_threshold: int,
):
    """
    Apply preprocessing pipeline to a frame and compute contours.

    Pipeline overview:
      1) Optional tilt luminance (LAB L-channel gradient across split line).
      2) Optional square crop around user-selected center, with size controlled by crop_square.
      3) Optional circular crop (mask + bounding rect trim), with size controlled by crop_circle.
      4) Optional piecewise linear curve transformation between lower_curve and upper_curve.
      5) Convert to grayscale.
      6) Morphological close to fill holes (kernel size + iterations).
      7) Gaussian blur.
      8) Thresholding using lower_threshold/upper_threshold.
      9) Find contours and select the largest/most-central contour; then include inner contours (holes).

    Args:
        image_frame (np.ndarray): Original BGR frame.
        frame_index (int): Frame index (currently unused by preprocessing, but kept for compatibility).
        crop_square (float): 0 disables; >0 crops a square region.
        crop_circle (float): 0 disables; >0 crops circular region.
        crop_center_x (float): 0..100 crop center x percentage.
        crop_center_y (float): 0..100 crop center y percentage.
        tilt_strength (float): -100..100 tilt strength (0 disables).
        tilt_center_x (float): 0..100 tilt center x percentage.
        tilt_center_y (float): 0..100 tilt center y percentage.
        tilt_rotation (float): Tilt line angle in degrees.
        lower_curve (int): Lower curve threshold (0..255).
        upper_curve (int): Upper curve threshold (0..255).
        blur_kernel (int): Gaussian blur kernel radius (kernel size = 2*k + 1).
        morph_kernel (int): Morph kernel radius (kernel size = 2*k + 1).
        morph_iterations (int): Number of close iterations.
        lower_threshold (int): Lower threshold for selection (0 disables).
        upper_threshold (int): Upper threshold for selection (0 disables).

    Returns:
        tuple:
            (image_frame_cropped,
             blur,
             thresh,
             thresh_contours,
             thresh_contours_filtered,
             cropped_x,
             cropped_y)

        Where:
            image_frame_cropped (np.ndarray): Cropped BGR frame.
            blur (np.ndarray): Blurred grayscale frame.
            thresh (np.ndarray): Threshold-masked grayscale frame.
            thresh_contours (list[np.ndarray]): All contours sorted by score.
            thresh_contours_filtered (list[np.ndarray]): [main contour] + [inner contours].
            cropped_x (int): X offset from original frame to cropped frame.
            cropped_y (int): Y offset from original frame to cropped frame.
    """
    height, width, channels = image_frame.shape

    # Optional tilt luminance adjustment (for processing branch)
    if tilt_strength != 0:
        image_frame_tilted = copy.deepcopy(
            tilt_luminance_from_line(
                image_frame,
                int(0.01 * tilt_center_x * width),
                int(0.01 * tilt_center_y * height),
                tilt_rotation,
                tilt_strength,
                max_contrast=1.00,
                max_brightness=100.0,
            )
        )
    else:
        image_frame_tilted = copy.deepcopy(image_frame)

    # Optional square crop
    if crop_square > 0:
        frame_center = (
            int(0.01 * crop_center_x * width),
            int(0.01 * crop_center_y * height),
        )

        # side length scales like circle logic: crop_square=0 => full, 100 => tiny
        side = (100 - crop_square) * min(width, height) / 100.0
        half = side / 2.0

        x0 = int(round(frame_center[0] - half))
        y0 = int(round(frame_center[1] - half))
        x1 = int(round(frame_center[0] + half))
        y1 = int(round(frame_center[1] + half))

        # clamp to image bounds
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(width, x1)
        y1 = min(height, y1)

        # keep square even if clamped
        w = x1 - x0
        h = y1 - y0
        side_int = min(w, h)
        x1 = x0 + side_int
        y1 = y0 + side_int

        image_frame_cropped = image_frame[y0:y1, x0:x1]
        image_frame_tilted_cropped = image_frame_tilted[y0:y1, x0:x1]
        xsc, ysc, wsc, hsc = x0, y0, side_int, side_int
    else:
        image_frame_cropped = copy.deepcopy(image_frame)
        image_frame_tilted_cropped = image_frame_tilted
        xsc, ysc, wsc, hsc = 0, 0, width, height

    height, width, channels = image_frame_cropped.shape

    # Optional circular crop (mask + bounding rect)
    if crop_circle > 0:
        if crop_square:
            frame_center = (int(0.5 * width), int(0.5 * height))
        else:
            frame_center = (
                int(0.01 * crop_center_x * width),
                int(0.01 * crop_center_y * height),
            )
        radius = (100 - crop_circle) * min(width / 2, height / 2) / 100

        crop_mask = np.zeros_like(image_frame_cropped)
        cv2.circle(crop_mask, frame_center, int(radius), (255, 255, 255), -1)

        image_frame_cropped = cv2.bitwise_and(image_frame_cropped, crop_mask)
        crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_BGR2GRAY)

        pts = cv2.findNonZero(crop_mask)
        x, y, w, h = cv2.boundingRect(pts)
        image_frame_cropped = image_frame_cropped[y : y + h, x : x + w]
        image_frame_tilted_cropped = image_frame_tilted_cropped[y : y + h, x : x + w]
        xcc, ycc, wcc, hcc = x, y, w, h
    else:
        xcc, ycc, wcc, hcc = 0, 0, width, height

    height, width, channels = image_frame_cropped.shape
    cropped_x = xcc + xsc
    cropped_y = ycc + ysc

    # Optional curve transformation
    if lower_curve < upper_curve:
        image_frame_tilted_cropped_curved = apply_curve_transformation(
            image_frame_tilted_cropped,
            piecewise_linear_transformation,
            1.0 * (lower_curve / 255),
            1.0 * (upper_curve / 255),
        )
    else:
        image_frame_tilted_cropped_curved = copy.deepcopy(image_frame_tilted_cropped)

    # Convert to grayscale
    gray_frame = cv2.cvtColor(image_frame_tilted_cropped_curved, cv2.COLOR_BGR2GRAY)

    # Close holes using morphology close
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * morph_kernel + 1, 2 * morph_kernel + 1)
    )
    closed = cv2.morphologyEx(
        gray_frame, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations
    )

    # Gaussian blur
    blur = cv2.GaussianBlur(closed, (2 * blur_kernel + 1, 2 * blur_kernel + 1), 0)

    # Center reference for contour scoring
    last_center = (int(width / 2), int(height / 2))

    # Sort contours by area * proximity-to-center weight
    def sort_func(contour):
        """
        Sort key used to prioritize the largest contour near the image center.

        Args:
            contour (np.ndarray): Contour points.

        Returns:
            float: Weighted score.
        """
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centerX = int(M["m10"] / M["m00"])
            centerY = int(M["m01"] / M["m00"])
            weight = 1 / (
                1 + (centerX - last_center[0]) ** 2 + (centerY - last_center[1]) ** 2
            )
        else:
            weight = 0
        return area * weight

    # Apply thresholding
    if lower_threshold > 0 and upper_threshold > 0:
        _, lower_mask = cv2.threshold(blur, lower_threshold - 1, 255, cv2.THRESH_BINARY)
        _, upper_mask = cv2.threshold(
            blur, upper_threshold - 1, 255, cv2.THRESH_BINARY_INV
        )
        thresh_mask = cv2.bitwise_and(lower_mask, upper_mask)
        thresh = cv2.bitwise_and(blur, blur, mask=thresh_mask)
    elif lower_threshold > 0 and upper_threshold == 0:
        _, thresh_mask = cv2.threshold(
            blur, lower_threshold - 1, 255, cv2.THRESH_BINARY
        )
        thresh = cv2.bitwise_and(blur, blur, mask=thresh_mask)
    elif lower_threshold == 0 and upper_threshold > 0:
        _, thresh_mask = cv2.threshold(
            blur, upper_threshold - 1, 255, cv2.THRESH_BINARY_INV
        )
        thresh = cv2.bitwise_and(blur, blur, mask=thresh_mask)
    else:
        # Default: no thresholding applied (kept historical behavior)
        thresh_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)
        thresh = blur
        thresh_contours = []

    thresh_contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    thresh_contours = sorted(thresh_contours, key=sort_func, reverse=True)

    # Filter inner contours (holes) that lie inside the main contour
    thresh_contours_filtered = []
    if len(thresh_contours) > 0:
        thresh_contours_filtered.append(thresh_contours[0])
        for contour in thresh_contours[1:]:
            if is_contour_inside(contour, thresh_contours[0]) is True:
                thresh_contours_filtered.append(contour)

    return (
        image_frame_cropped,
        blur,
        thresh,
        thresh_contours,
        thresh_contours_filtered,
        cropped_x,
        cropped_y,
    )


def background_alpha_from_corners(
    bgr: np.ndarray,
    lo_diff=(0, 0, 0),
    up_diff=(0, 0, 0),
    connectivity: int = 4,
) -> np.ndarray:
    """
    Convert a BGR image to BGRA and set alpha=0 for pixels that match background.

    Background color is chosen as the most common among the 4 corners.
    Any pixel within tolerance (lo_diff/up_diff) of bg_color is treated as background.

    Args:
        bgr (np.ndarray): Input BGR image of shape (H, W, 3), dtype uint8 preferred.
        lo_diff (tuple[int,int,int]): Lower tolerance for inRange per channel (B,G,R).
        up_diff (tuple[int,int,int]): Upper tolerance for inRange per channel (B,G,R).
        connectivity (int): Reserved (not used) - kept for future flood-fill variant.

    Returns:
        np.ndarray: BGRA image of shape (H, W, 4) with alpha set to 0 for background pixels.

    Raises:
        ValueError: If input is not a BGR image of shape (H,W,3).
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image of shape (H, W, 3).")
    if bgr.dtype != np.uint8:
        bgr = bgr.astype(np.uint8, copy=False)

    h, w = bgr.shape[:2]

    corner_coords = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    corner_vals = [tuple(int(x) for x in bgr[y, x]) for (y, x) in corner_coords]

    counts = {}
    for v in corner_vals:
        counts[v] = counts.get(v, 0) + 1

    bg_color, bg_count = max(counts.items(), key=lambda kv: kv[1])
    if bg_count < 2:
        bg_color = corner_vals[0]

    bg = np.array(bg_color, dtype=np.int16)
    lo = np.array(lo_diff, dtype=np.int16)
    up = np.array(up_diff, dtype=np.int16)

    lower = np.clip(bg - lo, 0, 255).astype(np.uint8)
    upper = np.clip(bg + up, 0, 255).astype(np.uint8)

    mask = cv2.inRange(bgr, lower, upper)  # 255 where bg-like

    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = 255
    bgra[mask != 0, 3] = 0
    return bgra


def process_interpolated_frames(
    media_path,
    configs,
    analyze=True,
    include_holes=True,
    export_tilt=True,
    export_tilt_bar=True,
    export_masked=True,
    export_alpha=True,
    export_contours=True,
    export_selects=False,
    frames_path=None,
    extract_fd_fits="",
    extract_fd_frames="",
    box_sizes_series="",
    contour_thickness=1,
):
    """
    Worker helper to process a batch of frame configurations.

    This is intended to be run inside a separate process via ProcessPoolExecutor.

    For each config in configs:
      - Read the specified frame
      - Run preprocess_frame(**params)
      - If analyze=True: run extract_data() and accumulate results
      - Else: run export_frame() and write output_image to frames_path

    Args:
        media_path (str): Media file path or image directory.
        configs (list[dict]): List of frame configs (each must include 'Frame Index' and slider fields).
        analyze (bool): If True, compute and return extracted metrics; if False, export images.
        include_holes (bool): Whether to include holes in masking and analysis.
        export_tilt (bool): Apply tilt luminance in export (requires tilt_strength != 0).
        export_tilt_bar (bool): Draw tilt line in export.
        export_masked (bool): Mask export to selected contour.
        export_alpha (bool): Convert export to BGRA with background alpha.
        export_contours (bool): Draw contour overlays.
        export_selects (bool): Export threshold/select visualization instead of cropped image.
        frames_path (str|None): Output directory for exported frames when analyze=False.
        extract_fd_fits (str): Directory for writing FD fit tables.
        extract_fd_frames (str): Directory for writing FD visualization frames.
        box_sizes_series (str): Box sizes series string for FD computation.
        contour_thickness (int): Thickness used for boundary masks / overlays.

    Returns:
        dict|bool:
            If analyze=True: dict of extracted columns -> list of values.
            If analyze=False: True (success flag).
    """
    total_frames, read_frame, cleanup = open_media(media_path)

    extracted_data = {}
    for config in configs:
        frame_index = config["Frame Index"]
        ret, current_frame = read_frame(frame_index)
        if not ret:
            continue

        params = {
            "frame_index": config["Frame Index"],
            "crop_square": config["(Preprocess) Crop Square"],
            "crop_circle": config["(Preprocess) Crop Circle"],
            "crop_center_x": config["(Preprocess) Crop CenterX"],
            "crop_center_y": config["(Preprocess) Crop CenterY"],
            "tilt_strength": config["(Preprocess) Tilt Strength"],
            "tilt_center_x": config["(Preprocess) Tilt CenterX"],
            "tilt_center_y": config["(Preprocess) Tilt CenterY"],
            "tilt_rotation": config["(Preprocess) Tilt Rotation"],
            "lower_curve": config["(Preprocess) Lower Curve"],
            "upper_curve": config["(Preprocess) Upper Curve"],
            "blur_kernel": config["(Preprocess) Blur Kernel"],
            "morph_kernel": config["(Preprocess) Morph Kernel"],
            "morph_iterations": config["(Preprocess) Morph Iterations"],
            "lower_threshold": config["(Select) Lower Threshold"],
            "upper_threshold": config["(Select) Upper Threshold"],
        }

        (
            current_frame_cropped,
            current_frame_blurred,
            current_frame_thresh,
            thresh_contours,
            thresh_contours_filtered,
            cropped_x,
            cropped_y,
        ) = preprocess_frame(current_frame, **params)

        if analyze:
            results = extract_data(
                current_frame_blurred,
                thresh_contours_filtered,
                contour_thickness=contour_thickness,
                box_sizes_series=box_sizes_series,
                include_holes=include_holes,
                extract_fd_fits=extract_fd_fits,
                extract_fd_frames=extract_fd_frames,
                **params,
            )
            for col, value in results.items():
                extracted_data.setdefault(col, []).append(value)
        else:
            select_image, resulting_image, output_image = export_frame(
                current_frame,
                current_frame_cropped,
                current_frame_thresh,
                thresh_contours,
                thresh_contours_filtered,
                cropped_x=cropped_x,
                cropped_y=cropped_y,
                contour_thickness=contour_thickness,
                include_holes=include_holes,
                export_tilt=export_tilt,
                export_tilt_bar=export_tilt_bar,
                export_masked=export_masked,
                export_alpha=export_alpha,
                export_contours=export_contours,
                export_selects=export_selects,
                **params,
            )
            cv2.imwrite(
                os.path.join(frames_path, f"{frame_index:05}.png"),
                output_image,
                [cv2.IMWRITE_PNG_COMPRESSION, 9],
            )

    cleanup()

    return extracted_data if analyze else True


def main():
    """
    Launch the Tkinter GUI, initialize media sources, and wire UI callbacks.

    High-level responsibilities:
      - open initial media (default None => synthetic frames)
      - build UI: sliders, panels, record list, replay/extract/export controls
      - maintain state: records list, last_frame cache, current media
      - provide commands that:
          * update previews (update_region)
          * record/load/save JSON configs
          * interpolate between configs for replay/extraction/export
          * run multiprocessing extraction/export across available CPUs

    Returns:
        None
    """
    # Open the video file.
    media_path = None
    total_frames, read_frame, cleanup = open_media(media_path)
    ret, frame = read_frame(0)
    if not ret:
        print("Error reading video!")
        return

    image_shape = frame.shape  # (height, width, channels)
    video_w, video_h = image_shape[1], image_shape[0]
    video_aspect = video_w / video_h
    init_panel_w = video_w // 2
    init_panel_h = int(init_panel_w / video_aspect)
    init_disp_width = init_panel_w * 2 + 20
    init_disp_height = init_panel_h * 2 + 20

    root = tk.Tk()
    root.title("pyPetana v1.1")

    def apply_geometry():
        """
        Resize the main window based on current video dimensions/aspect.

        Uses nonlocal init_panel_w/init_panel_h/init_disp_width/init_disp_height
        and updates root geometry.
        """
        nonlocal init_panel_w, init_panel_h, init_disp_width, init_disp_height
        init_panel_w = video_w // 2
        init_panel_h = int(init_panel_w / video_aspect)
        init_disp_width = init_panel_w * 2 + 20
        init_disp_height = init_panel_h * 2 + 20

        root.geometry(f"{init_disp_width + 220}x{init_disp_height + 50}")
        root.update_idletasks()

    apply_geometry()

    # --- Left Control Panel ---
    control_frame = tk.Frame(root)
    control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)

    # Top row buttons
    initial_row_frame = tk.Frame(control_frame)
    initial_row_frame.pack(side=tk.TOP, fill=tk.X, pady=0, padx=0)
    open_file_button = tk.Button(initial_row_frame, text="Open JSON/Video File")
    open_file_button.pack(side=tk.LEFT, padx=0)
    open_directory_button = tk.Button(initial_row_frame, text="Open Image Directory")
    open_directory_button.pack(side=tk.LEFT, padx=0)
    save_json_button = tk.Button(initial_row_frame, text="Save JSON File")
    save_json_button.pack(side=tk.RIGHT, padx=0)

    # --- Right Display Panel (2x2 image grid) ---
    display_frame = tk.Frame(root)
    display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=0, pady=0)
    display_frame.grid_rowconfigure(0, weight=1)
    display_frame.grid_rowconfigure(1, weight=1)
    display_frame.grid_columnconfigure(0, weight=1)
    display_frame.grid_columnconfigure(1, weight=1)

    image_labels = []
    for r in range(2):
        for c in range(2):
            lbl = tk.Label(display_frame, bg="black")
            lbl.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
            image_labels.append(lbl)

    # Slider storage
    sliders = {}
    slider_scales = {}

    def create_slider(name, parent, frm, to, init, length=100, width=20, step=1):
        """
        Create a labeled Tk Scale with optional increment/decrement buttons.

        Stores the associated Tk variable in sliders[name] and the Scale widget in
        slider_scales[name].

        Args:
            name (str): Label/lookup key for slider.
            parent (tk.Widget): Parent container.
            frm (float|int): Minimum slider value.
            to (float|int): Maximum slider value.
            init (float|int): Initial slider value.
            length (int): Pixel length of scale widget.
            width (int): Label widget width.
            step (float|int): Slider resolution. If integer, uses IntVar; otherwise DoubleVar.

        Returns:
            None
        """
        # decide var type
        if float(step).is_integer():
            var = tk.IntVar(value=int(init))
            step_val = int(step)
        else:
            var = tk.DoubleVar(value=float(init))
            step_val = float(step)

        sliders[name] = var

        row_frame = tk.Frame(parent)
        row_frame.pack(fill=tk.X, pady=0, padx=0)
        row_frame.grid_columnconfigure(1, weight=1)

        label = tk.Label(row_frame, text=name, anchor="w", width=width)
        label.grid(row=0, column=0, sticky="w")

        scale = tk.Scale(
            row_frame,
            from_=frm,
            to=to,
            orient=tk.HORIZONTAL,
            variable=var,
            command=lambda _x: update_region(),
            showvalue=True,
            length=length,
            resolution=step_val,
        )
        scale.grid(row=0, column=1, sticky="ew", pady=0, padx=(0, 6))

        def _bump(delta):
            """
            Move the slider by a step delta, clamping to range and snapping floats.

            Args:
                delta (float|int): Amount to add to current scale value.
            """
            v = scale.get() + delta
            v = max(float(scale["from"]), min(float(scale["to"]), float(v)))
            if not float(step).is_integer():
                v = round((v - float(scale["from"])) / step_val) * step_val + float(
                    scale["from"]
                )
                v = round(v, 10)
            scale.set(v)

        up_button = tk.Button(row_frame, text="▲", command=lambda: _bump(step_val))
        up_button.grid(row=0, column=2, sticky="e", padx=(0, 4), pady=0)

        down_button = tk.Button(row_frame, text="▼", command=lambda: _bump(-step_val))
        down_button.grid(row=0, column=3, sticky="e", padx=0, pady=0)

        slider_scales[name] = scale

    def update_region(export_single_frame=False):
        """
        Core UI update: reads current slider values, processes the current frame,
        renders four preview panels, and optionally returns export data.

        Responsibilities:
          - Manage cached frame reads (last_frame / last_frame_index).
          - Enable/disable replay/export buttons based on records and frame index.
          - Build params dict from sliders.
          - Run preprocess_frame(**params) and export_frame(..., **params).
          - If export_single_frame=True: optionally run extract_data and return outputs.
          - Else: resize images for display and update Tk PhotoImages.

        Args:
            export_single_frame (bool): If True, do not update UI panels; instead return
                (output_image, thresh_contours_filtered, results).

        Returns:
            If export_single_frame:
                tuple[np.ndarray, list[np.ndarray], dict]: (output_image, contours, results)
            Else:
                tuple[np.ndarray, list[np.ndarray]]: (output_image, contours)
        """
        frame_index = sliders["Frame Index"].get()

        global last_frame_index
        global last_frame
        nonlocal records, total_frames

        if last_frame_index == frame_index:
            ret, current_frame = True, last_frame
        else:
            ret, current_frame = read_frame(frame_index)
            last_frame_index = frame_index
            last_frame = current_frame

            # Update navigation button states
            if records and len(records) > 0:
                interp_frame_button.config(state=tk.NORMAL)
                export_frame_button.config(state=tk.NORMAL)
                last_frame_button.config(
                    state=tk.NORMAL if frame_index > 0 else tk.DISABLED
                )
                next_frame_button.config(
                    state=tk.NORMAL if frame_index < total_frames - 1 else tk.DISABLED
                )
            else:
                last_frame_button.config(state=tk.DISABLED)
                interp_frame_button.config(state=tk.DISABLED)
                next_frame_button.config(state=tk.DISABLED)
                export_frame_button.config(state=tk.DISABLED)

        if not ret:
            return

        params = {
            "frame_index": sliders["Frame Index"].get(),
            "crop_square": sliders["(Preprocess) Crop Square"].get(),
            "crop_circle": sliders["(Preprocess) Crop Circle"].get(),
            "crop_center_x": sliders["(Preprocess) Crop CenterX"].get(),
            "crop_center_y": sliders["(Preprocess) Crop CenterY"].get(),
            "tilt_strength": sliders["(Preprocess) Tilt Strength"].get(),
            "tilt_center_x": sliders["(Preprocess) Tilt CenterX"].get(),
            "tilt_center_y": sliders["(Preprocess) Tilt CenterY"].get(),
            "tilt_rotation": sliders["(Preprocess) Tilt Rotation"].get(),
            "lower_curve": sliders["(Preprocess) Lower Curve"].get(),
            "upper_curve": sliders["(Preprocess) Upper Curve"].get(),
            "blur_kernel": sliders["(Preprocess) Blur Kernel"].get(),
            "morph_kernel": sliders["(Preprocess) Morph Kernel"].get(),
            "morph_iterations": sliders["(Preprocess) Morph Iterations"].get(),
            "lower_threshold": sliders["(Select) Lower Threshold"].get(),
            "upper_threshold": sliders["(Select) Upper Threshold"].get(),
        }

        (
            current_frame_cropped,
            current_frame_blurred,
            current_frame_thresh,
            thresh_contours,
            thresh_contours_filtered,
            cropped_x,
            cropped_y,
        ) = preprocess_frame(
            current_frame,
            **params,
        )

        select_image, resulting_image, output_image = export_frame(
            current_frame,
            current_frame_cropped,
            current_frame_thresh,
            thresh_contours,
            thresh_contours_filtered,
            cropped_x=cropped_x,
            cropped_y=cropped_y,
            contour_thickness=contour_thickness_var.get(),
            include_holes=include_holes_var.get(),
            export_tilt=export_tilt_var.get(),
            export_tilt_bar=export_tilt_bar_var.get(),
            export_masked=export_masked_var.get(),
            export_alpha=export_alpha_var.get(),
            export_contours=export_contours_var.get(),
            export_selects=export_selects_var.get(),
            **params,
        )

        if export_single_frame is True:
            extract_fd_fits = ""
            if extract_fd_fits_var.get():
                extract_fd_fits = media_path + ".extract"
                os.makedirs(extract_fd_fits, exist_ok=True)

            extract_fd_frames = ""
            if extract_fd_frames_var.get():
                extract_fd_frames = media_path + ".extract"
                os.makedirs(extract_fd_frames, exist_ok=True)

            if extract_fd_frames_var.get() or extract_fd_fits_var.get():
                results = extract_data(
                    current_frame_blurred,
                    thresh_contours_filtered,
                    contour_thickness=contour_thickness_var.get(),
                    box_sizes_series=box_sizes_series_var.get(),
                    include_holes=include_holes_var.get(),
                    extract_fd_fits=extract_fd_fits,
                    extract_fd_frames=extract_fd_frames,
                    **params,
                )
            else:
                results = {}
            return output_image, thresh_contours_filtered, results

        # --- UI display update branch ---
        this_image_shape = current_frame_cropped.shape
        this_video_w, this_video_h = this_image_shape[1], this_image_shape[0]
        this_video_aspect = this_video_w / this_video_h

        display_frame.update_idletasks()
        avail_w = display_frame.winfo_width() // 2 - 10
        avail_h = display_frame.winfo_height() // 2 - 10

        if avail_w / avail_h > this_video_aspect:
            panel_h = avail_h
            panel_w = int(avail_h * this_video_aspect)
        else:
            panel_w = avail_w
            panel_h = int(avail_w / this_video_aspect)

        original_image_resized = cv2.resize(
            current_frame_cropped, (panel_w, panel_h), interpolation=cv2.INTER_AREA
        )
        select_image_resized = cv2.resize(
            select_image, (panel_w, panel_h), interpolation=cv2.INTER_AREA
        )
        resulting_image_resized = cv2.resize(
            resulting_image, (panel_w, panel_h), interpolation=cv2.INTER_AREA
        )
        output_image_resized = cv2.resize(
            output_image, (panel_w, panel_h), interpolation=cv2.INTER_AREA
        )

        for index, image in enumerate(
            [
                original_image_resized,
                select_image_resized,
                resulting_image_resized,
                output_image_resized,
            ]
        ):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            imgtk = ImageTk.PhotoImage(pil_image)
            image_labels[index].imgtk = imgtk
            image_labels[index].config(image=imgtk)

        return output_image, thresh_contours_filtered

    # Primary slider
    create_slider(
        "Frame Index", control_frame, 0, total_frames - 1, 0, length=100, width=10
    )

    # --- Tabbed slider groups ---
    preprocess_tabs = ttk.Notebook(control_frame)
    preprocess_tabs.pack(side=tk.TOP, fill=tk.X, expand=False, padx=0, pady=0)

    preprocess_tab_common = ttk.Frame(preprocess_tabs)
    preprocess_tab_crop = ttk.Frame(preprocess_tabs)
    preprocess_tab_tilt = ttk.Frame(preprocess_tabs)

    preprocess_tabs.add(preprocess_tab_common, text="Common")
    preprocess_tabs.add(preprocess_tab_crop, text="Crop")
    preprocess_tabs.add(preprocess_tab_tilt, text="Tilt")

    select_tabs = ttk.Notebook(control_frame)
    select_tabs.pack(side=tk.TOP, fill=tk.X, expand=False, padx=0, pady=0)

    select_tab_threshold = ttk.Frame(select_tabs)
    select_tabs.add(select_tab_threshold, text="Threshold")

    # Slider defaults: key -> [parent, min, max, default]
    slider_defaults = {
        "(Preprocess) Lower Curve": [preprocess_tab_common, 0, 255, 0],
        "(Preprocess) Upper Curve": [preprocess_tab_common, 0, 255, 255],
        "(Preprocess) Blur Kernel": [preprocess_tab_common, 0, 50, 1],
        "(Preprocess) Morph Iterations": [preprocess_tab_common, 0, 10, 1],
        "(Preprocess) Morph Kernel": [preprocess_tab_common, 0, 50, 1],
        "(Preprocess) Crop Square": [preprocess_tab_crop, 0, 100, 0],
        "(Preprocess) Crop Circle": [preprocess_tab_crop, 0, 100, 0],
        "(Preprocess) Crop CenterX": [preprocess_tab_crop, 0, 100, 50],
        "(Preprocess) Crop CenterY": [preprocess_tab_crop, 0, 100, 50],
        "(Preprocess) Tilt Strength": [preprocess_tab_tilt, -100, 100, 0],
        "(Preprocess) Tilt CenterX": [preprocess_tab_tilt, 0, 100, 50],
        "(Preprocess) Tilt CenterY": [preprocess_tab_tilt, 0, 100, 50],
        "(Preprocess) Tilt Rotation": [preprocess_tab_tilt, 0, 180, 0],
        "(Select) Lower Threshold": [select_tab_threshold, 0, 256, 0],
        "(Select) Upper Threshold": [select_tab_threshold, 0, 256, 256],
    }

    def tilt_apply_all_command():
        """
        Apply the current tilt slider values to all records.

        This updates existing records in-place so replay/extraction uses the same tilt settings.
        """
        nonlocal records, sliders
        tilt_strength = sliders["(Preprocess) Tilt Strength"].get()
        tilt_centerx = sliders["(Preprocess) Tilt CenterX"].get()
        tilt_centery = sliders["(Preprocess) Tilt CenterY"].get()
        tilt_rotation = sliders["(Preprocess) Tilt Rotation"].get()
        for record in records:
            record["(Preprocess) Tilt Strength"] = tilt_strength
            record["(Preprocess) Tilt CenterX"] = tilt_centerx
            record["(Preprocess) Tilt CenterY"] = tilt_centery
            record["(Preprocess) Tilt Rotation"] = tilt_rotation

    def crop_apply_all_command():
        """
        Apply the current crop slider values to all records.

        This updates existing records in-place so replay/extraction uses the same crop settings.
        """
        nonlocal records, sliders
        crop_square = sliders["(Preprocess) Crop Square"].get()
        crop_circle = sliders["(Preprocess) Crop Circle"].get()
        crop_centerx = sliders["(Preprocess) Crop CenterX"].get()
        crop_centery = sliders["(Preprocess) Crop CenterY"].get()
        for record in records:
            record["(Preprocess) Crop Square"] = crop_square
            record["(Preprocess) Crop Circle"] = crop_circle
            record["(Preprocess) Crop CenterX"] = crop_centerx
            record["(Preprocess) Crop CenterY"] = crop_centery

    crop_apply_all = tk.Button(
        preprocess_tab_crop, text="Apply All", command=crop_apply_all_command
    )
    crop_apply_all.pack(side=tk.BOTTOM, fill=tk.X, pady=0)

    tilt_apply_all = tk.Button(
        preprocess_tab_tilt, text="Apply All", command=tilt_apply_all_command
    )
    tilt_apply_all.pack(side=tk.BOTTOM, fill=tk.X, pady=0)

    # Create remaining sliders
    for key, args in slider_defaults.items():
        step = 1
        if key in [
            "(Preprocess) Crop Square",
            "(Preprocess) Crop Circle",
            "(Preprocess) Crop CenterX",
            "(Preprocess) Crop CenterY",
            "(Preprocess) Tilt Strength",
            "(Preprocess) Tilt CenterX",
            "(Preprocess) Tilt CenterY",
            "(Preprocess) Tilt Rotation",
        ]:
            step = 0.25
        create_slider(key, *args, step=step)

    # --- Record Configuration Section ---
    records = []

    def add_record(override_config=None):
        """
        Add a new record (or replace existing record) for the current frame index.

        Records are kept sorted by 'Frame Index'. If a record with the same frame index
        already exists, it is replaced.

        Args:
            override_config (dict|None): If provided, insert/replace using this dict
                rather than reading current slider values.

        Returns:
            None
        """
        if override_config is not None:
            config = override_config
        else:
            config = {key: var.get() for key, var in sliders.items()}

        new_frame_index = config["Frame Index"]
        frame_label = f"Frame {new_frame_index:03d}"

        insert_pos = 0
        replace_pos = -1
        for i, rec in enumerate(records):
            if rec["Frame Index"] > new_frame_index:
                insert_pos = i
                break
            if records[i]["Frame Index"] == new_frame_index:
                replace_pos = i
                break
        else:
            insert_pos = len(records)

        if replace_pos != -1:
            records[replace_pos] = config
        else:
            records.insert(insert_pos, config)
            record_listbox.insert(insert_pos, frame_label)

    def remove_record():
        """
        Remove the currently selected record from the listbox and records array.

        Returns:
            None
        """
        selection = record_listbox.curselection()
        if selection:
            index = selection[0]
            record_listbox.delete(index)
            del records[index]

    def save_records():
        """
        Save records to a JSON file (with media_path and relative json_path).

        Writes:
          payload = {version, media_path, records, json_path}

        Returns:
            None
        """
        if len(records) > 0:
            payload = {
                "version": 1.1,
                "media_path": str(media_path),
                "records": records,
            }
            json_path = filedialog.asksaveasfilename(
                title="Save Records",
                initialfile="".join(media_path.split("/")[-1].split(".")[:-1]),
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            parent = os.path.dirname(os.path.abspath(json_path))
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            json_path_rel = json_path_relative_to_media(media_path, json_path)
            payload["json_path"] = json_path_rel
            with open(json_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
                f.write("\n")

            messagebox.showinfo("Info", f"Records saved to:\n{json_path}!", parent=root)
        else:
            messagebox.showerror("Error", "No records to save!", parent=root)

    def load_records(override_json_path=None):
        """
        Load records from a JSON file produced by save_records().

        Validates that the loaded media_path filename matches current media filename.

        Args:
            override_json_path (str|None): If provided, loads from this path rather than dialog.

        Returns:
            None
        """
        if override_json_path is not None:
            json_path = override_json_path
        else:
            json_path = filedialog.askopenfilename(
                title="Load Records",
                initialfile="".join(media_path.split("/")[-1].split(".")[:-1]),
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if (
            not isinstance(payload, dict)
            or "records" not in payload
            or "media_path" not in payload
        ):
            messagebox.showerror(
                "Error",
                "JSON file doesn't have necessary fields, is it the right JSON?",
            )
        elif (
            payload["media_path"].split("/")[-1].strip()
            != str(media_path).split("/")[-1].strip()
        ):
            messagebox.showerror("Error", "The media path doesn't match, not loading!")
        else:
            if payload.get("version") != 1.1:
                messagebox.showwarning(
                    "Warning",
                    f"JSON file specified pyPetana version {payload.get('version')}, you are running version 1.1! Will attempt load anyways.",
                    parent=root,
                )
            record_listbox.selection_clear(0, tk.END)
            record_listbox.delete(0, tk.END)
            records.clear()
            for record in payload["records"]:
                add_record(override_config=record)
            messagebox.showinfo(
                "Info", f"Records loaded from:\n{json_path}!", parent=root
            )

    # Record list UI
    record_frame = tk.Frame(control_frame)
    record_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=0, pady=0)

    record_buttons_frame = tk.Frame(record_frame)
    record_buttons_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

    record_button_add = tk.Button(
        record_buttons_frame, text="Add Record", command=add_record
    )
    record_button_add.pack(fill=tk.X, pady=1)

    record_button_remove = tk.Button(
        record_buttons_frame, text="Remove Record", command=remove_record
    )
    record_button_remove.pack(fill=tk.X, pady=1)

    record_list_frame = tk.Frame(record_frame)
    record_list_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    record_listbox = tk.Listbox(record_list_frame, height=8, width=20)
    record_listbox.pack(fill=tk.BOTH, expand=True)

    def on_record_select(event):
        """
        When a record is selected in the listbox:
          - load its values into sliders
          - call update_region() to update views

        Args:
            event: Tk event object.
        """
        nonlocal slider_defaults, records
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            config = records[index]
            sliders["Frame Index"].set(config.get("Frame Index"))
            for key, [_, _, _, def_val] in slider_defaults.items():
                val = config.get(key, def_val)
                sliders[key].set(val)
            update_region()

    record_listbox.bind("<<ListboxSelect>>", on_record_select)

    # --- Replay Section / Vars ---
    stop_replay_flag = False
    frame_skip = tk.IntVar(value=0)
    frame_delay = tk.StringVar(value="16.67")
    include_holes_var = tk.BooleanVar(value=False)
    export_tilt_var = tk.BooleanVar(value=False)
    export_tilt_bar_var = tk.BooleanVar(value=False)
    export_masked_var = tk.BooleanVar(value=False)
    export_alpha_var = tk.BooleanVar(value=False)
    export_contours_var = tk.BooleanVar(value=False)
    export_selects_var = tk.BooleanVar(value=False)
    extract_fd_fits_var = tk.BooleanVar(value=False)
    extract_fd_frames_var = tk.BooleanVar(value=False)
    contour_thickness_var = tk.IntVar(value=1)
    box_sizes_series_var = tk.StringVar(value="5, 6, 7, 8, 9")

    def stop_replay_command():
        """
        Set flag to stop the replay loop at the next iteration.
        """
        nonlocal stop_replay_flag
        stop_replay_flag = True

    def prepare_interp_configs():
        """
        Build a list of interpolated configs between consecutive records.

        For each consecutive record pair (start, end):
          - include start frame config
          - for each intermediate frame: linearly interpolate each slider value
          - include final record config

        Returns:
            list[dict]: List of per-frame configs including 'Frame Index' and slider keys.
        """
        nonlocal slider_defaults, records

        interp_configs = []
        for i in range(len(records) - 1):
            start_config = records[i]
            end_config = records[i + 1]
            start_frame = start_config["Frame Index"]
            end_frame = end_config["Frame Index"]
            steps = end_frame - start_frame
            if steps <= 0:
                continue

            # Append the start frame
            interp_config = {"Frame Index": start_frame}
            for key, [_, _, _, def_val] in slider_defaults.items():
                interp_config[key] = start_config.get(key, def_val)
            interp_configs.append(interp_config)

            # Intermediate frames
            this_frame = start_frame + 1
            while this_frame < end_frame:
                t = (this_frame - start_frame) / steps
                interp_config = {"Frame Index": this_frame}
                for key, [_, _, _, def_val] in slider_defaults.items():
                    start_val = start_config.get(key, def_val)
                    end_val = end_config.get(key, def_val)
                    if float(slider_scales[key]["resolution"]).is_integer():
                        interp_config[key] = int(
                            round(start_val + t * (end_val - start_val))
                        )
                    else:
                        interp_config[key] = start_val + t * (end_val - start_val)
                interp_configs.append(interp_config)
                this_frame += 1

        # Append final record
        interp_config = {"Frame Index": records[-1]["Frame Index"]}
        for key, [_, _, _, def_val] in slider_defaults.items():
            interp_config[key] = records[-1].get(key, def_val)
        interp_configs.append(interp_config)

        return interp_configs

    def replay_records_command():
        """
        Replay interpolated configs across frames, updating sliders and display.

        Uses frame_skip and frame_delay to control playback.

        Returns:
            None
        """
        nonlocal stop_replay_flag
        if len(records) < 2:
            print("At least two records are required for replay.")
            return

        replay_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
        stop_replay_flag = False

        interp_configs = prepare_interp_configs()
        skip_val = frame_skip.get()
        delay_val = float(frame_delay.get())

        for i in range(0, len(interp_configs), skip_val + 1):
            if stop_replay_flag:
                break

            for key in sliders:
                if key in interp_configs[i]:
                    sliders[key].set(interp_configs[i][key])
            update_region()
            root.update()
            time.sleep(delay_val / 1000)

        replay_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
        stop_replay_flag = False

    def extract_data_command():
        """
        Run extraction across all interpolated frames using multiprocessing.

        Writes a .dat file containing extracted columns (sorted by frame_id).
        Optionally writes FD frames and FD fit tables under media_path + ".extract".

        Returns:
            None
        """
        nonlocal media_path
        file_path = filedialog.asksaveasfilename(
            title="Save Extraction Results",
            initialfile="".join(media_path.split("/")[-1].split(".")[:-1]),
            defaultextension=".dat",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")],
        )

        interp_configs = prepare_interp_configs()

        num_workers = _available_logical_cpus()
        frames_per_worker = int(np.ceil(len(interp_configs) / num_workers))

        extract_fd_fits = ""
        if extract_fd_fits_var.get():
            extract_fd_fits = media_path + ".extract"
            os.makedirs(extract_fd_fits, exist_ok=True)

        extract_fd_frames = ""
        if extract_fd_frames_var.get():
            extract_fd_frames = media_path + ".extract"
            os.makedirs(extract_fd_frames, exist_ok=True)

        extracted_data = {}
        tasks = []
        with concurrent.futures.ProcessPoolExecutor(
            initializer=_worker_init
        ) as executor:
            for worker in range(num_workers):
                fc = int(worker * frames_per_worker)
                lc = int(fc + frames_per_worker)
                future = executor.submit(
                    process_interpolated_frames,
                    media_path,
                    interp_configs[fc:lc],
                    analyze=True,
                    include_holes=include_holes_var.get(),
                    export_tilt=export_tilt_var.get(),
                    export_tilt_bar=export_tilt_bar_var.get(),
                    export_masked=export_masked_var.get(),
                    export_alpha=export_alpha_var.get(),
                    export_contours=export_contours_var.get(),
                    export_selects=export_selects_var.get(),
                    extract_fd_fits=extract_fd_fits,
                    extract_fd_frames=extract_fd_frames,
                    box_sizes_series=box_sizes_series_var.get(),
                    contour_thickness=contour_thickness_var.get(),
                )
                tasks.append(future)

            for future in concurrent.futures.as_completed(tasks):
                result = future.result()
                if result is None:
                    continue
                for col, values in result.items():
                    extracted_data.setdefault(col, []).extend(values)

        array = np.column_stack([extracted_data[key] for key in extracted_data])
        dtype = [
            (key, "i4") if key == "frame_id" else (key, "f8") for key in extracted_data
        ]
        structured_array = np.array([tuple(row) for row in array], dtype=dtype)
        structured_array.sort(order="frame_id")

        keys = [key for key, _ in dtype]
        header = f"{keys[0]:>{max(len(keys[0]),12)}}  " + "  ".join(
            f"{k:>{max(len(k),13)}}" for k in keys[1:]
        )
        fmt = [
            f"%{max(len(key),13)}d" if t == "i4" else f"%{max(len(key)+1,14)}e"
            for key, t in dtype
        ]
        meta = {
            "media_basename": os.path.basename(os.path.normpath(media_path)),
            "contour_thickness": contour_thickness_var.get(),
            "box_sizes_series": box_sizes_series_var.get(),
        }
        meta_lines = [f" meta_json={json.dumps(meta, separators=(',', ':'))}"]
        np.savetxt(
            file_path,
            structured_array,
            fmt=fmt,
            header="\n".join([header] + meta_lines),
            comments="#",
        )

        messagebox.showinfo(
            "Info", f"Extraction Results Saved to:\n{file_path}", parent=root
        )

    def on_key_press(event):
        """
        Left/Right arrow navigation for frame index slider.

        Args:
            event: Tk keypress event.
        """
        curr = sliders["Frame Index"].get()
        if event.keysym == "Left" and curr > 0:
            sliders["Frame Index"].set(curr - 1)
        elif event.keysym == "Right" and curr < total_frames - 1:
            sliders["Frame Index"].set(curr + 1)
        update_region()

    root.bind("<Left>", on_key_press)
    root.bind("<Right>", on_key_press)

    def export_frame_command():
        """
        Export the currently displayed output image to a PNG chosen by user.

        Uses update_region(export_single_frame=True) so it returns the actual export image.

        Returns:
            None
        """
        nonlocal sliders
        this_frame_index = sliders["Frame Index"].get()
        file_path = filedialog.asksaveasfilename(
            title="Export Frame Filename",
            initialfile="".join(media_path.split("/")[-1].split(".")[:-1])
            + f"_f{this_frame_index}.png",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
        )
        if file_path:
            output_image, output_contour, results = update_region(
                export_single_frame=True
            )
            cv2.imwrite(file_path, output_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def interp_frame_command(frame_index_offset, export_frame=False):
        """
        Jump to a frame index (relative) and apply interpolated config closest to it.

        Args:
            frame_index_offset (int): Delta to apply to current Frame Index.
            export_frame (bool): If True, prompt for output file and export after update.

        Returns:
            None
        """
        nonlocal sliders, total_frames

        this_frame_index = sliders["Frame Index"].get() + frame_index_offset
        this_frame_index = max(0, min(total_frames, this_frame_index))

        interp_configs = prepare_interp_configs()

        interp_configs_index = 0
        while (
            interp_configs_index < len(interp_configs) - 1
            and interp_configs[interp_configs_index]["Frame Index"] < this_frame_index
        ):
            interp_configs_index += 1
        interp_config = interp_configs[interp_configs_index]

        for key in sliders:
            if key in interp_config and key != "Frame Index":
                sliders[key].set(interp_config[key])

        if frame_index_offset != 0:
            sliders["Frame Index"].set(this_frame_index)

        output_image, output_contour = update_region()
        root.update()

        if export_frame:
            file_path = filedialog.asksaveasfilename(
                title="Export Frame Filename",
                initialfile="".join(media_path.split("/")[-1].split(".")[:-1])
                + f"_f{this_frame_index}.png",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png")],
            )
            cv2.imwrite(file_path, output_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # --- Replay Controls UI ---
    tk.Frame(record_buttons_frame).pack(fill=tk.Y, expand=True)
    last_frame_button = tk.Button(
        record_buttons_frame,
        text="Last Frame",
        command=lambda: interp_frame_command(-1),
        state=tk.DISABLED,
    )
    last_frame_button.pack(fill=tk.X, pady=1)
    interp_frame_button = tk.Button(
        record_buttons_frame,
        text="Interp Frame",
        command=lambda: interp_frame_command(0),
        state=tk.DISABLED,
    )
    interp_frame_button.pack(fill=tk.X, pady=1)
    next_frame_button = tk.Button(
        record_buttons_frame,
        text="Next Frame",
        command=lambda: interp_frame_command(1),
        state=tk.DISABLED,
    )
    next_frame_button.pack(fill=tk.X, pady=1)
    export_frame_button = tk.Button(
        record_buttons_frame,
        text="Export Frame",
        command=lambda: export_frame_command(),
        state=tk.DISABLED,
    )
    export_frame_button.pack(fill=tk.X, pady=1)
    tk.Frame(record_buttons_frame).pack(fill=tk.Y, expand=True)
    replay_button = tk.Button(
        record_buttons_frame, text="Start Replay", command=replay_records_command
    )
    replay_button.pack(fill=tk.X, pady=1)
    stop_button = tk.Button(
        record_buttons_frame,
        text="Stop Replay",
        command=stop_replay_command,
        state=tk.DISABLED,
    )
    stop_button.pack(fill=tk.X, pady=1)

    skip_row = tk.Frame(record_buttons_frame)
    skip_row.pack(fill=tk.X, pady=1)

    skip_text = tk.StringVar()

    def _refresh_skip_label():
        """
        Update the skip label text based on current frame_skip value.
        """
        skip_text.set(f"Frame Skip: {frame_skip.get()}")

    def _set_skip(v):
        """
        Set frame skip value with bounds checking and refresh label.

        Args:
            v (int): New skip value.
        """
        v = int(v)
        v = max(0, min(100, v))
        frame_skip.set(v)
        _refresh_skip_label()

    skip_label = tk.Label(skip_row, textvariable=skip_text, anchor="w")
    skip_label.pack(side=tk.LEFT)

    btn_up = tk.Button(
        skip_row, text="▲", width=2, command=lambda: _set_skip(frame_skip.get() + 1)
    )
    btn_up.pack(side=tk.RIGHT, padx=(2, 0))

    btn_dn = tk.Button(
        skip_row, text="▼", width=2, command=lambda: _set_skip(frame_skip.get() - 1)
    )
    btn_dn.pack(side=tk.RIGHT, padx=(6, 0))

    _refresh_skip_label()

    delay_row = tk.Frame(record_buttons_frame)
    delay_row.pack(fill=tk.X, pady=1)

    delay_float_entry = tk.Entry(
        delay_row, textvariable=frame_delay, width=8, justify="right"
    )
    delay_float_entry.pack(side=tk.RIGHT, padx=0)
    delay_label = tk.Label(delay_row, text="Frame Delay (ms)")
    delay_label.pack(side=tk.LEFT, padx=0)

    def export_frames_command():
        """
        Export all interpolated frames to disk using multiprocessing.

        Prompts for a folder, then writes frames to:
            <chosen_dir>/<basename(media_path)>.export/

        Returns:
            None
        """
        nonlocal media_path
        media_path_parent = os.path.dirname(os.path.normpath(media_path))
        frames_path = filedialog.askdirectory(
            title="Export Frames Folder",
            initialdir=media_path_parent,
        )
        frames_path = os.path.join(
            frames_path, os.path.basename(media_path) + ".export"
        )
        os.makedirs(frames_path, exist_ok=True)

        interp_configs = prepare_interp_configs()

        num_workers = _available_logical_cpus()
        frames_per_worker = int(np.ceil(len(interp_configs) / num_workers))

        tasks = []
        with concurrent.futures.ProcessPoolExecutor(
            initializer=_worker_init
        ) as executor:
            for worker in range(num_workers):
                fc = int(worker * frames_per_worker)
                lc = int(fc + frames_per_worker)
                future = executor.submit(
                    process_interpolated_frames,
                    media_path,
                    interp_configs[fc:lc],
                    analyze=False,
                    include_holes=include_holes_var.get(),
                    export_tilt=export_tilt_var.get(),
                    export_tilt_bar=export_tilt_bar_var.get(),
                    export_masked=export_masked_var.get(),
                    export_alpha=export_alpha_var.get(),
                    export_contours=export_contours_var.get(),
                    export_selects=export_selects_var.get(),
                    frames_path=frames_path,
                )
                tasks.append(future)

            for future in concurrent.futures.as_completed(tasks):
                _ = future.result()

        messagebox.showinfo(
            "Info", f"Exported Frames Saved to:\n{frames_path}", parent=root
        )

    # --- Process Tabs ---
    process_tabs = ttk.Notebook(control_frame)
    process_tabs.pack(side=tk.TOP, fill=tk.X, expand=False, padx=0, pady=0)

    process_tab_extract_export = ttk.Frame(process_tabs)
    process_tab_advanced = ttk.Frame(process_tabs)

    process_tabs.add(process_tab_extract_export, text="Extract/Export")
    process_tabs.add(process_tab_advanced, text="Advanced Settings")

    contour_thickness_text = tk.StringVar()

    def _refresh_contour_thickness_label():
        """
        Update contour thickness label text and refresh preview.
        """
        contour_thickness_text.set(
            f"FD Contour Thickness: {contour_thickness_var.get()}"
        )
        update_region()

    def _set_contour_thickness(v):
        """
        Set contour thickness (bounded) and refresh label.

        Args:
            v (int): New contour thickness value.
        """
        v = int(v)
        v = max(1, min(100, v))
        contour_thickness_var.set(v)
        _refresh_contour_thickness_label()

    # --- UI layout continues (unchanged logic) ---
    process_tab_rows = [
        tk.Frame(process_tab_extract_export),
        tk.Frame(process_tab_extract_export),
    ]
    process_tab_rows[0].pack(fill=tk.X)
    process_tab_rows[1].pack(fill=tk.X)

    contour_thickness_label = tk.Label(
        process_tab_rows[0], textvariable=contour_thickness_text, anchor="w"
    )
    contour_thickness_label.pack(side=tk.LEFT, padx=(0, 0))

    contour_thickness_btn_up = tk.Button(
        process_tab_rows[0],
        text="▲",
        width=2,
        command=lambda: _set_contour_thickness(contour_thickness_var.get() + 1),
    )
    contour_thickness_btn_up.pack(side=tk.LEFT, padx=(0, 0))

    contour_thickness_btn_dn = tk.Button(
        process_tab_rows[0],
        text="▼",
        width=2,
        command=lambda: _set_contour_thickness(contour_thickness_var.get() - 1),
    )
    contour_thickness_btn_dn.pack(side=tk.LEFT, padx=(0, 0))

    _refresh_contour_thickness_label()

    box_sizes_series_label = tk.Label(
        process_tab_rows[1], text="FD Box Size Series", anchor="w", width=15
    )
    box_sizes_series_label.pack(side=tk.LEFT, padx=0)
    box_sizes_series_entry = tk.Entry(
        process_tab_rows[1],
        textvariable=box_sizes_series_var,
        width=12,
        justify="right",
    )
    box_sizes_series_entry.pack(side=tk.LEFT, padx=0)

    export_frames_button = tk.Button(
        process_tab_rows[0],
        text="Export Frames",
        command=export_frames_command,
        width=10,
    )
    export_frames_button.pack(side=tk.RIGHT, padx=0)
    analyze_frames_button = tk.Button(
        process_tab_rows[1], text="Extract Data", command=extract_data_command, width=10
    )
    analyze_frames_button.pack(side=tk.RIGHT, padx=0)

    advanced_tab_rows = [
        tk.Frame(process_tab_advanced),
        tk.Frame(process_tab_advanced),
        tk.Frame(process_tab_advanced),
    ]
    advanced_tab_rows[0].pack(fill=tk.X)
    advanced_tab_rows[1].pack(fill=tk.X)
    advanced_tab_rows[2].pack(fill=tk.X)

    advanced_cb_width = 14
    include_holes_cb = ttk.Checkbutton(
        advanced_tab_rows[0],
        text="Include Holes",
        variable=include_holes_var,
        command=update_region,
        width=advanced_cb_width,
    )
    include_holes_cb.pack(side=tk.LEFT, padx=2)
    extract_fd_fits_cb = ttk.Checkbutton(
        advanced_tab_rows[0],
        text="Extract FD Fits",
        variable=extract_fd_fits_var,
        command=update_region,
        width=advanced_cb_width,
    )
    extract_fd_fits_cb.pack(side=tk.LEFT, padx=2)
    extract_fd_frames_cb = ttk.Checkbutton(
        advanced_tab_rows[0],
        text="Extract FD Frames",
        variable=extract_fd_frames_var,
        command=update_region,
        width=advanced_cb_width,
    )
    extract_fd_frames_cb.pack(side=tk.LEFT, padx=2)

    export_selects_cb = ttk.Checkbutton(
        advanced_tab_rows[1],
        text="Export Selects",
        variable=export_selects_var,
        command=update_region,
        width=advanced_cb_width,
    )
    export_selects_cb.pack(side=tk.LEFT, padx=2)
    export_tilt_cb = ttk.Checkbutton(
        advanced_tab_rows[1],
        text="Export Tilt",
        variable=export_tilt_var,
        command=update_region,
        width=advanced_cb_width,
    )
    export_tilt_cb.pack(side=tk.LEFT, padx=2)
    export_tilt_bar_cb = ttk.Checkbutton(
        advanced_tab_rows[1],
        text="Export Tilt Bar",
        variable=export_tilt_bar_var,
        command=update_region,
        width=advanced_cb_width,
    )
    export_tilt_bar_cb.pack(side=tk.LEFT, padx=2)

    export_contours_cb = ttk.Checkbutton(
        advanced_tab_rows[2],
        text="Export Contours",
        variable=export_contours_var,
        command=update_region,
        width=advanced_cb_width,
    )
    export_contours_cb.pack(side=tk.LEFT, padx=2)
    export_masked_cb = ttk.Checkbutton(
        advanced_tab_rows[2],
        text="Export Masked",
        variable=export_masked_var,
        command=update_region,
        width=advanced_cb_width,
    )
    export_masked_cb.pack(side=tk.LEFT, padx=2)
    export_alpha_cb = ttk.Checkbutton(
        advanced_tab_rows[2],
        text="Export Alpha",
        variable=export_alpha_var,
        command=update_region,
        width=advanced_cb_width,
    )
    export_alpha_cb.pack(side=tk.LEFT, padx=2)

    bottom_frame = tk.Frame(control_frame)
    bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=0, padx=0)
    quit_button = tk.Button(
        bottom_frame, text="Quit", command=lambda: (cleanup(), root.destroy())
    )
    quit_button.pack(side=tk.RIGHT, padx=0)

    def switch_to_new_media(media_path_new, directory=False):
        """
        Switch the current media source (video or image directory) and reset UI state.

        Steps:
          - Validate path exists; prompt user to locate it if not.
          - Open new media via open_media().
          - Cleanup old capture resources and reset records/sliders.
          - Reconfigure frame index slider maximum and reset to 0.
          - Reset cached last_frame and refresh display.

        Args:
            media_path_new (str): New media file or directory path.
            directory (bool): Reserved (not used); kept for API compatibility.

        Returns:
            None
        """
        nonlocal media_path, total_frames, read_frame, cleanup
        global last_frame_index, last_frame

        if not os.path.exists(media_path_new):
            messagebox.showwarning(
                "Warning",
                "JSON file had incorrect path to media/folder: select the matching media/folder in the next dialogue!",
                parent=root,
            )
            media_filename = os.path.basename(os.path.normpath(media_path_new))
            if media_path_new.lower().endswith(
                (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm")
            ):
                media_path_new = filedialog.askopenfilename(
                    title=f"Select Video: {media_filename}",
                    initialdir=os.getcwd(),
                    filetypes=[
                        ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.wmv *.webm")
                    ],
                )
            else:
                media_path_new = filedialog.askdirectory(
                    title=f"Select Directory: {media_filename}",
                    initialdir=os.getcwd(),
                )

        total_frames_new, read_frame_new, cleanup_new = open_media(media_path_new)
        if total_frames_new is None:
            messagebox.showerror(
                "Error", f"Unable to open media:\n{media_path}", parent=root
            )
            return

        # cleanup previous media state
        cleanup()
        slider_scales["Frame Index"].set(0)
        record_listbox.selection_clear(0, tk.END)
        record_listbox.delete(0, tk.END)
        records.clear()

        # assign new media
        media_path = media_path_new
        total_frames = total_frames_new
        read_frame = read_frame_new
        cleanup = cleanup_new

        # update GUI
        slider_scales["Frame Index"].config(to=total_frames - 1)
        apply_geometry()
        root.update()
        slider_scales["Frame Index"].set(0)
        for key, args in slider_defaults.items():
            slider_scales[key].set(args[-1])

        last_frame = None
        last_frame_index = -1
        update_region()

    def open_file():
        """
        Open a JSON or video file via file dialog.

        If JSON is selected:
          - load payload, switch media to payload["media_path"], then load records.
        Else:
          - treat selection as media path and switch directly.

        Returns:
            None
        """
        file_path = filedialog.askopenfilename(
            title="Select JSON/Video File",
            initialdir=os.getcwd(),
            filetypes=[
                ("Supported", "*.json *.mp4 *.mov *.avi *.mkv *.m4v *.wmv *.webm"),
                ("JSON files", "*.json"),
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.wmv *.webm"),
            ],
        )
        if file_path:
            if file_path.lower().endswith(".json"):
                json_path = file_path
                with open(file_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                file_path = payload["media_path"]
                switch_to_new_media(file_path)
                load_records(override_json_path=json_path)
            else:
                switch_to_new_media(file_path)

    def open_directory():
        """
        Open an image directory via directory selection dialog and switch media.

        Returns:
            None
        """
        image_directory = filedialog.askdirectory(
            title="Select Image Directory",
            initialdir=os.getcwd(),
        )
        if image_directory:
            switch_to_new_media(image_directory)

    # Wire buttons
    open_file_button.config(command=open_file)
    open_directory_button.config(command=open_directory)
    save_json_button.config(command=save_records)

    # Initial update to display the first frame.
    update_region()
    root.mainloop()


if __name__ == "__main__":
    main()
