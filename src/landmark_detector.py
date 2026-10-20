import os
import sys
import cv2

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import numpy as np
import matplotlib.pyplot as plt
from landmark_visualizer import draw_landmarks_on_image, plot_face_blendshapes_bar_graph


# Key landmark indices (similar to dlib's 79-point model) + some additional points for better coverage
KEY_LANDMARKS = {
    # Jaw line (25 points) - from right ear, along jaw to chin, to left ear
    'jaw': [
        234, 93, 132, 58, 172, 136, 150, 149, 176, 148, # Right side (ear to chin)
        152,  # Chin center
        377, 400, 378, 379, 365, 397, 288, 435, 361, 401,323, 366, 454  # Left side (chin to ear)
    ],
    'right_eye_center': [468],  # Center of right eye (1 point)
    'left_eye_center': [473],   # Center of left eye (1 point)
    # Right eyebrow (5 points) - lateral to medial
    'right_eyebrow': [46, 53, 52, 65, 55],
    # Left eyebrow (5 points) - medial to lateral
    'left_eyebrow': [285, 295, 282, 283, 276],
    # Nose sides (7 points) - from right to left
    'nose_sides': [1, 49, 48, 219, 279, 278, 439],
    # Right eye (11 points) - clockwise from lateral corner
    'right_eye': [33, 161, 160, 159, 158, 157, 133, 153, 145, 144, 163],
    # Left eye (11 points) - clockwise from lateral corner
    'left_eye': [362, 384, 385, 386, 387, 263, 390, 373, 374, 380, 381],
    # lips (14 points) - clockwise from right corner
    'outer_lips': [61, 40, 37, 0, 267, 270, 291, 375, 405, 314, 17, 84, 181, 91]
}

# Flattened list of all 79 key landmark indices
KEY_LANDMARK_INDICES = []
for group in KEY_LANDMARKS.values():
    KEY_LANDMARK_INDICES.extend(group)

# RGB color (0-1 per channel) assigned to each facial-feature group. Used to
# composite all groups into a single 3-channel image while keeping group identity
# distinguishable by hue.
GROUP_COLORS = {
    'jaw':              (1.0, 0.0, 0.0),  # red
    'forehead':         (0.6, 0.0, 1.0),  # purple
    'right_eye_center': (0.0, 1.0, 1.0),  # cyan
    'left_eye_center':  (0.0, 1.0, 1.0),  # cyan
    'right_eyebrow':    (1.0, 0.6, 0.0),  # orange
    'left_eyebrow':     (1.0, 0.6, 0.0),  # orange
    'nose_sides':       (0.0, 1.0, 0.0),  # green
    'right_eye':        (0.0, 0.4, 1.0),  # blue
    'left_eye':         (0.0, 0.4, 1.0),  # blue
    'outer_lips':       (1.0, 0.0, 0.6),  # pink
}

# Image Configuration
TARGET_IMAGE_SIZE = 256  # The final size for all images (256x256)
PADDING_FACTOR = 1.4     # 1.0 is a tight crop, 1.4 adds 40% margin around the face

# Heatmap Configuration
HEATMAP_SIZE = 256       # Output heatmap resolution (256x256)
HEATMAP_SIGMA = 4.0      # Gaussian std (in pixels) for each landmark blob, controls the spread/"blur" of the point

# Motion (displacement) heatmap configuration.
# Instead of marking WHERE the landmarks are, these encode HOW FAR each landmark
# moved away from its position in the neutral frame.
NEUTRAL_IMAGE = '00000.jpg'      # the neutral / rest-pose reference frame
MOTION_SIGMA = 4.0               # Gaussian std (px) of the blob carrying each displacement
MOTION_PERCENTILE = 99.9         # percentile of |displacement| used to normalize to ~[-1, 1]
MOTION_ANCHOR = 'neutral'        # draw each blob at the landmark's 'neutral' or 'deformed' position
BLOB_RADIUS_SIGMAS = 4.0         # only rasterize a blob within this many sigmas (beyond it exp() ~ 0)

# Precompute the pixel grid once; reused for every Gaussian we render
hm_y, hm_x = np.mgrid[0:HEATMAP_SIZE, 0:HEATMAP_SIZE]  # image coordinates (y, x) for the heatmap grid

# Setup paths (relative to this file's location)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'face_landmarker_v2_with_blendshapes.task')
image_dir = os.path.join(BASE_DIR, 'data', 'raw')
heatmap_dir = os.path.join(BASE_DIR, 'data', 'processed', 'Landmarks', 'Heatmaps')
motion_dir = os.path.join(BASE_DIR, 'data', 'processed', 'Landmarks', 'MotionHeatmaps')
motion_npy_dir = os.path.join(motion_dir, 'npys')

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

def normalize_spatial_pose(landmarks, image_width, image_height,
                           nose_idx=1, left_eye_idx=33, right_eye_idx=263):
    """Centers, rotates, and scales a single frame of (N, 2) landmarks.

    MediaPipe returns x normalized by image width and y by image height, so on a
    non-square image the two axes have different pixels-per-unit. We first convert
    to pixel coordinates (x*width, y*height) so x and y share one isotropic scale;
    otherwise the face is distorted (e.g. stretched vertically on 16:9 images).
    """
    # Convert NormalizedLandmark objects to PIXEL (x, y) coords (isotropic scale)
    landmarks = np.array([[lm.x * image_width, lm.y * image_height] for lm in landmarks]) #elongated face fix
    
    # 1. Translation: Center everything around the nose tip
    nose_pos = landmarks[nose_idx]
    centered_landmarks = landmarks - nose_pos

    # 2. Rotation: Level the face by aligning the eyes horizontally
    left_eye = centered_landmarks[left_eye_idx]
    right_eye = centered_landmarks[right_eye_idx]
    
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.arctan2(dy, dx)
    
    c, s = np.cos(-angle), np.sin(-angle)
    rotation_matrix = np.array(((c, -s), (s, c)))
    rotated_landmarks = np.dot(centered_landmarks, rotation_matrix.T)

    # 3. Scale: Normalize so the distance between outer eyes is exactly 1.0
    left_eye_rot = rotated_landmarks[left_eye_idx]
    right_eye_rot = rotated_landmarks[right_eye_idx]
    
    eye_distance = np.linalg.norm(right_eye_rot - left_eye_rot)
    
    # Prevent division by zero
    if eye_distance < 1e-6:
        eye_distance = 1.0 
        
    scaled_landmarks = rotated_landmarks / eye_distance
    

    return scaled_landmarks


def landmarks_to_pixels(normalized_landmarks, size=HEATMAP_SIZE, scale=None):
    """Map pose-normalized (nose-centered) coords into a size x size pixel frame.

    The nose (origin after normalization) is placed at the image center. Because
    every image shares the same canonical scale (inter-eye distance = 1.0), a
    fixed `scale` keeps the face size consistent across the whole dataset.

    Args:
        normalized_landmarks: (N, 2) pose-normalized landmark array.
        scale: pixels per normalized unit. Defaults to the dataset-wide
            HEATMAP_SCALE, which is only computed after the detection pass, so
            it is looked up at call time rather than baked into the signature.

    Returns:
        (N, 2) array of (x, y) pixel coordinates (may fall outside [0, size)).
    """
    center = size / 2.0
    px = normalized_landmarks[:, 0] * scale + center
    py = normalized_landmarks[:, 1] * scale + center
    return np.stack([px, py], axis=1)


def generate_group_heatmaps(normalized_landmarks, groups=KEY_LANDMARKS,
                            size=HEATMAP_SIZE, sigma=HEATMAP_SIGMA,
                            scale=None, group_colors=GROUP_COLORS):
    """Render all facial-feature groups into a single 3-channel RGB heatmap.

    Each group's landmarks are drawn as 2D Gaussian blobs (peaking at 1.0) and
    tinted with the group's RGB color, then composited per channel with a max so
    every blob stays at full intensity regardless of overlap. Groups are kept
    distinguishable by hue instead of by a dedicated channel.

    Args:
        normalized_landmarks: (N, 2) pose-normalized landmark array (full set,
            so group indices like 468/473 resolve correctly).
        groups: ordered dict mapping group name -> list of landmark indices.
        scale: pixels per normalized unit
        group_colors: dict mapping group name -> (r, g, b) color in [0, 1].

    Returns:
        heatmap: (size, size, 3) float32 RGB array in [0, 1].
        group_names: list of the group names (kept for reference / labelling).
    """
    pixels = landmarks_to_pixels(normalized_landmarks, size=size, scale=scale)
    group_names = list(groups.keys())
    heatmap = np.zeros((size, size, 3), dtype=np.float32)
    two_sigma_sq = 2.0 * sigma * sigma

    for name in group_names:
        color = group_colors[name]
        for idx in groups[name]:
            cx, cy = pixels[idx]
            blob = np.exp(-(((hm_x - cx) ** 2 + (hm_y - cy) ** 2) / two_sigma_sq))
            for ch in range(3):
                np.maximum(heatmap[:, :, ch], blob * color[ch], out=heatmap[:, :, ch]) #multiply each channel in the hue by the blob intensity and take the max to avoid overlap issues

    return heatmap, group_names


def landmark_displacements(normalized_landmarks, neutral_landmarks,
                           indices=KEY_LANDMARK_INDICES, scale=None):
    """How far each key landmark moved between the neutral and the deformed frame.

    Both frames are already pose-normalized (nose-centered, eyes leveled,
    inter-eye distance = 1.0), so the rigid head pose is gone and what is left is
    the actual facial deformation. The difference is converted to heatmap pixels
    with the same `scale` used to rasterize the landmarks, so a displacement of
    `n` here means "n pixels on the 256x256 canvas".

    Args:
        normalized_landmarks: (N, 2) pose-normalized landmarks of the deformed frame.
        neutral_landmarks: (N, 2) pose-normalized landmarks of the neutral frame.
        indices: landmark indices to measure (defaults to the key subset).
        scale: pixels per normalized unit (the dataset-wide HEATMAP_SCALE).

    Returns:
        delta: (K, 2) signed (dx, dy) displacement in pixels.
        magnitude: (K,) euclidean distance moved, in pixels.
    """
    delta = (normalized_landmarks[indices] - neutral_landmarks[indices]) * scale
    magnitude = np.linalg.norm(delta, axis=1)
    return delta, magnitude


def compute_motion_scale(delta_list, percentile=MOTION_PERCENTILE):
    """Single dataset-wide divisor that maps the displacements into ~[-1, 1].

    Uses a high percentile of |dx|, |dy| over the whole dataset rather than the
    max so that a handful of outlier frames (or a bad detection) don't crush the
    range every other frame lives in. Values past the percentile are clipped.
    Store this to invert the normalization later.

    Args:
        delta_list: iterable of (K, 2) signed displacement arrays, in pixels.
        percentile: percentile of |displacement| mapped to 1.0.

    Returns:
        float scale (pixels per unit of heatmap value).
    """
    all_abs = np.abs(np.concatenate([d.ravel() for d in delta_list]))
    scale = float(np.percentile(all_abs, percentile))
    if scale < 1e-6:
        return 1.0  # degenerate fallback: no motion anywhere in the dataset
    return scale


def generate_motion_heatmap(normalized_landmarks, neutral_landmarks,
                            indices=KEY_LANDMARK_INDICES, size=HEATMAP_SIZE,
                            sigma=MOTION_SIGMA, scale=None, motion_scale=1.0,
                            anchor=MOTION_ANCHOR):
    """Render per-landmark displacement-from-neutral as a 3-channel heatmap.

    Each key landmark contributes one Gaussian blob whose *value* (not just its
    presence) encodes how that landmark moved relative to the neutral frame:

        channel 0 (R): dx / motion_scale   signed, negative = moved left
        channel 1 (G): dy / motion_scale   signed, negative = moved up
        channel 2 (B): |d| / motion_scale  unsigned distance moved

    The blob is anchored at the landmark's neutral position by default, so every
    frame in the dataset has its blobs in exactly the same places and only the
    values differ -- the map reads as a displacement field sampled on the neutral
    face. Set anchor='deformed' to instead place each blob where the landmark
    ended up.

    Where blobs overlap, the nearest landmark wins (largest Gaussian weight)
    instead of the per-channel max used for the group heatmaps: a max over signed
    values would silently drop negative displacements. The kept value is then
    scaled by that Gaussian weight, so intensity falls off to 0 in the background
    exactly like the landmark heatmaps.

    Args:
        normalized_landmarks: (N, 2) pose-normalized landmarks of this frame.
        neutral_landmarks: (N, 2) pose-normalized landmarks of the neutral frame.
        indices: landmark indices to render.
        size: output resolution (size x size).
        sigma: Gaussian std in pixels.
        scale: pixels per normalized unit (dataset-wide HEATMAP_SCALE).
        motion_scale: divisor from compute_motion_scale(), maps px -> ~[-1, 1].
        anchor: 'neutral' or 'deformed' blob placement.

    Returns:
        (size, size, 3) float32 array; channels 0/1 in [-1, 1], channel 2 in [0, 1].
    """
    if anchor == 'neutral':
        anchor_pixels = landmarks_to_pixels(neutral_landmarks, size=size, scale=scale)[indices]
    elif anchor == 'deformed':
        anchor_pixels = landmarks_to_pixels(normalized_landmarks, size=size, scale=scale)[indices]
    else:
        raise ValueError(f"Unknown anchor: {anchor!r} (expected 'neutral' or 'deformed')")

    delta, magnitude = landmark_displacements(
        normalized_landmarks, neutral_landmarks, indices=indices, scale=scale)

    # (K, 3) value carried by each landmark's blob, normalized to ~[-1, 1].
    values = np.stack([delta[:, 0], delta[:, 1], magnitude], axis=1) / motion_scale

    heatmap = np.zeros((size, size, 3), dtype=np.float32)
    best_weight = np.zeros((size, size), dtype=np.float32)  # nearest-landmark Gaussian weight
    two_sigma_sq = 2.0 * sigma * sigma
    radius = int(np.ceil(BLOB_RADIUS_SIGMAS * sigma))

    for i, (cx, cy) in enumerate(anchor_pixels):
        # Rasterize only the local window; outside ~4 sigma the Gaussian is ~0.
        x0, x1 = max(0, int(cx) - radius), min(size, int(cx) + radius + 1)
        y0, y1 = max(0, int(cy) - radius), min(size, int(cy) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue  # landmark falls outside the canvas

        blob = np.exp(-(((hm_x[y0:y1, x0:x1] - cx) ** 2
                         + (hm_y[y0:y1, x0:x1] - cy) ** 2) / two_sigma_sq))

        win_best = best_weight[y0:y1, x0:x1]     # views into the full-size arrays
        win_heat = heatmap[y0:y1, x0:x1]
        closer = blob > win_best
        win_best[closer] = blob[closer]
        win_heat[closer] = values[i]

    heatmap *= best_weight[:, :, None]  # fade each blob out with its Gaussian
    return np.clip(heatmap, -1.0, 1.0)


def motion_heatmap_preview(heatmap):
    """Turn a motion heatmap into a viewable BGR uint8 image (optical-flow style).

    Hue encodes the direction the landmark moved, value encodes how far, so the
    preview answers "what moved, which way, how much" at a glance. The .npy
    written next to it holds the raw signed values -- this is display only.
    """
    dx, dy, mag = heatmap[:, :, 0], heatmap[:, :, 1], heatmap[:, :, 2]
    hsv = np.zeros(heatmap.shape, dtype=np.uint8)
    hsv[:, :, 0] = ((np.arctan2(dy, dx) + np.pi) * (90.0 / np.pi)).astype(np.uint8)  # 0-180
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = (np.clip(mag, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def save_motion_heatmap(heatmap, image_file, png_dir, npy_dir):
    """Save one motion heatmap: a flow-colored PNG preview and the raw float array.

    Writes <stem>_motion.png into `png_dir` and <stem>_motion.npy -- the
    (H, W, 3) float32 array with channels 0/1 in [-1, 1] and channel 2 in
    [0, 1] -- into `npy_dir`, keeping the raw model input separate from the
    previews the way Heatmaps/ and npys/ already are.
    """
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    stem = os.path.splitext(image_file)[0]

    cv2.imwrite(os.path.join(png_dir, f"{stem}_motion.png"), motion_heatmap_preview(heatmap))
    np.save(os.path.join(npy_dir, f"{stem}_motion.npy"), heatmap.astype(np.float32))


def heatmap_max_projection(heatmap):
    """Collapse a (H, W, C) heatmap to a single (H, W) image (max over channels).

    Useful for quick visualization of all groups at once.
    """
    return heatmap.max(axis=2)


def compute_dataset_scale(normalized_landmarks_list, key_indices=KEY_LANDMARK_INDICES,
                          size=HEATMAP_SIZE, margin=0.9):
    """Pick a single pixels-per-unit scale so the widest face in the dataset fits.

    Measures the largest distance of any key landmark from the center (the nose,
    at the origin) across ALL images, then sizes it to fill `margin` of the
    half-frame. Using one global scale (instead of a per-image one) keeps every
    face the same size across the dataset, preserving the canonical alignment.

    Args:
        normalized_landmarks_list: iterable of (N, 2) pose-normalized arrays.
        key_indices: landmark indices to consider when measuring extent.
        size: heatmap resolution (the frame is size x size).
        margin: fraction of the half-frame to fill (0.9 leaves a 10% border).

    Returns:
        float scale (pixels per normalized unit).
    """
    max_extent = 0.0
    for nl in normalized_landmarks_list:
        extent = np.abs(nl[key_indices]).max()
        max_extent = max(max_extent, extent)

    if max_extent < 1e-6:
        return size / 4.0  # degenerate fallback
    return margin * (size / 2.0) / max_extent


def extract_geometric_features(normalized_landmarks):
    """
    Extract geometric features from normalized landmarks.
    
    Args:
        normalized_landmarks: (N, 2) numpy array of normalized landmark coordinates
        
    Returns:
        dict: Dictionary containing all geometric features
    """
    # Landmark indices for feature extraction
    # Eyes
    RIGHT_EYE_TOP = 159      # Top of right eye
    RIGHT_EYE_BOTTOM = 145   # Bottom of right eye
    RIGHT_EYE_CENTER = 468   # Center of right eye (or use average of corners: 33, 133)
    LEFT_EYE_TOP = 386       # Top of left eye
    LEFT_EYE_BOTTOM = 374    # Bottom of left eye
    LEFT_EYE_CENTER = 473    # Center of left eye (or use average of corners: 362, 263)
    
    # Eyebrows
    RIGHT_EYEBROW_TOP = 66   # Top of right eyebrow (middle point)
    LEFT_EYEBROW_TOP = 296   # Top of left eyebrow (middle point)
    
    # Nose
    NOSE_LEFT = 278          # Leftmost nose point
    NOSE_RIGHT = 48        # Rightmost nose point
    
    # Mouth/Lips
    MOUTH_LEFT = 61          # Left mouth corner
    MOUTH_RIGHT = 291        # Right mouth corner
    UPPER_LIP_TOP = 0        # Top of upper lip (center)
    LOWER_LIP_BOTTOM = 17    # Bottom of lower lip (center)
    
    # Lower eyelid points (for lip corner to eyelid distance)
    RIGHT_LOWER_EYELID = 145  # Lower eyelid of right eye
    LEFT_LOWER_EYELID = 374   # Lower eyelid of left eye
    
    features = {}
    
    # 1. Left/Right Eyebrow Height
    # Vertical distance between top of eyebrow and center of eye 
    features['right_eyebrow_height'] = abs(normalized_landmarks[RIGHT_EYEBROW_TOP][1] - normalized_landmarks[RIGHT_EYE_CENTER][1])
    features['left_eyebrow_height'] = abs(normalized_landmarks[LEFT_EYEBROW_TOP][1] - normalized_landmarks[LEFT_EYE_CENTER][1])
    
    # 2. Left/Right Eyelid Height (Eye Openness)
    # Vertical distance between top and bottom of eye
    features['right_eyelid_height'] = abs(normalized_landmarks[RIGHT_EYE_TOP][1] - normalized_landmarks[RIGHT_EYE_BOTTOM][1])
    features['left_eyelid_height'] = abs(normalized_landmarks[LEFT_EYE_TOP][1] - normalized_landmarks[LEFT_EYE_BOTTOM][1])
    
    # 3. Nose Width
    # Horizontal distance between leftmost and rightmost nose landmarks
    features['nose_width'] = abs(normalized_landmarks[NOSE_LEFT][0] - normalized_landmarks[NOSE_RIGHT][0])
    
    # 4. Mouth Width
    # Distance from left mouth corner to right mouth corner
    mouth_left = normalized_landmarks[MOUTH_LEFT]
    mouth_right = normalized_landmarks[MOUTH_RIGHT]
    features['mouth_width'] = np.linalg.norm(mouth_right - mouth_left)
    
    # 5. Closed Mouth Measure
    # Vertical distance between upper and lower lip
    features['closed_mouth_measure'] = abs(normalized_landmarks[UPPER_LIP_TOP][1] - normalized_landmarks[LOWER_LIP_BOTTOM][1])
    
    # 6. Left/Right Lip Height
    # Vertical distance from lip corner to lower eyelid
    features['right_lip_height'] = abs(normalized_landmarks[MOUTH_RIGHT][1] - normalized_landmarks[RIGHT_LOWER_EYELID][1])
    features['left_lip_height'] = abs(normalized_landmarks[MOUTH_LEFT][1] - normalized_landmarks[LEFT_LOWER_EYELID][1])
    
    return features


def print_geometric_features(features):
    """Pretty print geometric features."""
    print("\n" + "="*50)
    print("GEOMETRIC FEATURES (Normalized)")
    print("="*50)
    print(f"  Right Eyebrow Height:  {features['right_eyebrow_height']:.4f}")
    print(f"  Left Eyebrow Height:   {features['left_eyebrow_height']:.4f}")
    print(f"  Right Eyelid Height:   {features['right_eyelid_height']:.4f}")
    print(f"  Left Eyelid Height:    {features['left_eyelid_height']:.4f}")
    print(f"  Nose Width:            {features['nose_width']:.4f}")
    print(f"  Mouth Width:           {features['mouth_width']:.4f}")
    print(f"  Closed Mouth Measure:  {features['closed_mouth_measure']:.4f}")
    print(f"  Right Lip Height:      {features['right_lip_height']:.4f}")
    print(f"  Left Lip Height:       {features['left_lip_height']:.4f}")
    print("="*50 + "\n")


def plot_normalized_landmarks(normalized_landmarks, ax, key_indices=None, title="Normalized Face Landmarks"):
    """
    Visualize normalized landmarks on a standalone plot.
    
    Args:
        normalized_landmarks: (N, 2) numpy array of normalized landmark coordinates
        key_indices: Optional list of indices to highlight (defaults to KEY_LANDMARK_INDICES)
        title: Plot title
        ax: Matplotlib axis object to plot on
    """
    if key_indices is None:
        key_indices = KEY_LANDMARK_INDICES
    
    # Plot all landmarks in light gray
    ax.scatter(normalized_landmarks[:, 0], normalized_landmarks[:, 1], 
               c='lightgray', s=10, alpha=0.5, label='All landmarks')
    
    # Highlight key landmarks in blue
    key_points = normalized_landmarks[key_indices]
    ax.scatter(key_points[:, 0], key_points[:, 1], 
               c='blue', s=30, label='Key landmarks')
    
    # Draw connections for facial features
    colors = {
        'jaw': 'green',
        'forehead': 'purple', 
        'right_eyebrow': 'orange',
        'left_eyebrow': 'orange',
        'nose_sides': 'red',
        'right_eye': 'cyan',
        'left_eye': 'cyan',
        'outer_lips': 'pink'
    }
    
    for feature, indices in KEY_LANDMARKS.items():
        points = normalized_landmarks[indices]
        ax.plot(points[:, 0], points[:, 1], c=colors.get(feature, 'blue'), 
                linewidth=1.5, alpha=0.7)
    
    # Invert y-axis to match image coordinates (y increases downward)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)

def save_landmarks_data(data_dict, output_path):
    """
    Save landmarks and features to a PyTorch .pt file.
    
    Args:
        data_dict: Dictionary mapping image_name -> {
            'landmarks': (N, 2) numpy array of normalized landmarks,
            'features': dict of geometric features
        }
        output_path: Path to save the .pt file
        
    The saved file structure:
    {
        'image1.jpg': {
            'landmarks': torch.Tensor of shape (N, 2),
            'features': torch.Tensor of shape (num_features,),
            'feature_names': ['right_eyebrow_height', 'left_eyebrow_height', ...]
        },
        'image2.jpg': {...},
        ...
        '_metadata': {
            'num_images': int,
            'num_landmarks': int,
            'feature_names': list of feature names
        }
    }
    """
    # Define consistent feature order
    feature_names = [
        'right_eyebrow_height', 'left_eyebrow_height',
        'right_eyelid_height', 'left_eyelid_height',
        'nose_width', 'mouth_width', 'closed_mouth_measure',
        'right_lip_height', 'left_lip_height'
    ]

    image_names = list(data_dict.keys())
    
    # Step 1: Flatten all landmarks into a single array for Z-score computation
    # Each row is one image's flattened landmarks (N*2,)
    all_landmarks_flat = np.array([
        data_dict[name]['landmarks'].flatten() for name in image_names
    ])

    # Step 2: Compute Z-score statistics across the dataset
    # mean and std for each landmark coordinate (x and y indvidually) across all images
    landmarks_mean = all_landmarks_flat.mean(axis=0)  # (N*2,)
    landmarks_std = all_landmarks_flat.std(axis=0)    # (N*2,)
    
     
    # Prevent division by zero
    landmarks_std[landmarks_std < 1e-8] = 1.0
    
    # Step 3: Apply Z-score standardization
    all_landmarks_standardized = (all_landmarks_flat - landmarks_mean) / landmarks_std

    #build the save dictionary
    save_dict = {}

    for i, image_name in enumerate(image_names):
        # Convert standardized landmarks to tensor
        landmarks_tensor = torch.tensor(all_landmarks_standardized[i], dtype=torch.float32)
        
        # Convert features dict to ordered tensor
        features_tensor = torch.tensor(
            [data_dict[image_name]['features'][name] for name in feature_names],
            dtype=torch.float32
        )
        
        save_dict[image_name] = {
            'landmarks': landmarks_tensor, # (N*2,) flattened & standardized
            'features': features_tensor,
            'feature_names': feature_names
        }
    
     # Add metadata (including Z-score stats for inference-time normalization)
    save_dict['_metadata'] = {
        'num_images': len(data_dict),
        'num_landmarks': data_dict[next(iter(data_dict))]['landmarks'].shape[0] if data_dict else 0,
        'feature_names': feature_names,
        'zscore_mean': torch.tensor(landmarks_mean, dtype=torch.float32),
        'zscore_std': torch.tensor(landmarks_std, dtype=torch.float32)
    }
    
    torch.save(save_dict, output_path)
    print(f"Saved landmark data for {len(data_dict)} images to {output_path}")
    print(f"Landmarks shape per image: ({len(landmarks_mean)},) [flattened & Z-score standardized]")


def save_heatmap(heatmap, group_names, image_file, output_dir):
    """Save an RGB heatmap for one image to `output_dir`.

    Writes two files (sharing the source image's stem):
      - <stem>_heatmap.png : the viewable RGB heatmap.
      - <stem>_heatmap.npy : the (H, W, 3) float32 RGB array in [0, 1], for
        downstream model input.

    Args:
        heatmap: (H, W, 3) float32 RGB heatmap in [0, 1].
        group_names: list of group labels (kept for reference / future use).
        image_file: source image filename (used to derive the output stem).
        output_dir: directory to write the heatmap files into.
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(image_file)[0]

    # Viewable image: scale RGB to 0-255 (OpenCV writes BGR, so reverse channels)
    rgb_uint8 = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"{stem}_heatmap.png"), rgb_uint8[:, :, ::-1])

    # Full RGB heatmap for model input
    np.save(os.path.join(output_dir, f"{stem}_heatmap.npy"), heatmap.astype(np.float32))


all_landmarks_data = {}  # Dictionary to hold landmarks and features for all images
per_image = {}           # Per-image info needed for the heatmap/visualization pass



# --- Pass 1: detect landmarks for every image and collect what we need ---
for image_file in os.listdir(image_dir):
    if image_file.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(image_dir, image_file)

        # Load image using mediapipe
        image = mp.Image.create_from_file(image_path)

        # Detect landmarks
        detection_result = detector.detect(image)

        if detection_result.face_landmarks:
            normalized_landmarks = normalize_spatial_pose(
                detection_result.face_landmarks[0], image.width, image.height)
            normalized_key_landmarks = normalized_landmarks[KEY_LANDMARK_INDICES]

            # Extract and print geometric features
            features = extract_geometric_features(normalized_landmarks)
            #print_geometric_features(features)

            # Store landmarks and features for this image (for the .pt save)
            all_landmarks_data[image_file] = {
                'landmarks': normalized_key_landmarks,
                'features': features
            }

            # Raw (image-aligned) key-landmark coords in [0, 1], for annotation redraw
            raw_key_coords = np.array([
                [detection_result.face_landmarks[0][idx].x,
                 detection_result.face_landmarks[0][idx].y]
                for idx in KEY_LANDMARK_INDICES
            ])

            per_image[image_file] = {
                'image_path': image_path,
                'normalized_landmarks': normalized_landmarks,
                'raw_key_coords': raw_key_coords
            }

# --- Compute one global scale so faces stay the same size across the dataset ---
HEATMAP_SCALE = compute_dataset_scale(
    [info['normalized_landmarks'] for info in per_image.values()] #input array of (n, 2) normalized landmarks
)
print(f"Computed dataset HEATMAP_SCALE = {HEATMAP_SCALE:.2f} px/unit "
      f"(over {len(per_image)} images)")

# --- The neutral (rest-pose) frame every other frame is measured against ---
if NEUTRAL_IMAGE not in per_image:
    raise RuntimeError(
        f"Neutral reference {NEUTRAL_IMAGE} has no detected landmarks in {image_dir}; "
        "displacements cannot be computed without it."
    )
neutral_landmarks = per_image[NEUTRAL_IMAGE]['normalized_landmarks']

# Every frame except the neutral one itself: its displacement is all zeros by
# construction, which would only teach a model to output an empty map.
deformed_files = [f for f in sorted(per_image) if f != NEUTRAL_IMAGE]

# --- Pass 1.5: one dataset-wide divisor so the values land in ~[-1, 1] ---
all_deltas = [
    landmark_displacements(per_image[f]['normalized_landmarks'], neutral_landmarks,
                           scale=HEATMAP_SCALE)[0]
    for f in deformed_files
]
MOTION_SCALE = compute_motion_scale(all_deltas)
all_mags = np.linalg.norm(np.concatenate(all_deltas), axis=1)
print(f"Computed dataset MOTION_SCALE = {MOTION_SCALE:.2f} px "
      f"(p{MOTION_PERCENTILE} of |displacement| over {len(deformed_files)} frames)")
print(f"Displacement magnitude: mean {all_mags.mean():.2f} px, "
      f"median {np.median(all_mags):.2f} px, max {all_mags.max():.2f} px")

# --- Pass 2: render each frame's displacement-from-neutral as a heatmap ---
for i, image_file in enumerate(deformed_files):
    motion_heatmap = generate_motion_heatmap(
        per_image[image_file]['normalized_landmarks'], neutral_landmarks,
        scale=HEATMAP_SCALE, motion_scale=MOTION_SCALE)
    save_motion_heatmap(motion_heatmap, image_file, motion_dir, motion_npy_dir)

    if i % 300 == 0:
        print(f"Processed {i}/{len(deformed_files)} images: {image_file}")

# Record what the values mean, so the normalization can be inverted later.
np.save(os.path.join(motion_dir, 'motion_metadata.npy'), {
    'neutral_image': NEUTRAL_IMAGE,
    'heatmap_scale': HEATMAP_SCALE,   # px per normalized unit (inter-eye distance = 1)
    'motion_scale': MOTION_SCALE,     # px per unit of heatmap value
    'motion_sigma': MOTION_SIGMA,
    'anchor': MOTION_ANCHOR,
    'channels': ['dx', 'dy', 'magnitude'],
    'key_landmark_indices': KEY_LANDMARK_INDICES,
    'num_frames': len(deformed_files),
}, allow_pickle=True)
print(f"Wrote {len(deformed_files)} motion heatmaps to {motion_dir}")

    # Reload the image and redraw key landmarks for the annotated view
    # image = mp.Image.create_from_file(info['image_path'])
    # annotated_image = np.copy(image.numpy_view())
    # h, w = annotated_image.shape[:2]
    # for x_norm, y_norm in info['raw_key_coords']:
    #     cv2.circle(annotated_image, (int(x_norm * w), int(y_norm * h)), 3, (255, 0, 0), -1)

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 10), gridspec_kw={'width_ratios': [1, 2, 1]})

    # # Left: Annotated image with landmarks
    # ax1.imshow(annotated_image)
    # ax1.set_title(f"Annotated Image: {image_file}")
    # ax1.axis("off")

    # # Middle: Normalized landmarks plot
    # plot_normalized_landmarks(normalized_landmarks, ax2, title=f"Normalized Landmarks: {image_file}")
    # #plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0], ax2)

    # # Right: RGB heatmap (groups color-coded into 3 channels)
    # ax3.imshow(np.clip(heatmap, 0.0, 1.0))
    # ax3.set_title(f"Landmark Heatmap ({HEATMAP_SIZE}x{HEATMAP_SIZE}x3)")
    # ax3.axis("off")

    # plt.tight_layout()
    # plt.show()

# Save all collected landmarks data to a .pt file
# if all_landmarks_data:
#     output_path = os.path.join(BASE_DIR, 'data', 'processed', 'Landmarks', 'landmarks_data.pt')
#     save_landmarks_data(all_landmarks_data, output_path)
