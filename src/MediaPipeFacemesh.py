"""Run the MediaPipe Face Mesh model over every image in data/Train and data/Test.

The Face Landmarker task outputs 478 3-dimensional landmarks per face, of which
the last 10 (indices 468-477) are the iris refinement points. Those are dropped
here, keeping only the 468-point face mesh.

Coordinates as returned by MediaPipe:
    x, y : normalized to [0, 1] by image width / height respectively.
    z    : depth relative to the head center, using roughly the same scale as
           x (i.e. normalized by image width); smaller = closer to the camera.

Each split is read from data/<Split>/Images and written to data/<Split>/FaceMeshes:
    <stem>.npy               per-image (256, 256, 4) float32 deformation map: the
                             pose-normalized offset from the neutral frame,
                             rasterized over the face mesh, channels
                             (dx, dy, dz, mask), the offsets signed and unscaled
                             and mask 1.0 on face pixels / 0.0 elsewhere.
                             Sits directly in FaceMeshes/ so it pairs by stem
                             with data/<Split>/DeformationMaps/<stem>.exr
    facemesh_468.pt          the raw (468, 3) normalized landmarks for every
                             image of the split in one torch dict, see
                             save_facemesh() for the layout
    previews/<stem>_mesh.png the mesh drawn over the source image

Both splits share one neutral frame (see NEUTRAL_SPLIT / NEUTRAL_IMAGE), so the
zero the deltas are measured against and the canvas layout they are rasterized
into are identical everywhere -- otherwise a model trained on Train would be
evaluated on Test maps written in a different coordinate convention.

Usage:
    python src/MediaPipeFacemesh.py               # full pass over both splits
    python src/MediaPipeFacemesh.py --preview 10  # only draw the first 10 overlays
    python src/MediaPipeFacemesh.py --split Test  # restrict to one split
"""

import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
import torch
import numpy as np

# Landmarks kept per face. The model returns 478; indices 468..477 are the iris
# refinement points appended after the mesh, and are truncated away.
NUM_FACEMESH_LANDMARKS = 468

# Rigid landmarks used to cancel head pose. These sit on bone (brow ridge, eye
# corners, nose) rather than on skin an expression can pull, so the frame they
# define tracks the head and not the face.
NOSE_TIP = 1
LEFT_EYE_OUTER = 33     # outer corner of the eye on the image-left side
RIGHT_EYE_OUTER = 263   # outer corner of the eye on the image-right side
NASION = 168            # bridge of the nose, between the brows

# The nose tip -> nasion reference used to build the head frame points up *and
# back*, because the nasion is set deep between the brows. Left uncorrected that
# tilts the whole canonical frame back by a constant ~19.7 deg (measured over the
# dataset: std 1.3 deg, so it really is a fixed offset and not per-frame noise),
# which shows up as depth increasing steadily from forehead to chin. Composing
# this constant pitch into the frame cancels it; being constant, it cannot affect
# pose invariance -- it just makes the canonical view anatomically frontal.
# Solved so the mesh-wide slope of z against y comes out at 0.0 deg.
CANONICAL_PITCH_DEG = 26.37

# How many of the first images get a mesh overlay written during a full pass
NUM_PREVIEWS = 10

# Overlay styling, all BGR (OpenCV order)
TESSELATION_COLOR = (120, 120, 120)  # gray wireframe of the full mesh
CONTOUR_COLOR = (0, 255, 0)          # green outlines: eyes, brows, lips, oval, nose
POINT_COLOR = (0, 200, 255)          # amber landmark dots

# --- Canonical canvas the pose-normalized mesh is rendered into ---
# After normalize_spatial_pose() the nose tip is at the origin and the inter-eye
# distance is 1.0, so these constants are dataset-independent: the same
# normalized coordinate always lands on the same canvas pixel, in every frame.
NORM_CANVAS_SIZE = 256
NORM_CANVAS_SCALE = 110.0   # px per normalized unit; the mesh reaches ~1.05 units
                            # from the nose, so 1.05*110 = 116 < 128 half-frame
NORM_CANVAS_Y_OFFSET = 0.06 # the nose tip sits just below the mesh's vertical
                            # middle (y spans -1.05 to +0.93), so shift down a
                            # little to center the face rather than the nose
# Depth range mapped across the colormap. Fixed, not per-frame, so the same
# color means the same depth in every rendered frame. z runs from ~0 at the nose
# tip, the nearest point, to ~1.1 at the cheeks as they wrap toward the ears.
NORM_DEPTH_RANGE = (0.0, 1.15)
NORM_BACKGROUND = (16, 16, 16)       # near-black canvas
# Delta magnitude mapped to the ends of a delta-map preview, in normalized units
# (inter-eye distance = 1.0). Fixed, so the same gray means the same offset in
# every preview; the deltas this dataset produces stay well inside +/-0.1.
DELTA_PREVIEW_RANGE = 0.1
NORM_CONTOUR_COLOR = (255, 255, 255) # white feature outlines over the depth shading

# Turbo lookup table, built once: 256 BGR entries indexed by quantized depth
DEPTH_LUT = cv2.applyColorMap(
    np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO
).reshape(256, 3)

def _tessellation_triangles(connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION):
    """Recover the canonical face mesh's triangle list from MediaPipe's edge list.

    FACE_LANDMARKS_TESSELATION is published as 2556 directed connections rather
    than as faces, but they are not in arbitrary order: they come in consecutive
    triples, each triple being the three edges of one triangle walked in order
    (a->b, b->c, c->a). Taking every third connection's start therefore recovers
    the original 852 faces exactly, with no guesswork -- unlike enumerating
    3-cycles of the edge graph, which also picks up cycles that are not faces.

    The closure of each triple is asserted, so a future MediaPipe release that
    reorders the list fails loudly here instead of silently producing a mesh
    with wrong or missing faces.

    Returns:
        (852, 3) int32 array of landmark indices, one row per triangle.
    """
    triangles = []
    for i in range(0, len(connections) - 2, 3):
        a, b, c = connections[i], connections[i + 1], connections[i + 2]
        if not (a.end == b.start and b.end == c.start and c.end == a.start):
            raise RuntimeError(
                "FACE_LANDMARKS_TESSELATION is no longer grouped into closing "
                f"edge triples at connection {i}; the triangle recovery above "
                "needs revisiting for this MediaPipe version.")
        triangles.append((a.start, b.start, c.start))
    return np.array(triangles, dtype=np.int32)


# The face mesh's own topology, so the rasterized surface is the one the model
# actually defines: it has genuine holes at the eye apertures and the mouth
# opening, where there is no skin. (A Delaunay triangulation of the projected
# points cannot know that, and bridges straight across them -- which blends
# upper- and lower-lip offsets together on any open-mouth frame.)
FACEMESH_TRIANGLES = _tessellation_triangles()

# Setup paths (relative to this file's location)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'face_landmarker_v2_with_blendshapes.task')
data_dir = os.path.join(BASE_DIR, 'data', 'Omda')

# The dataset is split on disk, each split holding its own Images/ alongside the
# FaceMeshes/ this script fills and the DeformationMaps/ it pairs with.
SPLITS = ('Train', 'Test')

# The single neutral frame every delta in every split is measured against. Frame
# 00000 is present in both splits for exactly this reason (its two
# DeformationMaps/.exr are byte-identical), so one split's copy can define the
# zero for both; the two JPEGs differ slightly in encoding, which is why the
# neutral is pinned to one file rather than taken per split.
NEUTRAL_SPLIT = 'Train'
NEUTRAL_IMAGE = '00000.jpg'


def split_paths(split):
    """Input and output paths for one split, as a dict.

    'facemesh' is the split's FaceMeshes root and holds the delta maps directly,
    one <stem>.npy per image, so they pair by stem with DeformationMaps/<stem>.exr
    and are found by the training loader's non-recursive *.npy glob. The
    auxiliary outputs sit in subdirectories of it, out of that glob's way.
    """
    split_dir = os.path.join(data_dir, split)
    facemesh_dir = os.path.join(split_dir, 'FaceMeshes')
    return {
        'images': os.path.join(split_dir, 'Images'),
        'facemesh': facemesh_dir,
        'previews': os.path.join(facemesh_dir, 'previews'),
        'bundle': os.path.join(facemesh_dir, 'facemesh_468.pt'),
    }

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)


def landmarks_to_array(face_landmarks, num_landmarks=NUM_FACEMESH_LANDMARKS):
    """Convert MediaPipe's NormalizedLandmark list into an (N, 3) float32 array.

    np.array() on the list alone yields an object array of NormalizedLandmark
    objects, so the fields have to be pulled out explicitly. The list is
    truncated to `num_landmarks`, which drops the trailing iris points; both the
    tessellation and contour connection sets only reference indices < 468, so
    nothing downstream needs them.

    Args:
        face_landmarks: list of NormalizedLandmark for one detected face.
        num_landmarks: how many leading landmarks to keep.

    Returns:
        (num_landmarks, 3) float32 array of (x, y, z) in MediaPipe's normalized space.
    """
    return np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks[:num_landmarks]],
                    dtype=np.float32)


def landmarks_to_pixels(landmarks, image_width, image_height):
    """Convert normalized mesh coords to pixels on the source image.

    x and y are normalized by different denominators (width vs height), so on a
    non-square image they must be scaled separately or the face comes out
    stretched. z shares x's scale, so it is scaled by the width too.

    Args:
        landmarks: (N, 3) array of normalized (x, y, z).
        image_width, image_height: source image size in pixels.

    Returns:
        (N, 3) float32 array of (x, y, z) in pixel units.
    """
    scale = np.array([image_width, image_height, image_width], dtype=np.float32)
    return landmarks * scale


def _draw_connections(canvas, points, connections, color, thickness=1):
    """Stroke one MediaPipe connection set (list of start/end index pairs)."""
    for connection in connections:
        start, end = points[connection.start], points[connection.end]
        cv2.line(canvas, (int(start[0]), int(start[1])),
                 (int(end[0]), int(end[1])), color, thickness, cv2.LINE_AA)


def draw_facemesh(image_bgr, landmarks, draw_tesselation=True, draw_points=True):
    """Return a copy of `image_bgr` with the 468-point face mesh drawn on it.

    Layers are drawn back to front: the gray tessellation wireframe first, then
    the feature contours, then the landmark dots, so the detail stays readable
    on top of the dense wireframe. The iris connections are skipped along with
    the iris landmarks themselves.

    Args:
        image_bgr: (H, W, 3) uint8 BGR source image.
        landmarks: (N, 3) array of normalized mesh coords.
        draw_tesselation: include the full ~2.5k-edge wireframe.
        draw_points: mark each individual landmark.

    Returns:
        Annotated (H, W, 3) uint8 BGR copy.
    """
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    # Integer pixel coords once, reused by every layer below
    points = np.column_stack([
        np.round(landmarks[:, 0] * w).astype(int),
        np.round(landmarks[:, 1] * h).astype(int),
    ])

    C = FaceLandmarksConnections
    if draw_tesselation:
        _draw_connections(annotated, points, C.FACE_LANDMARKS_TESSELATION, TESSELATION_COLOR)
    _draw_connections(annotated, points, C.FACE_LANDMARKS_CONTOURS, CONTOUR_COLOR)

    if draw_points:
        for x, y in points:
            cv2.circle(annotated, (int(x), int(y)), 1, POINT_COLOR, -1, cv2.LINE_AA)

    return annotated


def detect_facemesh(detector, image_path):
    """Detect the full face mesh for one image.

    Returns:
        (landmarks, image_bgr) where landmarks is an (N, 3) float32 array of
        normalized coords, or (None, image_bgr) if no face was detected.
    """
    image = mp.Image.create_from_file(image_path)
    # numpy_view() is RGB and read-only; convert to BGR so OpenCV can draw/write it
    image_bgr = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)

    detection_result = detector.detect(image)
    if not detection_result.face_landmarks:
        return None, image_bgr

    return landmarks_to_array(detection_result.face_landmarks[0]), image_bgr


def save_facemesh(landmarks_by_image, sizes_by_image, output_path):
    """Bundle every image's mesh into one .pt file.

    Saved structure:
    {
        'image_names': ['00000.jpg', ...]                  # row order of the tensors
        'landmarks':   torch.Tensor (M, 468, 3)            # normalized coords
        'image_sizes': torch.Tensor (M, 2)  int32          # (width, height) per image
        '_metadata': {
            'num_images': int,
            'num_landmarks': int,
            'coordinate_space': str,
            'iris_landmarks': False,   # the 10 iris points are dropped
        }
    }

    Args:
        landmarks_by_image: dict image_name -> (468, 3) float32 array.
        sizes_by_image: dict image_name -> (width, height).
        output_path: destination .pt path.
    """
    image_names = sorted(landmarks_by_image)
    landmarks = np.stack([landmarks_by_image[name] for name in image_names])
    sizes = np.array([sizes_by_image[name] for name in image_names], dtype=np.int32)

    torch.save({
        'image_names': image_names,
        'landmarks': torch.tensor(landmarks, dtype=torch.float32),
        'image_sizes': torch.tensor(sizes, dtype=torch.int32),
        '_metadata': {
            'num_images': len(image_names),
            'num_landmarks': landmarks.shape[1],
            'coordinate_space': 'mediapipe normalized (x/width, y/height, z~/width)',
            'iris_landmarks': False,
        },
    }, output_path)

    print(f"Saved face mesh for {len(image_names)} images to {output_path}")
    print(f"Landmarks tensor shape: {tuple(landmarks.shape)}")


def list_images(directory):
    """Sorted list of image filenames in `directory`."""
    return sorted(
        f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )


def preview(limit=NUM_PREVIEWS, split=NEUTRAL_SPLIT):
    """Write two previews per image for the first `limit` images of `split`.

    <stem>_mesh.png       the mesh drawn over the source image, at full size
    <stem>_normalized.png the pose-normalized mesh in its canonical 256x256 frame

    A quick visual check that the mesh lands on the face, and that the
    normalization really has cancelled head pose, before committing to a full
    pass over the dataset.
    """
    paths = split_paths(split)
    preview_dir = paths['previews']
    image_dir = paths['images']

    os.makedirs(preview_dir, exist_ok=True)
    image_files = list_images(image_dir)[:limit]
    detector = vision.FaceLandmarker.create_from_options(options)

    for image_file in image_files:
        landmarks, image_bgr = detect_facemesh(
            detector, os.path.join(image_dir, image_file))

        if landmarks is None:
            print(f"{image_file}: no face detected")
            continue

        h, w = image_bgr.shape[:2]
        normalized = normalize_spatial_pose(landmarks, w, h)
        yaw, pitch, roll = head_pose_angles(
            head_frame(landmarks_to_pixels(landmarks.astype(np.float64), w, h)))

        stem = os.path.splitext(image_file)[0]
        cv2.imwrite(os.path.join(preview_dir, f"{stem}_mesh.png"),
                    draw_facemesh(image_bgr, landmarks))
        cv2.imwrite(os.path.join(preview_dir, f"{stem}_normalized.png"),
                    render_normalized_mesh(normalized))

        print(f"{split}/{image_file}: {landmarks.shape[0]} landmarks, "
              f"head pose yaw {yaw:+6.1f} pitch {pitch:+6.1f} roll {roll:+6.1f} "
              f"-> {stem}_mesh.png, {stem}_normalized.png")

    print(f"Wrote {2 * len(image_files)} previews to {preview_dir}")

def _unit(vector):
    """Normalize a vector to unit length, leaving a degenerate one untouched."""
    norm = np.linalg.norm(vector)
    return vector if norm < 1e-8 else vector / norm


def head_frame(landmarks, nose_idx=NOSE_TIP, left_eye_idx=LEFT_EYE_OUTER,
               right_eye_idx=RIGHT_EYE_OUTER, nasion_idx=NASION,
               pitch_deg=CANONICAL_PITCH_DEG):
    """Build the 3x3 rotation matrix describing this head's orientation.

    The three axes are derived from landmarks that sit on bone rather than on
    skin an expression can pull, so the frame follows head pose and not what the
    face is doing:

        x (row 0): eye line, pointing toward `right_eye_idx`
        y (row 1): nose tip -> nasion, flipped to point down, then made
                   perpendicular to x by Gram-Schmidt
        z (row 2): x cross y, i.e. the face-plane normal pointing back into
                   the image, away from the camera

    The rows are an orthonormal basis expressed in image coordinates, so
    `points @ R.T` re-expresses points in the head's own frame -- which is
    exactly what cancels roll, pitch and yaw at once. A 2D eye-leveling rotation
    can only cancel roll; pitch (nodding) and yaw (turning) survive it and leak
    into the landmark values as if they were expression changes.

    Args:
        landmarks: (N, 3) array in an isotropic space (see landmarks_to_pixels).

    Returns:
        (3, 3) float64 rotation matrix whose rows are the head's x, y, z axes.
    """
    x_axis = _unit(landmarks[right_eye_idx] - landmarks[left_eye_idx])

    # Nose tip -> nasion points up the bridge; negate for a downward reference,
    # then strip its x component so it is perpendicular to the eye line.
    down = landmarks[nose_idx] - landmarks[nasion_idx]
    y_axis = _unit(down - np.dot(down, x_axis) * x_axis)

    z_axis = np.cross(x_axis, y_axis)

    # Tip the y/z pair back around x by a constant, cancelling the lean the
    # deep-set nasion puts into the reference (see CANONICAL_PITCH_DEG).
    c, s = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    y_axis, z_axis = c * y_axis + s * z_axis, -s * y_axis + c * z_axis

    return np.stack([x_axis, y_axis, z_axis]).astype(np.float64)


def head_pose_angles(rotation):
    """Yaw / pitch / roll in degrees for a head frame, for logging and checks.

    Roll is rotation in the image plane, pitch is nodding, yaw is turning left
    or right. After normalize_spatial_pose() all three should come out ~0.
    """
    x_axis, _, z_axis = rotation
    roll = np.degrees(np.arctan2(x_axis[1], x_axis[0]))
    pitch = np.degrees(np.arcsin(np.clip(-z_axis[1], -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(z_axis[0], z_axis[2]))
    return yaw, pitch, roll


def build_delta_raster_map(neutral_landmarks, size=NORM_CANVAS_SIZE,
                           scale=NORM_CANVAS_SCALE, y_offset=NORM_CANVAS_Y_OFFSET):
    """Precompute the pixel -> (triangle, barycentric weights) mapping, once.

    The canvas layout is fixed by the neutral frame alone: pose normalization
    puts every frame in the same canonical position and scale, so the neutral's
    projected triangulation is a valid rasterization layout for every other
    frame. That means the expensive half of the bake -- walking 852 triangles
    and solving barycentric coordinates per pixel -- happens once here and is
    reused by rasterize_deltas() for the whole dataset.

    Triangles come from FACEMESH_TRIANGLES, the model's own topology, so the
    eye apertures and the mouth opening stay uncovered rather than being
    bridged by triangles that do not correspond to any skin.

    Where the mesh folds over itself in projection (the silhouette wrapping
    toward the ears, the nostrils) several triangles claim the same pixel. The
    nearest one wins, by z-buffer on the neutral's interpolated depth, which is
    the same painter's ordering render_normalized_mesh() uses.

    Args:
        neutral_landmarks: (N, 3) pose-normalized neutral mesh.
        size: canvas resolution (size x size).
        scale: pixels per normalized unit.
        y_offset: normalized units to shift down, centering the face in frame.

    Returns:
        dict with
            'size':     canvas resolution,
            'pixels':   (K,) int32 flat indices of the covered pixels, row-major,
            'vertices': (K, 3) int32 landmark indices of each pixel's triangle,
            'weights':  (K, 3) float32 barycentric weights, summing to 1,
            'mask':     (size, size) float32, 1.0 on face pixels, 0.0 elsewhere.
    """
    points_2d = normalized_to_canvas(neutral_landmarks, size=size, scale=scale,
                                     y_offset=y_offset)
    depth = np.asarray(neutral_landmarks, dtype=np.float64)[:, 2]

    triangle_of = np.full(size * size, -1, dtype=np.int32)
    weights = np.zeros((size * size, 3), dtype=np.float32)
    zbuffer = np.full(size * size, np.inf)

    for t, tri in enumerate(FACEMESH_TRIANGLES):
        p0, p1, p2 = points_2d[tri]

        # Barycentric denominator; zero means the triangle is edge-on and has
        # no area to fill in this projection.
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(denom) < 1e-12:
            continue

        # Only the triangle's bounding box can contain it, so that is all we test.
        x0 = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
        x1 = min(int(np.ceil(max(p0[0], p1[0], p2[0]))), size - 1)
        y0 = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
        y1 = min(int(np.ceil(max(p0[1], p1[1], p2[1]))), size - 1)
        if x1 < x0 or y1 < y0:
            continue

        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        w0 = ((p1[1] - p2[1]) * (gx - p2[0]) + (p2[0] - p1[0]) * (gy - p2[1])) / denom
        w1 = ((p2[1] - p0[1]) * (gx - p2[0]) + (p0[0] - p2[0]) * (gy - p2[1])) / denom
        w2 = 1.0 - w0 - w1

        # A small negative tolerance keeps pixels exactly on a shared edge from
        # falling through the crack between two triangles.
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue

        flat = (gy * size + gx)[inside]
        bary = np.stack([w0[inside], w1[inside], w2[inside]], axis=1)
        z = bary @ depth[tri]

        nearer = z < zbuffer[flat]
        flat, bary, z = flat[nearer], bary[nearer], z[nearer]
        triangle_of[flat] = t
        weights[flat] = bary
        zbuffer[flat] = z

    covered = triangle_of >= 0
    return {
        'size': size,
        'pixels': np.flatnonzero(covered).astype(np.int32),
        'vertices': FACEMESH_TRIANGLES[triangle_of[covered]],
        'weights': weights[covered],
        'mask': covered.reshape(size, size).astype(np.float32),
    }


def rasterize_deltas(delta_landmarks, raster_map):
    """Rasterize 468 per-landmark deltas into a dense (size, size, 4) float32 map.

    Each covered pixel gets the barycentric blend of its triangle's three
    landmark deltas -- the same interpolation a UV-space deformation bake does,
    but in the canonical canvas rather than a UV atlas.

    Channel 3 is the coverage mask, which is what keeps the holes readable: an
    eye aperture, the mouth opening and the background are all zero in the
    offset channels, and only the mask distinguishes "no skin here" from "skin
    that did not move".

    The offsets are signed and stored as raw float32: dx, dy and dz keep their
    sign and their normalized-unit magnitude, so a delta of -0.03 survives the
    trip to disk intact. Nothing here rescales, offsets or clips into an image
    range; any normalization for training is a separate, explicit step.

    Args:
        delta_landmarks: (N, 3) deformed-minus-neutral offsets, in the
            pose-normalized frame (inter-eye distance = 1.0).
        raster_map: output of build_delta_raster_map(), built from the neutral
            that `delta_landmarks` is measured against.

    Returns:
        (size, size, 4) float32 array, channels (dx, dy, dz, mask).
    """
    deltas = np.asarray(delta_landmarks, dtype=np.float32)
    size = raster_map['size']

    flat = np.zeros((size * size, 4), dtype=np.float32)
    corners = deltas[raster_map['vertices']]        # (K, 3 corners, 3 axes)
    flat[raster_map['pixels'], :3] = np.einsum('kv,kvc->kc', raster_map['weights'], corners)
    flat[raster_map['pixels'], 3] = 1.0
    return flat.reshape(size, size, 4)


def render_delta_map(delta_map, value_range=DELTA_PREVIEW_RANGE):
    """Turn a signed delta map into a uint8 BGR preview for eyeballing.

    Mid-gray is zero, so the sign is visible: a channel darker than gray is a
    negative offset, brighter is positive. Everything the mask marks as
    off-face -- the background, the eye apertures, the mouth opening -- is
    painted black instead, so the holes are visibly holes and not mistaken for
    skin that happened not to move.

    Args:
        delta_map: (size, size, 4) float32 map of (dx, dy, dz, mask). A
            3-channel map is accepted too and treated as fully covered.
        value_range: delta magnitude mapped to full black / full white.

    Returns:
        (size, size, 3) uint8 BGR image.
    """
    delta_map = np.asarray(delta_map)
    t = np.clip(delta_map[:, :, :3] / value_range, -1.0, 1.0)
    rgb = np.round((t * 0.5 + 0.5) * 255).astype(np.uint8)
    if delta_map.shape[2] > 3:
        rgb[delta_map[:, :, 3] <= 0] = 0
    return rgb[:, :, ::-1]  # (dx, dy, dz) as RGB -> BGR for cv2.imwrite


def normalize_spatial_pose(landmarks, image_width, image_height,
                           nose_idx=NOSE_TIP, left_eye_idx=LEFT_EYE_OUTER,
                           right_eye_idx=RIGHT_EYE_OUTER, nasion_idx=NASION,
                           return_transform=False):
    """Remove rigid head pose from one frame of (N, 3) landmarks.

    Cancels all six rigid degrees of freedom, so what survives is facial
    deformation only:

      1. Isotropic units. MediaPipe normalizes x by image width and y by image
         height, so on a non-square image the axes have different pixels-per-unit
         and the face comes out stretched. Converting to pixels (x*width,
         y*height, z*width -- z shares x's scale) puts all three on one scale.
      2. Translation (head movement around the frame): the nose tip becomes the
         origin.
      3. Rotation (head tilts): the landmarks are re-expressed in the head's own
         frame, which cancels roll, pitch and yaw together. See head_frame().
      4. Scale (moving toward or away from the camera): the inter-eye distance
         becomes exactly 1.0.

    Caveat worth knowing: MediaPipe's z is a weak-perspective estimate and is
    noticeably noisier than x and y, so pitch and yaw are cancelled less
    precisely than roll. Frames near-frontal to begin with normalize best.

    Args:
        landmarks: (N, 3) array of MediaPipe-normalized (x, y, z) coords.
        image_width, image_height: source image size, for the isotropic step.
        nose_idx, left_eye_idx, right_eye_idx, nasion_idx: the rigid landmarks
            defining the origin and the head frame.
        return_transform: also return the (rotation, translation, scale) that
            were removed, e.g. to log head pose or to map points back.

    Returns:
        (N, 3) float64 array of pose-normalized landmarks, or
        (landmarks, (rotation, translation, scale)) if return_transform.
    """
    landmarks = np.asarray(landmarks, dtype=np.float64)
    if landmarks.ndim != 2 or landmarks.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) landmarks, got {landmarks.shape}")

    # 1. Isotropic pixel coords, so x/y/z share one unit (elongated-face fix)
    pixels = landmarks_to_pixels(landmarks, image_width, image_height)

    # 2. Translation: the nose tip becomes the origin
    translation = pixels[nose_idx].copy()
    centered = pixels - translation

    # 3. Rotation: re-express in the head's own frame (kills roll+pitch+yaw)
    rotation = head_frame(centered, nose_idx=nose_idx, left_eye_idx=left_eye_idx,
                          right_eye_idx=right_eye_idx, nasion_idx=nasion_idx)
    rotated = centered @ rotation.T # Matrix multiplication: new_coords = old_coords @ R.T

    # 4. Scale: inter-eye distance becomes exactly 1.0
    scale = float(np.linalg.norm(rotated[right_eye_idx] - rotated[left_eye_idx]))
    if scale < 1e-8:
        scale = 1.0  # degenerate detection; leave the frame unscaled

    normalized = rotated / scale

    if return_transform:
        return normalized, (rotation, translation, scale)
    return normalized


def normalized_to_canvas(normalized_landmarks, size=NORM_CANVAS_SIZE,
                         scale=NORM_CANVAS_SCALE, y_offset=NORM_CANVAS_Y_OFFSET):
    """Map pose-normalized coords onto a size x size canvas.

    The mapping is a fixed affine transform, not a per-frame fit: because every
    normalized frame already shares one canonical position and scale, the same
    normalized coordinate lands on the same pixel in every frame, so a stack of
    renders is directly comparable and only expression moves between them.

    Args:
        normalized_landmarks: (N, 3) output of normalize_spatial_pose().
        size: canvas resolution.
        scale: pixels per normalized unit.
        y_offset: normalized units to shift down, centering the face in frame.

    Returns:
        (N, 2) float array of (x, y) canvas pixel coordinates.
    """
    center = size / 2.0
    px = normalized_landmarks[:, 0] * scale + center
    py = (normalized_landmarks[:, 1] + y_offset) * scale + center
    return np.stack([px, py], axis=1)


def depth_colors(depth, depth_range=NORM_DEPTH_RANGE):
    """Map depth values to BGR colors through the turbo colormap.

    Args:
        depth: (N,) array of z values in normalized units.
        depth_range: (near, far) z mapped to the two ends of the colormap.

    Returns:
        (N, 3) uint8 BGR array. Cool colors are near the camera, warm are far.
    """
    near, far = depth_range
    t = np.clip((depth - near) / (far - near), 0.0, 1.0)
    return DEPTH_LUT[(t * 255).astype(np.uint8)]


def render_normalized_mesh(normalized_landmarks, size=NORM_CANVAS_SIZE,
                           scale=NORM_CANVAS_SCALE, draw_tesselation=True,
                           draw_contours=True, draw_points=False):
    """Render pose-normalized landmarks into a (size, size, 3) BGR image.

    The mesh is drawn in its canonical frame -- nose at the origin, eyes level,
    inter-eye distance 1.0 -- so head pose is gone and every frame shows the
    face from the same viewpoint at the same size. Comparing two renders shows
    expression only.

    The third dimension is not thrown away to fit a flat canvas: each edge is
    tinted by its depth (cool = toward the camera, warm = away) and the edges
    are stroked far-to-near, so nearer geometry paints over farther geometry.
    That painter's ordering is what makes the nose and lips read as in front of
    the cheeks instead of as a flat tangle of lines.

    Args:
        normalized_landmarks: (N, 3) output of normalize_spatial_pose().
        size: canvas resolution (size x size).
        scale: pixels per normalized unit.
        draw_tesselation: stroke the full ~2.5k-edge wireframe.
        draw_contours: outline eyes, brows, lips, face oval and nose in white.
        draw_points: mark each landmark with a depth-colored dot.

    Returns:
        (size, size, 3) uint8 BGR image.
    """
    canvas = np.full((size, size, 3), NORM_BACKGROUND, dtype=np.uint8)

    points = np.round(normalized_to_canvas(normalized_landmarks, size=size,
                                           scale=scale)).astype(int)
    depth = np.asarray(normalized_landmarks)[:, 2]
    colors = depth_colors(depth)

    C = FaceLandmarksConnections
    if draw_tesselation:
        edges = np.array([[c.start, c.end] for c in C.FACE_LANDMARKS_TESSELATION])
        # Painter's algorithm: stroke the farthest edges first so the nearer
        # ones overwrite them, giving the flat render its depth ordering.
        edge_depth = depth[edges].mean(axis=1)
        edge_colors = depth_colors(edge_depth)
        for e in np.argsort(-edge_depth):
            start, end = points[edges[e, 0]], points[edges[e, 1]]
            cv2.line(canvas, tuple(start), tuple(end),
                     edge_colors[e].tolist(), 1, cv2.LINE_AA)

    if draw_contours:
        _draw_connections(canvas, points, C.FACE_LANDMARKS_CONTOURS,
                          NORM_CONTOUR_COLOR)

    if draw_points:
        for i in np.argsort(-depth):
            cv2.circle(canvas, tuple(points[i]), 1, colors[i].tolist(), -1, cv2.LINE_AA)

    return canvas


def build_neutral(detector, split=NEUTRAL_SPLIT, image_file=NEUTRAL_IMAGE):
    """Detect the neutral frame and bake the rasterization layout from it.

    Built once and shared by every split: the neutral is what the deltas are
    measured against, so a per-split neutral would put Train and Test maps in
    different coordinate conventions and quietly bias anything evaluated across
    the two.

    Returns:
        (normalized_neutral, raster_map)
    """
    image_path = os.path.join(split_paths(split)['images'], image_file)
    neutral_landmarks, neutral_image = detect_facemesh(detector, image_path)
    if neutral_landmarks is None:
        raise RuntimeError(f"No face detected in the neutral frame {image_path}")

    neutral_h, neutral_w = neutral_image.shape[:2]
    normalized_neutral = normalize_spatial_pose(neutral_landmarks, neutral_w, neutral_h)
    raster_map = build_delta_raster_map(normalized_neutral)

    face_pixels = len(raster_map['pixels'])
    coverage = face_pixels / float(NORM_CANVAS_SIZE ** 2)
    print(f"Neutral: {split}/Images/{image_file} ({neutral_w}x{neutral_h}), "
          f"delta maps {NORM_CANVAS_SIZE}x{NORM_CANVAS_SIZE}x4 (dx, dy, dz, mask), "
          f"{len(FACEMESH_TRIANGLES)} mesh triangles, "
          f"face covers {100 * coverage:.1f}% of the canvas ({face_pixels} px)")
    return normalized_neutral, raster_map


def process_split(detector, split, normalized_neutral, raster_map):
    """Run the full pass over one split, writing into data/<split>/FaceMeshes."""
    paths = split_paths(split)
    image_dir, facemesh_dir = paths['images'], paths['facemesh']

    os.makedirs(facemesh_dir, exist_ok=True)
    if NUM_PREVIEWS:
        os.makedirs(paths['previews'], exist_ok=True)

    image_files = list_images(image_dir)
    print(f"\n[{split}] Found {len(image_files)} images in {image_dir}")
    if not image_files:
        return

    landmarks_by_image = {}
    sizes_by_image = {}
    failed = []
    delta_min = np.full(3, np.inf)
    delta_max = np.full(3, -np.inf)

    for i, image_file in enumerate(image_files):
        landmarks, image_bgr = detect_facemesh(
            detector, os.path.join(image_dir, image_file))

        if landmarks is None or landmarks.shape[0] != NUM_FACEMESH_LANDMARKS:
            failed.append(image_file)
            continue

        h, w = image_bgr.shape[:2]
        stem = os.path.splitext(image_file)[0]

        # The raw (468, 3) landmarks are collected for the bundled .pt rather
        # than written per image: one <stem>.npy in FaceMeshes/ is the delta map.
        landmarks_by_image[image_file] = landmarks
        sizes_by_image[image_file] = (w, h)

        # Deltas live in the canonical frame, so what they measure is expression
        # only -- head pose is already gone on both sides of the subtraction.
        normalized_landmarks = normalize_spatial_pose(landmarks, w, h)
        delta_landmarks = normalized_landmarks - normalized_neutral
        delta_map = rasterize_deltas(delta_landmarks, raster_map)
        np.save(os.path.join(facemesh_dir, f"{stem}.npy"), delta_map)

        # Stats over face pixels only; the zeros outside would otherwise pin
        # every axis's range to include 0 whatever the deformation did.
        on_face = delta_map.reshape(-1, 4)[raster_map['pixels'], :3]
        delta_min = np.minimum(delta_min, on_face.min(axis=0))
        delta_max = np.maximum(delta_max, on_face.max(axis=0))

        if i < NUM_PREVIEWS:
            cv2.imwrite(os.path.join(paths['previews'], f"{stem}_delta.png"),
                        render_delta_map(delta_map))

        if i % 300 == 0:
            print(f"[{split}] Processed {i}/{len(image_files)} images: {image_file}")

    save_facemesh(landmarks_by_image, sizes_by_image, paths['bundle'])

    print(f"[{split}] Wrote {len(landmarks_by_image)} delta maps "
          f"({NORM_CANVAS_SIZE}x{NORM_CANVAS_SIZE}x4 float32) to {facemesh_dir}")
    print(f"[{split}] Delta range (normalized units, inter-eye = 1.0):")
    for axis, lo, hi in zip('xyz', delta_min, delta_max):
        print(f"  d{axis}: [{lo:+.4f}, {hi:+.4f}]")
    if failed:
        print(f"[{split}] No face detected in {len(failed)} images: {failed[:10]}"
              + (" ..." if len(failed) > 10 else ""))


def main(splits=SPLITS):
    detector = vision.FaceLandmarker.create_from_options(options)

    # --- Neutral frame: defines the deltas' zero and the rasterization layout. ---
    normalized_neutral, raster_map = build_neutral(detector)

    for split in splits:
        process_split(detector, split, normalized_neutral, raster_map)


def _arg_value(flag, default=None):
    """Value following `flag` on the command line, or `default` if absent."""
    if flag not in sys.argv:
        return default
    pos = sys.argv.index(flag) + 1
    return sys.argv[pos] if pos < len(sys.argv) else default

# python src/MediaPipeFacemesh.py                # full pass over Train and Test
# python src/MediaPipeFacemesh.py --split Test   # only one split (Train or Test)
# python src/MediaPipeFacemesh.py --preview 10   # only draw the first 10 mesh overlays

if __name__ == '__main__':
    split_arg = _arg_value('--split')
    if split_arg is not None and split_arg not in SPLITS:
        raise SystemExit(f"--split must be one of {SPLITS}, got {split_arg!r}")

    if '--preview' in sys.argv:
        preview(int(_arg_value('--preview', NUM_PREVIEWS)),
                split=split_arg or NEUTRAL_SPLIT)
    else:
        main((split_arg,) if split_arg else SPLITS)
