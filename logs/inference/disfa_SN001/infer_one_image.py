"""Run the facemesh -> deformation pipeline on one arbitrary image.

MediaPipeFacemesh.py only walks whole splits, so this reproduces its per-image
path for a single file: detect the 468-point mesh, cancel head pose against the
same shared neutral the dataset was built on, rasterize the deltas into the
canonical 256x256 canvas, then hand that .npy to infer_comb.translate_map().

The neutral matters. The dataset's zero is the Unity avatar's neutral frame
(Train/Images/00000.jpg), so a delta measured against it for a *real* face is
identity + expression, not expression. --subject-neutral measures against a
resting frame of the same person instead, which isolates expression the way the
training deltas do.
"""
import argparse, os, sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
REPO = "/home/habiba/Documents/GitHub/Dataset_Curation_pipeline"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "src", "models"))
os.chdir(REPO)

import cv2, numpy as np, torch
import MediaPipeFacemesh as MPF
from mediapipe.tasks.python import vision
import infer_comb


def mesh_for(detector, path):
    lm, bgr = MPF.detect_facemesh(detector, path)
    if lm is None:
        raise SystemExit(f"No face detected in {path}")
    h, w = bgr.shape[:2]
    return lm, bgr, MPF.normalize_spatial_pose(lm, w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--stem", required=True)
    ap.add_argument("--checkpoint", default="models/checkpoints/comb_step500000.pt")
    ap.add_argument("--preview-dir", default="logs/inference/single_image")
    ap.add_argument("--npy-dir", default="logs/inference/single_image/facemesh")
    ap.add_argument("--out-dir", default="data/translated/deformation")
    ap.add_argument("--subject-neutral", help="resting frame of the same person "
                    "to measure the deltas against instead of the dataset neutral")
    a = ap.parse_args()

    os.makedirs(a.preview_dir, exist_ok=True)
    os.makedirs(a.npy_dir, exist_ok=True)

    detector = vision.FaceLandmarker.create_from_options(MPF.options)

    # The raster layout is baked from the dataset neutral either way: it fixes
    # which canvas pixel each mesh triangle owns, and that convention has to
    # match the one the encoder was trained on.
    dataset_neutral, raster_map = MPF.build_neutral(detector)

    lm, bgr, normalized = mesh_for(detector, a.image)
    h, w = bgr.shape[:2]
    yaw, pitch, roll = MPF.head_pose_angles(
        MPF.head_frame(MPF.landmarks_to_pixels(lm.astype(np.float64), w, h)))
    print(f"\n{a.image} ({w}x{h}): {lm.shape[0]} landmarks, "
          f"head pose yaw {yaw:+.1f} pitch {pitch:+.1f} roll {roll:+.1f}")

    zero, tag = dataset_neutral, "dataset-neutral"
    if a.subject_neutral:
        _, _, zero = mesh_for(detector, a.subject_neutral)
        tag = "subject-neutral"
        print(f"zero = {a.subject_neutral}")

    delta = normalized - zero
    delta_map = MPF.rasterize_deltas(delta, raster_map)

    on_face = delta_map.reshape(-1, 4)[raster_map["pixels"], :3]
    print(f"delta range ({tag}, inter-eye = 1.0):")
    for ax, lo, hi in zip("xyz", on_face.min(0), on_face.max(0)):
        print(f"  d{ax}: [{lo:+.4f}, {hi:+.4f}]")
    print(f"  |delta| mean {np.abs(on_face).mean():.4f}  p99 {np.percentile(np.abs(on_face),99):.4f}")

    npy = os.path.join(a.npy_dir, f"{a.stem}.npy")
    np.save(npy, delta_map)

    # Landmark previews, the same three MediaPipeFacemesh.py writes.
    p = lambda n: os.path.join(a.preview_dir, f"{a.stem}_{n}.png")
    cv2.imwrite(p("mesh"), MPF.draw_facemesh(bgr, lm))
    cv2.imwrite(p("normalized"), MPF.render_normalized_mesh(normalized))
    cv2.imwrite(p("delta"), MPF.render_delta_map(delta_map))
    print(f"previews -> {p('mesh')}, {p('normalized')}, {p('delta')}")

    # How much of the map survives the encoder's input clamp: the facemesh
    # scale is what the deltas are divided by before the [-1, 1] clip, so
    # anything past it is saturated and the model cannot see its magnitude.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, scales, step = infer_comb.load_checkpoint(a.checkpoint, device)
    sat = float((np.abs(on_face) > scales["facemesh"]).mean())
    print(f"\ncheckpoint step {step}, facemesh scale {scales['facemesh']:.5f} "
          f"-> {100*sat:.1f}% of on-face delta components clip at +/-1")

    infer_comb.translate_map(model, cfg, scales, device, npy, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
