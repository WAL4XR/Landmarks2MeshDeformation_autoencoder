import os

# 1. Set the variable BEFORE importing cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# 2. Now import cv2
import cv2
import numpy as np
import torch


def load_exr(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img[..., :3]  # Keep only the first three channels (RGB)


def summarize(directory, ext, load):
    files = sorted(f for f in os.listdir(directory) if f.endswith(ext))
    print(f"\nFound {len(files)} {ext} files in {directory}")

    n_read = 0
    n_failed = 0
    total_min = np.inf
    total_max = -np.inf
    total_sum = 0.0
    total_zeros = 0
    total_elems = 0
    shapes = set()
    dtypes = set()

    for name in files:
        try:
            arr = load(os.path.join(directory, name))
        except Exception as e:
            arr = None
            print(f"Failed to read {name}: {e}")

        if arr is None:
            n_failed += 1
            continue

        n_read += 1
        shapes.add(arr.shape)
        dtypes.add(str(arr.dtype))
        total_min = min(total_min, float(arr.min()))
        total_max = max(total_max, float(arr.max()))
        total_sum += float(arr.astype(np.float64).sum())
        total_zeros += int((arr == 0).sum())
        total_elems += arr.size

    if n_read == 0:
        print("No files read - check the directory path!")
        return

    print(f"Read: {n_read}, Failed: {n_failed}")
    print(f"Shapes: {shapes}, Types: {dtypes}")
    print(f"Min: {total_min}, Max: {total_max}, Mean: {total_sum / total_elems}")
    print(f"Zeros: {total_zeros} / {total_elems} ({100 * total_zeros / total_elems:.2f}%)")


summarize("./data/processed/DeformationMaps", ".exr", load_exr)
summarize("./data/processed/Landmarks/npys", ".npy", np.load)



# v = torch.rand(5) * 1000
# v_1 = v.clone()
# v.requires_grad_(True)
# v_1.requires_grad_(True)

# loss = 1/2 * torch.sum(v_1 * v_1 + v * v)
# # Here grads of loss w.r.t v and v_1 should be v and v_1 respectively
# loss.backward()

# # Clip grads of v_1
# grad = float(torch.nn.utils.clip_grad_norm_(v_1, max_norm=1.0, norm_type=2))

# print(v.grad)
# print(v_1.grad)
# print(v.grad / torch.norm(v.grad, p=2))
# print(grad)
# print(torch.norm(v_1.grad, p=2))