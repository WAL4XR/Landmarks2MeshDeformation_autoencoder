import os
import sys
import math
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_dir = os.path.join(BASE_DIR, 'data', 'csv')
header = []
au_indices_r = []
au_counts = {}
au_avg = {}
au_std = {}
total_frames = 0

#intializing
sample_file = os.listdir(csv_dir)[0]
with open(os.path.join(csv_dir, sample_file), 'r') as f:
    header = f.readline().strip().split(',')
    au_indices_r = [i for i, col in enumerate(header) if col.startswith('AU') and col.endswith('_r')]
    au_counts = {col: 0 for col in header if col.startswith('AU') and col.endswith('_r')}
    au_avg = {col: 0.0 for col in header if col.startswith('AU') and col.endswith('_r')}
    au_sq_sum = {col: 0.0 for col in header if col.startswith('AU') and col.endswith('_r')}
    au_std = {col: 0.0 for col in header if col.startswith('AU') and col.endswith('_r')}

n_aus = len(au_indices_r)
cov_sum_x = np.zeros(n_aus)
cov_sum_xx = np.zeros((n_aus, n_aus))

for csv_file in os.listdir(csv_dir):
    if csv_file.endswith('.csv'):
        csv_path = os.path.join(csv_dir, csv_file)
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            total_frames += len(lines) - 1
            for line in lines[1:]:
                values = line.strip().split(',')
                row = np.array([float(values[idx]) for idx in au_indices_r])
                cov_sum_x += row
                cov_sum_xx += np.outer(row, row)
                for k, idx in enumerate(au_indices_r):
                    v = row[k]
                    if v > 0.0:
                        au_counts[header[idx]] += 1
                        au_avg[header[idx]] += v
                        au_sq_sum[header[idx]] += v * v


for i in au_counts:
    if au_counts[i] > 0:
        au_avg[i] /= au_counts[i]
        variance = au_sq_sum[i] / au_counts[i] - au_avg[i] ** 2
        au_std[i] = math.sqrt(max(variance, 0.0))

cov_mean = cov_sum_x / total_frames
cov_matrix = cov_sum_xx / total_frames - np.outer(cov_mean, cov_mean)

cov_names = [header[idx].replace('_r', '') for idx in au_indices_r]
sort_order = sorted(range(n_aus), key=lambda k: cov_names[k])
cov_labels = [cov_names[k] for k in sort_order]
cov_matrix_sorted = cov_matrix[np.ix_(sort_order, sort_order)]
cov_matrix_rounded = np.round(cov_matrix_sorted, 2).tolist()


def plot_au_stats(au_counts, au_avg):
    aus = sorted(au_counts.keys())
    labels = [au.replace('_r', '') for au in aus]
    counts = [au_counts[au] for au in aus]
    avgs = [au_avg[au] for au in aus]

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.bar(labels, counts, color='steelblue')
    ax1.set_title('AU activation frequency (frames with intensity > 0)')
    ax1.set_ylabel('Frame count')
    ax1.tick_params(axis='x', rotation=45)

    ax2.bar(labels, avgs, color='darkorange')
    ax2.set_title('AU average intensity (when active)')
    ax2.set_ylabel('Mean intensity')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

# save AU stats to json
out_dir = os.path.join(BASE_DIR, 'data', 'processed', 'AUs')
os.makedirs(out_dir, exist_ok=True)
au_data = [
    {'name': name.replace('_r', ''), 'count': au_counts[name], 'average': round(au_avg[name], 2), 'std': round(au_std[name], 2)}
    for name in sorted(au_counts.keys())
]
with open(os.path.join(out_dir, 'au_stats.json'), 'w') as f:
    json.dump({
        'total_frames': total_frames,
        'au_data': au_data,
        'covariance': {'labels': cov_labels, 'matrix': cov_matrix_rounded},
    }, f, indent=2)

plot_au_stats(au_counts, au_avg)

