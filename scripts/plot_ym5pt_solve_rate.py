#!/usr/bin/env python
"""
Stacked bar chart: solve rate by phase (greedy vs beam search) for YM 5pt forms,
with CDS solve rate overlaid.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# ── Our data: (form_idx, initial_terms, phase) ──
# phase: 'greedy' = solved in greedy contrastive-grouping phase alone
#        'beam'   = required beam search (phase 2)
raw_data = [
    (0, 8, 'greedy'), (10, 14, 'beam'), (20, 17, 'beam'), (30, 20, 'beam'),
    (40, 24, 'greedy'), (50, 26, 'greedy'), (60, 28, 'greedy'), (70, 30, 'beam'),
    (80, 32, 'greedy'), (90, 36, 'beam'), (100, 36, 'greedy'), (110, 38, 'greedy'),
    (120, 41, 'beam'), (130, 43, 'beam'), (140, 44, 'greedy'), (150, 45, 'beam'),
    (160, 48, 'beam'), (170, 49, 'greedy'), (180, 50, 'greedy'), (190, 52, 'beam'),
    (200, 54, 'greedy'), (210, 54, 'beam'), (220, 55, 'beam'), (230, 57, 'beam'),
    (240, 58, 'beam'), (250, 59, 'greedy'), (260, 60, 'beam'), (270, 62, 'beam'),
    (280, 63, 'beam'), (290, 65, 'beam'), (300, 66, 'beam'), (310, 67, 'greedy'),
    (320, 68, 'beam'), (330, 69, 'beam'), (340, 70, 'beam'), (350, 71, 'beam'),
    (360, 72, 'greedy'), (370, 74, 'greedy'), (380, 75, 'beam'), (390, 75, 'greedy'),
    (400, 76, 'beam'), (410, 77, 'beam'), (420, 78, 'beam'), (430, 80, 'beam'),
    (440, 80, 'beam'), (450, 81, 'beam'), (460, 82, 'greedy'), (470, 83, 'beam'),
    (480, 84, 'beam'), (490, 85, 'beam'), (500, 86, 'beam'), (510, 87, 'beam'),
    (520, 88, 'beam'), (530, 88, 'beam'), (540, 89, 'beam'), (550, 90, 'beam'),
    (560, 92, 'beam'), (570, 93, 'beam'), (580, 95, 'beam'), (590, 96, 'beam'),
    (600, 96, 'beam'), (610, 98, 'beam'), (620, 99, 'beam'), (630, 100, 'beam'),
    (640, 101, 'beam'), (650, 103, 'beam'), (660, 104, 'beam'), (670, 105, 'beam'),
    (680, 106, 'beam'), (690, 107, 'beam'), (700, 108, 'beam'), (710, 110, 'beam'),
    (720, 111, 'beam'), (730, 112, 'beam'), (740, 113, 'beam'), (750, 115, 'beam'),
    (760, 117, 'beam'), (770, 118, 'beam'), (780, 119, 'beam'), (790, 121, 'beam'),
    (800, 122, 'beam'), (810, 124, 'beam'), (820, 126, 'beam'), (830, 128, 'beam'),
    (840, 130, 'beam'), (850, 131, 'beam'), (860, 134, 'beam'), (870, 136, 'beam'),
    (880, 138, 'beam'), (890, 139, 'beam'), (900, 142, 'beam'), (910, 144, 'beam'),
    (920, 146, 'greedy'), (930, 148, 'beam'), (940, 152, 'beam'), (950, 155, 'beam'),
    (960, 159, 'beam'), (970, 163, 'beam'), (980, 172, 'beam'), (990, 178, 'beam'),
    (1000, 184, 'beam'), (1010, 193, 'beam'), (1020, 214, 'beam'),
]

initial_terms = np.array([d[1] for d in raw_data])
phases = [d[2] for d in raw_data]

# ── Bin into groups of 25 terms ──
bin_width = 25
bin_min = 0
bin_max = 225
bin_edges = np.arange(bin_min, bin_max + bin_width, bin_width)

bin_centers = []
n_greedy = []
n_beam = []
n_total = []

for i in range(len(bin_edges) - 1):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    mask = (initial_terms >= lo) & (initial_terms < hi)
    total = mask.sum()
    if total == 0:
        continue
    greedy_count = sum(1 for j in range(len(raw_data)) if mask[j] and phases[j] == 'greedy')
    beam_count = sum(1 for j in range(len(raw_data)) if mask[j] and phases[j] == 'beam')
    bin_centers.append((lo + hi) / 2)
    n_greedy.append(greedy_count)
    n_beam.append(beam_count)
    n_total.append(total)

bin_centers = np.array(bin_centers)
n_greedy = np.array(n_greedy, dtype=float)
n_beam = np.array(n_beam, dtype=float)
n_total = np.array(n_total, dtype=float)

frac_greedy = n_greedy / n_total
frac_beam = n_beam / n_total

# ── CDS digitized data (from CDS Fig 9 right panel, arXiv:2408.04720v2) ──
# Sequential (orange) curve, rolling window of size 5, 850 amplitudes up to 200 terms
# Digitized via pixel-level extraction at 600 DPI with tick-mark axis calibration
# X ticks at pixels [2790, 3082, 3375, 3667, 3959] = data [0, 50, 100, 150, 200]
# Y ticks at pixels [775, 960, 1144, 1329, 1514, 1699] = data [100, 80, 60, 40, 20, 0]
cds_x = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
          85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145,
          150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215]
cds_y = [100, 100, 100, 100, 100, 100, 100, 98.5, 98.8, 98.9, 84.2, 84.2,
          84.2, 84.2, 78.9, 72.4, 77.9, 72.5, 65.8, 65.2, 56.8, 53.1, 43.3,
          43.2, 41.7, 44.5, 42.5, 39.0, 33.0, 29.4, 19.2, 11.0, 10.7, 7.7,
          13.7, 21.6, 41.0, 8.4, 1.0, 1.0, 1.0, 0.3, 1.0, 1.0]
cds_x = np.array(cds_x, dtype=float)
cds_y = np.array(cds_y, dtype=float) / 100.0

# Average CDS into the same 25-term bins as our data
# For bins beyond the CDS data range (max 165 terms), use 0% solve rate
cds_bin_centers = []
cds_bin_avg = []
for i in range(len(bin_edges) - 1):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    # Only include bins that have our data
    mask_ours = (initial_terms >= lo) & (initial_terms < hi)
    if mask_ours.sum() == 0:
        continue
    mask = (cds_x >= lo) & (cds_x < hi)
    cds_bin_centers.append((lo + hi) / 2)
    if mask.sum() > 0:
        cds_bin_avg.append(cds_y[mask].mean())
    else:
        cds_bin_avg.append(0.0)  # CDS data doesn't extend here
cds_bin_centers = np.array(cds_bin_centers)
cds_bin_avg = np.array(cds_bin_avg)

# ── Plot ──
fig, ax = plt.subplots(figsize=(8, 5))

bar_w = bin_width * 0.7

# Stacked bars: greedy on bottom, beam on top
ax.bar(bin_centers, frac_greedy, width=bar_w,
       color='#2196F3', edgecolor='black', linewidth=0.5,
       label='This work (greedy)')
ax.bar(bin_centers, frac_beam, width=bar_w, bottom=frac_greedy,
       color='#4CAF50', edgecolor='black', linewidth=0.5,
       label='This work (beam search)')

# CDS overlay (binned averages)
ax.plot(cds_bin_centers, cds_bin_avg, color='darkorange', linewidth=2.5, marker='o',
        markersize=6, label='CDS', zorder=5)

# Count labels on each bar
for x, ng, nb, nt in zip(bin_centers, n_greedy, n_beam, n_total):
    ax.text(x, 1.02, f'{int(nt)}',
            ha='center', va='bottom', fontsize=8, color='gray')

ax.set_xlabel('Initial number of terms', fontsize=13)
ax.set_ylabel('Solve rate', fontsize=13)
ax.set_ylim(0, 1.15)
ax.set_xlim(bin_edges[0] - 5, bin_edges[-1] + 5)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax.legend(fontsize=10, loc='center right')

ax.set_xticks(bin_centers)
ax.set_xticklabels([f'{int(c - bin_width/2)}--{int(c + bin_width/2)}' for c in bin_centers],
                   rotation=45, ha='right', fontsize=9)
ax.tick_params(axis='y', labelsize=11)

plt.tight_layout()

outpath = os.path.join(REPO_ROOT, 'figures', 'ym5pt_solve_rate.pdf')
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f"Saved to {outpath}")
