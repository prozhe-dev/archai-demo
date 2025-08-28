#!/usr/bin/env python
# coding: utf-8

# # ↓ 🏙️ Input Image
# Now we need an example image to work with.
# 
# What API will input to.

# In[1]:


import os

current_dir = os.getcwd()
tmp_dir = os.path.join(current_dir, "tmp")

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

main_img_path = os.path.join(tmp_dir, "floorplan.png")
img_path = main_img_path

print("Image path:", img_path)


# In[2]:


# This is the cell for API to place the uploaded image
# ---------------------------------------------------------------------------- #
#                               Images Tested are                              #
# ---------------------------------------------------------------------------- #
# /Images
# ├── atlantic-floor-plan.png
# ├── avenue_fab_floorplan.png
# ├── example.png
# ├── example8.png
# ├── f10.png
# ├── f11.png
# ├── f12.png
# ├── f3.png
# ├── f4.png
# ├── f5.png
# ├── f7.png
# ├── f8.png
# ├── f9.png
# ├── flrpln1.png
# ├── image2.png
# ├── iou-precision-recall.png
# ├── my_example1.png
# ├── my_example2.png
# └── my_example3.png

# SET IMAGE PATH HERE TOO

# main_img_path = "./Images/flrpln1.png"






# In[3]:


img_path = main_img_path


# # Imports

# In[4]:


# Import library
import matplotlib.pyplot as plt
import cv2
import numpy as np
import io 

# Define a dummy display function for non-Jupyter environments
try:
from IPython.display import display
except ImportError:
    def display(*args, **kwargs):
        pass  # Do nothing in non-Jupyter environments

from PIL import Image

import math
import matplotlib.pyplot as plt


import cv2 # for image gathering
import numpy as np
import json

# for visualize
from PIL import Image
from IPython.display import display


# SHAPELY
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

import cv2
from IPython.display import display
from PIL import Image

import shapely
from shapely.geometry import LineString, Point

from shapely.geometry import LineString




# ## Show image:
# 
# <!-- ![input](Images/image2.png) -->

# In[5]:


img = cv2.imread(img_path)
display(Image.fromarray(img))


# In[6]:


img = cv2.imread(main_img_path)


# display(Image.fromarray(img))


# img = cv2.detailEnhance(img, sigma_s=100, sigma_r=100)


# img = cv2.erode(img, np.ones((3, 3), np.uint8), iterations=1)
# display(Image.fromarray(img))

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
# display(Image.fromarray(thresh))
# thresh = 255 - thresh
# display(Image.fromarray(thresh))

# img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)




# img = cv2.detailEnhance(img, sigma_s=100, sigma_r=100)
# img = cv2.stylization(img, sigma_s=0.5, sigma_r=10)


display(Image.fromarray(img))


# # ↓ Download the model .pkl

# In[7]:


import os

model_path = 'model_best_val_loss_var.pkl'
if not os.path.exists(model_path):
    # Download only if file does not exist
    import subprocess
    subprocess.run(["gdown", "https://drive.google.com/uc?id=1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK"])
else:
    print(f"{model_path} already exists, skipping download.")



# # ⚙️ Utils

# 
# ## ⚙️ 💾 Save to file

# In[8]:


def save_to_file(file_path, data, show=True):
    '''
    Save to file
    Saves our resulting array as json in file.
    @Param file_path, path to outputfile
    @Param data, data to write to file
    '''
    with open(file_path+'.txt', 'w') as f:
        f.write(json.dumps(data))

    if show:
        print("Created file : " + file_path + ".txt")


# ## ⚙️ 📈 Plot Vertices

# In[9]:


def plot_vertices(wall_vertices, object_vertices):
    """
    Plot wall and window segments on a 2D floor plan with equal scaling
    @Param wallVertices list of wall groups, each group is a list of segments with (x, y) coordinates @mandatory
    @Param windowVertices list of window segments, each segment is a pair of (x, y) coordinates @mandatory
    @Return None
    
    Requires:
        matplotlib.pyplot as plt (must be imported in the above imports cell).
    """
    # plt.figure(figsize=(12, 12))
    
    # Plot walls as blue lines
    for wall_group in wall_vertices:
        for segment in wall_group:
            x = [segment[0][0], segment[1][0]]
            y = [-segment[0][1], -segment[1][1]] # Invert y for consistent orientation
            # plt.plot(x, y, 'b-', linewidth=1, label='Wall')
    
    # Plot windows as red lines
    for window in object_vertices:
        for segment in window:
            x = [segment[0][0], segment[1][0]]
            y = [-segment[0][1], -segment[1][1]] # Invert y for consistent orientation
            # plt.plot(x, y, 'r-', linewidth=2, label='Window')
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    
    # plt.title('Floor Plan - Walls and Windows')
    # plt.axis('equal')  # Equal aspect ratio
    # plt.grid(True)
    # plt.show()


# ## ⚙️ Export vertices to .txt

# In[10]:


def create_vertices(boxes, scale=100):
    """
    Convert a list of 2D contour boxes into scaled vertex pairs representing wall segments.

    @Param boxes list of box arrays, each array is a polygon/contour made of ordered points [[(x, y)], ...] @mandatory
    @Param scale numeric scaling factor to convert from pixel units to real-world coordinate units (default: 100) @optional
    @Return list of box vertex groups, where each group is a list of wall segments as [[(x1, y1), (x2, y2)], ...]

    Requires:
    - numpy as np
    - cv2 (OpenCV)
    - from PIL import Image
    - from IPython.display import display
    """
    verts = [] # final list of vertex groups

    for box in boxes:
        box_verts = [] # store vertices for current polygon
        for index in range(0, len(box)):
            temp_verts = []
            # Get current
            curr = box[index][0]

            # is last, link to first
            if(len(box)-1 >= index+1):
                next = box[index+1][0];
            else:
                next = box[0][0]; # link to first pos

            # Create all 3d poses for each wall
            temp_verts.extend([[curr[0]/scale, curr[1]/scale]])
            temp_verts.extend([[next[0]/scale, next[1]/scale]])

            # add wall verts to verts
            box_verts.extend([temp_verts])
    

        verts.extend([box_verts])

    return verts


# ## ⚙️ 📐 check if perpendicular 
# 
# (3 points by 3 points check perpendicularity)

# In[11]:


def is_perpendicular(p1, p2, p3):
    """
    Determine whether three points form an approximate right angle (90°) at a given corner point.

    @Param p1 numpy array [x, y] representing the first endpoint @mandatory
    @Param p2 numpy array [x, y] representing the corner/vertex point (angle is measured here) @mandatory
    @Param p3 numpy array [x, y] representing the second endpoint @mandatory
    @Return bool True if the angle at p2 is within a tolerance of 90°, otherwise False
    
    Requires:
        - numpy as np
    """
    v1 = p1 - p2
    v2 = p3 - p2
    dot = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos = dot / norms
    # Allow some tolerance around 90 degrees (cos ≈ 0)
    return abs(cos) < 0.1


# ## ⚙️ 📐📏 straighten to recilinear shapes

# In[12]:


def straighten_rectilinear(contour, length_threshold=10):
    """
    Snap contour edges to axis-aligned (rectilinear) segments and
    merge small stair-steps into clean 90° polygon lines.

    @Param contour: np.ndarray
        Contour points (n, 1, 2) as produced by cv2.findContours or approxPolyDP.
    @Param length_threshold: int
        Minimum pixel distance between consecutive kept vertices.
        Prevents adding redundant points that are too close together.
    @Return np.ndarray
        Rectified contour with edges constrained to horizontal/vertical,
        shape (m, 1, 2) ready for use with OpenCV drawing/mesh routines.

    Requires:
        - numpy as np
        - cv2 (OpenCV)
    """
    points = contour.reshape(-1, 2)
    straightened = []

    n = len(points)
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]

        dx, dy = p2 - p1

        # Force direction to axis-aligned
        # Horizontal if dx dominates, otherwise vertical
        if abs(dx) > abs(dy):
            p2[1] = p1[1]  # horizontal
        else:
            p2[0] = p1[0]  # vertical

        # Add p2 if it's far enough from last added point
        # Only keep p2 if it's far enough from last accepted point
        if len(straightened) == 0 or np.linalg.norm(p2 - straightened[-1]) > length_threshold:
            straightened.append(p2.copy())

    return np.array(straightened, dtype=np.int32).reshape(-1, 1, 2)


# ## ⚙️ ▫️⬜ Scale point to vector

# In[13]:


# def scale_point_to_vector(boxes, scale = 1, height = 0):
#     '''
#     Scale point to vector
#     scales a point to a vector
#     @Param boxes
#     @Param scale
#     @Param height
#     @source https://github.com/grebtsew/FloorplanToBlender3d/tree/master/FloorplanToBlenderLib
#     '''
#     res = []
#     for box in boxes:
#         for pos in box:
#             res.extend([(pos[0]/scale, pos[1]/scale, height)])
#     return res


# ## ⚙️ ▭ → ▬ fix rectangles of 🚪🪟
# 
# Fixing orders of vertices and lengths of 🚪🪟

# In[14]:


def fix_rectangles(windows):
    """
    Fix Rectangles from Window Segments

    Ensures each detected window (list of 4 edges) is reshaped into a clean 
    axis-aligned rectangle. If insufficient or degenerate points exist, 
    applies a small offset (0.1) to avoid zero-size shapes. 
    
    The function overwrites each window's 4 edges with consistent 
    top, right, bottom, and left segments.

    @Param windows list of window vertex groups (each group = 4 edges) @mandatory
    @Return list of window vertex groups corrected into proper rectangles

    Requires:
        - numpy as np (optional, only if you chain with NumPy-based ops)
        - windows data structure as list of edges [[(x1,y1),(x2,y2)], ...]
    """
    for w in windows:
        # Get unique vertices
        pts = {tuple(p) for s in w for p in s if s[0] != s[1]}
        pts = [list(p) for p in pts]
        
        # Find min/max or add offset for insufficient points
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min, x_max = min(xs), max(xs) + (0.1 if len(xs) == 1 else 0)
        y_min, y_max = min(ys), max(ys) + (0.1 if len(ys) == 1 else 0)
        
        # Assign segments: top, right, bottom, left
        w[0] = [[x_min, y_max], [x_max, y_max]]
        w[1] = [[x_max, y_max], [x_max, y_min]]
        w[2] = [[x_max, y_min], [x_min, y_min]]
        w[3] = [[x_min, y_min], [x_min, y_max]]
    
    return windows


# # ▭ [Floor] Detect Footprint

# ## ⚙️ ▭ [Floor] Detect Footprint Functions

# ### ⚙️ 📐 ▭ check contour perpendicularity score

# In[15]:


def check_contour_perpendicularity(points):
    """
    Check if all corners in a contour are approximately perpendicular.

    @Param points numpy array of contour points with shape (n, 1, 2) or (n, 2) @mandatory
    @Return tuple (bool, float) where:
        - bool: True if all corners are perpendicular within tolerance
        - float: proportion of corners that are perpendicular

    Requires:
        - numpy as np
        - helper function: is_perpendicular(p1, p2, p3)  (optional, if you prefer reusing instead of inline logic)
    """
    if len(points) < 3:
        return False, 0 # Need at least 3 points to form corners
    
    perpendicular_count = 0
    total_corners = 0

    # Flatten to (n, 2) for easier math
    points = points.reshape(-1, 2)
    n = len(points)
    
    for i in range(n):
        # Previous, current, and next points (with wrap-around)
        prev_point = points[(i-1) % n]
        curr_point = points[i]
        next_point = points[(i+1) % n]
        
        # Calculate vectors between points
        v1 = prev_point - curr_point
        v2 = next_point - curr_point
        
        # Dot product & norms → cos(angle)
        dot = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norms != 0:
            cos = dot / norms
            if abs(cos) < 0.1:  # Close to 90 degrees
                perpendicular_count += 1
        total_corners += 1
    
    # We count the number of corners that are perpendicular (over all corners)
    # Score = fraction of perpendicular corners
    perpendicularity_score = perpendicular_count / total_corners if total_corners > 0 else 0
    
    return perpendicularity_score == 1.0, perpendicularity_score


# ### ⚙️ ▭ evaluate contour 
# 
# (Ultimate score (70% area, 30% perpendicularity))

# In[16]:


def evaluate_contour(filtered_points):
    """
    Evaluate a contour based on its area and the perpendicularity of its corners.
    
    @Param filtered_points numpy array of contour points (n, 1, 2) @mandatory
    @Return tuple (score, area, perp_score)
            - score: weighted evaluation score (higher is better)
            - area: raw contour area
            - perp_score: ratio of perpendicular corners

    Requires:
        - numpy as np
        - cv2 (OpenCV)
        - helper function: check_contour_perpendicularity(points)

    Notes:
        - Area is normalized to image size (denominator may need tuning).
        - Weighted score prioritizes perpendicularity (70%) over area (30%).
    """
    # Get area
    area = max(0.01, cv2.contourArea(filtered_points))
    
    # Get perpendicularity score
    _, perp_score = check_contour_perpendicularity(filtered_points)
    
    # Combined score (70% area, 30% perpendicularity)
    # Normalize area to 0-1 range (you might need to adjust the denominator)
    normalized_area = area / 1000000  # Adjust based on your image size
    score = (0.3 * normalized_area) + (0.7 * perp_score)
    
    return score, area, perp_score


# ### ⚙️ ▭ detect Outer Contours [Explanation]
# 

# #### `detectOuterContours`: robust outer-footprint extraction via CV + grid search
# ```#### You can > close this explanation and read it at your own convenience. Code is Below```
# 
# 
# **Goal.** Given a noisy/faint scanned floor plan, recover a **clean, orthogonal outer footprint** suitable for downstream vectorization and 3D reconstruction.
# 
# **Input / Output**
# 
# * **Input:** `detect_img` (grayscale or BGR).
# * **Output:** `(filtered_contour, vis_image, perp_score)` where:
# 
#   * `filtered_contour` is the simplified, mostly-orthogonal outer polygon (`Nx1x2`).
#   * `vis_image` is a visualization with the selected polygon drawn.
#   * `perp_score ∈ [0,1]` is the fraction of right-angle corners in the selected polygon.
# 
# ---
# 
# #### Why a grid search?
# 
# Real scans vary: line thickness, gaps, compression artifacts, copier streaks. Instead of hard-coding brittle thresholds, we **sweep a small grid of morphology + simplification parameters** and pick the best candidate by a **data-driven score**. This mirrors common **industry practice** for classical CV pipelines: tune a few stable hyperparameters to maximize a task-specific metric.
# 
# ---
# 
# #### Method overview
# 
# 1. **Adaptive binarization**
#    Convert to a white-foreground mask robust to uneven illumination:
#    `adaptiveThreshold(..., ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, blockSize=35, C=5)`.
#    *Rationale:* floor plans are line art; local thresholds handle shadows and paper gradients.
# 
# 2. **Stroke thickening (dilation)**
#    Patch tiny breaks in lines with a `3×3` dilation.
#    *Rationale:* small gaps fragment contours; a single dilation pass improves continuity.
# 
# 3. **Grid search over two knobs**
# 
#    * **Gap closing (`gap_mult`)** → morphological closing with **rectangular kernels** along **X** and **Y** to seal short horizontal/vertical breaks. Kernel sizes are set **relative to image width**:
#      `gap_px = int(gap_mult * image_width)`.
#    * **Polygon simplification (`epsilon_mult`)** → Douglas–Peucker tolerance as a **fraction of perimeter**:
#      `epsilon = epsilon_mult * arcLength(contour, True)`.
#      *Rationale:* Both are **scale-aware** (relative, not absolute), so the method generalizes across resolutions.
# 
# 4. **Fill + isolate the footprint**
# 
#    * Connected components to prune noise.
#    * **Flood fill** background to produce solid interiors.
#    * Keep the **largest** filled component as the building footprint.
# 
# 5. **Contour extraction + simplification**
# 
#    * `findContours(..., RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` to get the outer loop.
#    * `approxPolyDP` to regularize wiggly edges into straight segments.
# 
# 6. **Orthogonality filtering**
# 
#    * For each vertex, compute angle cosine:
# 
#      $$
#      \cos\theta = \frac{(p_{i-1}-p_i)\cdot(p_{i+1}-p_i)}{\|p_{i-1}-p_i\|\,\|p_{i+1}-p_i\|}
#      $$
# 
#      Keep vertices with $|\cos\theta| < \tau$ (≈ right angle).
#    * This yields `filtered_contour` emphasizing orthogonal corners.
# 
# 7. **Scoring & model selection**
# 
#    * Area floor: $A=\max(0.01,\text{contourArea})$.
#    * Perpendicularity ratio: $\text{perp\_score} = \frac{\#\text{right-angle corners}}{\#\text{corners}}$.
#    * Combined metric (tunable weights):
# 
#      $$
#      \text{score} = 0.3 \cdot \frac{A}{10^6} + 0.7 \cdot \text{perp\_score}
#      $$
#    * **Early stop** when $\text{perp\_score} \ge 0.90$ or when score is high with good orthogonality.
# 
# 8. **Visualization grid (diagnostics)**
# 
#    * For every `(epsilon_mult, gap_mult)` pair, render the resulting polygon into a subplot.
#    * *Rationale:* fast, visual sanity-check; easy to justify parameter choices in reports.
# 
# ---
# 
# #### Design choices (and why they're sensible)
# 
# * **Relative hyperparameters** (`gap_mult`, `epsilon_mult`) → **resolution-invariant** behavior.
# * **Largest-component assumption** → building shell typically dominates area post-fill.
# * **Right-angle prior** → residential plans are largely **rectilinear**; enforcing orthogonality improves downstream meshing and CAD conversion.
# * **Area+orthogonality scoring** → favors **complete** footprints that are also **architecturally plausible**.
# 
# ---
# 
# #### Complexity & performance
# 
# For $E$ epsilon values and $G$ gap values, runtime is roughly:
# 
# $$
# O\big(EG \cdot (HW + C)\big)
# $$
# 
# where $HW$ is image size and $C$ the contour ops. In practice, $E,G$ are small (dozens), so this is **interactive** on notebook workflows. The search is **embarrassingly parallel** if needed.
# 
# ---
# 
# #### Limitations & notes
# 
# * Works best for **orthogonal** floor plans; curvilinear or highly skewed plans reduce `perp_score`.
# * The **area normalization** (denominator $10^6$) should be set to the **typical image scale** in your dataset.
# * If scans are extremely broken, widen the **gap kernel sweep** (e.g., `gap_mult ∈ [0.003, 0.03]`) and/or add a **pre-dilation** pass.
# * A final **axis-alignment snap** (e.g., `straighten_rectilinear`) can be applied to remove micro-stair-steps.
# 
# ---
# 
# #### TL;DR
# 
# We treat outer-contour recovery as a **small hyperparameter tuning problem** over morphology and polygon simplification. We **score** candidates by **(area, right angles)** and **early-stop** when geometry looks good. This yields a **robust, size-invariant, and auditable** pipeline—simple enough for production, transparent enough for research.

# ### ⚙️ ▭ detect Outer Contours [Code]
# 

# In[ ]:


def detectOuterContours(detect_img, output_img = None, color = [255, 255, 255]):
    """
    Detect the outer footprint of a floorplan and keep only vertices that look orthogonal,
    by sweeping two hyperparameters:
      - gap closing size (morphological closing along X and Y)
      - polygon simplification tolerance (Douglas–Peucker epsilon multiplier)

    @Param detect_img  numpy array (H, W) or (H, W, 3). Source image; grayscale or BGR.  @mandatory
    @Param output_img  numpy array canvas used for drawing previews. If None, a zeroed canvas is created.  @optional
    @Param color       list[int,int,int] BGR color used when drawing contours.  @optional

    @Return (filtered, vis, perp_score)
        filtered     Nx1x2 int array of the simplified outer contour after perpendicularity filtering
        vis          image with the filtered contour drawn (for quick visual inspection)
        perp_score   float in [0,1], fraction of right-angle corners in the filtered polygon

    Requires:
        - numpy as np
        - cv2 (OpenCV)
        - matplotlib.pyplot as plt
        - from PIL import Image  and  from IPython.display import display
        - helper functions:  is_perpendicular(prev, curr, next),  evaluate_contour(contour)
    """
    # -------------------------------
    # Normalize inputs and canvases
    # -------------------------------
    # Ensure we work on a single-channel image for thresholding.
    if detect_img.ndim == 3:
        # cv2.cvtColor(src, code)
        # code = cv2.COLOR_BGR2GRAY converts BGR color to 8-bit grayscale
        gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = detect_img.copy()

    # Visualization canvas. If none provided, create a black image with same shape as detect_img.
    if output_img is None:
        output_img = np.zeros_like(detect_img if detect_img.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    # --------------------------------------------
    # 1) Adaptive threshold handles uneven scans
    # --------------------------------------------
    # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
    #   maxValue: value assigned to pixels that pass the test (255 means white)
    #   adaptiveMethod: ADAPTIVE_THRESH_MEAN_C uses local mean inside blockSize window
    #   thresholdType: THRESH_BINARY_INV gives white foreground on black background
    #   blockSize: odd integer window size for local statistics
    #   C: constant subtracted from local mean (tunes sensitivity)
    bw = cv2.adaptiveThreshold(gray, 255,
                            cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY_INV,
                            blockSize=35, C=5)
    # -- thin walls are now white (foreground) on black background


    # -------------------------------------------------
    # 2) Dilate to thicken strokes and seal tiny gaps
    # ------------------------------------------------- 
    # cv2.dilate(src, kernel, iterations)
    #   kernel: 3x3 ones expands white regions by one pixel layer
    #   iterations: number of dilation passes
    bw = cv2.dilate(bw, np.ones((3, 3), np.uint8), 1)
    print("dilate")
    display(Image.fromarray(bw))

    

    # --------------------------------------------
    # Parameter sweeps for model selection
    # --------------------------------------------
    # epsilon_values multiplies the polygon perimeter to set Douglas–Peucker epsilon
    epsilon_values = np.arange(0.001, 0.202, 0.001)

    # gap_values controls morphological closing span as a fraction of image width.
    # Note: current step 0.001 over a small range yields a single value near 0.005.
    gap_values = np.arange(0.0035, 0.0202, 0.01)

    best_score = -float("inf")
    best_filtered_points = None
    results = {}
    h, w = bw.shape

    # plt.figure(figsize=(5 * len(gap_values), 5 * len(epsilon_values)))
    plot_idx = 1

    
    # Create visualization grid
    # plt.figure(figsize=(20, 5*len(epsilon_values)))

    # --------------------------------------------
    # Sweep (epsilon, gap) grid and evaluate
    # --------------------------------------------
    for eps_idx, epsilon_mult in enumerate(epsilon_values):
        print(f"\n🔍 Testing epsilon {epsilon_mult:.3f}...")
        for gap_idx, gap_mult in enumerate(gap_values):
            print(f"\n🔍 Testing epsilon {epsilon_mult:.3f}, gap_values {gap_mult:.3f}...")

            # -------------------------------------------------
            # 4a) Gap closing: morphological closing in X and Y
            # -------------------------------------------------
            # Translate relative gap size to pixels, see note above about step size.
            # Apply gap-closing to a fresh copy
            gap_px = max(0.0001, int(gap_mult * bw.shape[1]))
            if gap_px < 1:
                continue

            # cv2.getStructuringElement(shape, ksize)
            #   MORPH_RECT: rectangular kernel
            #   (gap_px, 1) closes horizontal breaks, then (1, gap_px) closes vertical breaks
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_px, 1))
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap_px))

            # cv2.morphologyEx(src, op, kernel, iterations)
            #   MORPH_CLOSE = dilation followed by erosion (seals small gaps)
            bw_gap = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_h, 1)
            bw_gap = cv2.morphologyEx(bw_gap, cv2.MORPH_CLOSE, kernel_v, 1)

            # Clean up
            # -------------------------------------------------
            # 4b) Remove stray components by connected components
            # -------------------------------------------------
            # cv2.connectedComponentsWithStats(image, connectivity, ltype)
            #   returns (num_labels, labels, stats, centroids)
            #   stats columns: [x, y, width, height, area]
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw_gap, 8, cv2.CV_32S)
            mask = np.zeros_like(bw_gap)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] > 0:
                    mask[labels == i] = 255
            bw_gap = mask.copy()
            # print("bw_gap")
            # display(Image.fromarray(bw_gap))

            # -------------------------------------------------
            # 4c) Flood-fill background, then invert to fill interiors
            # -------------------------------------------------
            # cv2.floodFill(image, mask, seedPoint, newVal)
            #   mask must be (H+2, W+2)
            ff = bw_gap.copy()
            cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
            # Now white background is filled. Invert and OR with original to get solid shapes.
            bw_filled = cv2.bitwise_not(ff) | bw_gap
            # print("flood fill")
            # display(Image.fromarray(bw_filled))

            # -------------------------------------------------
            # 4d) Keep only the single largest filled component
            # -------------------------------------------------
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw_filled, 8, cv2.CV_32S)
            if num_labels <= 1:
                continue
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            footprint = np.zeros_like(bw_filled)
            footprint[labels == largest] = 255
            # print("footprint")
            # display(Image.fromarray(footprint))

            # -------------------------------------------------
            # 4e) Find and simplify outer contour
            # -------------------------------------------------
            # cv2.findContours(image, mode, method)
            #   mode = RETR_EXTERNAL keeps outermost contours only
            #   method = CHAIN_APPROX_SIMPLE compresses lines to endpoints
            contours, _ = cv2.findContours(footprint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest_contour = max(contours, key=cv2.contourArea)

            # Simplify with Douglas–Peucker:
            # cv2.arcLength(curve, closed=True) → perimeter
            epsilon = epsilon_mult * cv2.arcLength(largest_contour, True)
            # cv2.approxPolyDP(curve, epsilon, closed)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            points = approx.reshape(-1, 2)

            # -------------------------------------------------
            # 4f) Keep only vertices that look orthogonal
            # -------------------------------------------------
            filtered = []
            for i in range(len(points)):
                prev = points[(i - 1) % len(points)]
                curr = points[i]
                next = points[(i + 1) % len(points)]
                if is_perpendicular(prev, curr, next):
                    filtered.append(curr)

            filtered = np.array(filtered).reshape(-1, 1, 2)

            # -------------------------------------------------
            # 4g) Score by area and perpendicularity fraction
            # -------------------------------------------------
            score, area, perp_score = evaluate_contour(filtered)
            results[(epsilon_mult, gap_mult)] = {
                'score': score,
                'area': area,
                'perp_score': perp_score,
                'filtered_points': filtered.copy()
            }
            print(f"score: {score:.3f}, area: {area:.3f}, perp_score: {perp_score:.3f}")


            # -------------------------------------------------
            # 4h) Visualization tile for this hyperparameter pair
            # -------------------------------------------------
            # Note: The following plt commands should only be used when running this notebook interactively. 
            # ! Warning:
            # ! Do not leave plotting enabled when executing the notebook via the API.  
            # ! Each call to plt will attempt to re-render the full-size floorplan image, 
            # ! which can significantly impact performance and memory usage.
            vis = np.zeros_like(output_img)
            # cv2.drawContours(image, contours, contourIdx, color, thickness)
            #   contourIdx = -1 draws all provided contours
            cv2.drawContours(vis, [filtered], -1, color, 5)
            # plt.subplot(len(epsilon_values), len(gap_values), plot_idx)
            # plt.imshow(vis)
            # plt.title(f"Eps: {epsilon_mult:.3f}, Gap: {gap_mult:.3f}\nScore: {score:.2f}, ⊥: {perp_score:.2f}")
            plot_idx += 1

            # -------------------------------------------------
            # 4i) Early exit when geometry is good enough
            # -------------------------------------------------
            # Accept when corners are mostly right-angled, or when combined score is high.
            if perp_score >= 0.90 or (score > 0.8 and perp_score >= 0.8):
            # if perp_score >= 0.99:
                print(f"✅ Perfect at ε={epsilon_mult:.3f}, gap={gap_mult:.3f}")
                # plt.tight_layout()
                # plt.show()
                return filtered, vis, perp_score

            # Track best-so-far candidate if it is reasonably orthogonal
            if score > best_score and perp_score > 0.5:
                best_score = score
                best_filtered_points = filtered.copy()


            # -------------------------------------------------
            # 4j) Last-chance acceptance at a specific epsilon
            # -------------------------------------------------
            if epsilon_mult == 0.02 and perp_score > 0.85:
                print(f"✅ LAST CHANCE Perfect at ε={epsilon_mult:.3f}, gap={gap_mult:.3f}")
                # plt.tight_layout()
                # plt.show()
                return filtered, vis, perp_score
            
            

                
        # End of current epsilon → try next if no perfect gap found



# ## ▭ [Floor] Detect Footprint Application

# In[18]:


# ============================================================
# Pipeline usage below: reading, preprocessing, and ROUND 2
# ============================================================

'''
    Floorplan contour refinement pipeline.

    This block takes a raw floorplan image, detects its outer contour, 
    evaluates perpendicularity, and performs a second pass ("ROUND 2") 
    to refine and straighten the contour into a rectilinear polygon. 
    The resulting contour is scaled and exported for downstream 
    geometry generation (e.g., wall meshes).

    Requires:
        - numpy as np
        - cv2 (OpenCV)
        - from PIL import Image
        - from IPython.display import display
        - helper functions:
            - detectOuterContours(gray, blank_image, color)
            - straighten_rectilinear(contour, length_threshold)
'''

# Read floorplan image (uncomment and set your path)
# img_path = "Images/example2.png"
# img = cv2.imread(img_path)  # BGR image
# display(Image.fromarray(img))  # quick preview


# Create a blank color canvas for drawing results
height, width, channels = img.shape
blank_image = np.zeros((height,width,3), np.uint8)


# Grayscale image
# BCZ: OpenCV operations in your pipeline (thresholding, morphology, contour detection) work best, or only work, on single-channel images.
# img is 3-channel (BGR) -> Grayscale condenses it to 1 channel (structure detection (lines, walls) we only care about intensity, not color).
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# ============================================================
# Detect outer contour and get a perpendicularity reference
contour, img, perp_score = detectOuterContours(gray, blank_image, color=(255,255,255))
print("detectOuterContours output")
# display(Image.fromarray(img)) # quick preview of detectOuterContours output
# ============================================================


# ========================= ROUND 2 ==========================
# Now get the contour from the contour (cleaning up the contour)

# Convert to grayscale for contour detection
bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# After finding contours
contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)  # Get largest contour

# Refine approximation based on perpendicularity score we got from detectOuterContours
# Adapt epsilon multiplier heuristically from the measured perp_score.
print("perp_score", perp_score)
if perp_score >= 0.99:
    epsilon_mult = 0.0001
elif perp_score >= 0.9:
    epsilon_mult = 0.0035
elif perp_score >= 0.8:
    epsilon_mult = 0.008
else:
    epsilon_mult = 0.005  # or even higher for very noisy shapes

# Again, approximate the contour to get straight lines (with tweaking the epsilon)
epsilon = epsilon_mult * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)

# Draw refined result
result_img = np.zeros_like(bw)
cv2.drawContours(result_img, [approx], -1, (255,255,255), 2)
# print("ROUND 2")
# display(Image.fromarray(result_img))

# -----------------------
# Applying straighten_rectilinear to the contour
# -----------------------
# Use this approximated contour for floor_boxes
contour = approx
# Apply snapping twice with different thresholds to progressively clean stair-steps
vis_image = np.zeros_like(bw)
contour = straighten_rectilinear(contour,15)
contour = straighten_rectilinear(contour,8) # Twice to clean both stair-steps and small gaps
# --------------------------

# Visualize snapped footprint
floor_contour = contour
cv2.drawContours(vis_image, [contour], -1, (255, 255, 255), 2)
# display(Image.fromarray(vis_image))


# -----------------------
# Scale and export points
# -----------------------
# Reminder: OpenCV uses image coordinates (x right, y down). If you convert to a metric world,
#           remember to handle axis directions and origin appropriately.
print("Before scaling:", [[point[0][0], point[0][1]] for point in contour])

# Save scaled reference polygon for downstream consumers (divide by 100 as an example)
floor_boxes = [[point[0][0]/100, point[0][1]/100] for point in contour]
# print("After scaling:", floor_boxes)
# display(Image.fromarray(img))


# ## ▭ [Floor] Export to .txt

# In[19]:


# save_to_file(os.path.join(tmp_dir, "floor_vertices"), floor_boxes, True)


# # 🖼️ [Canvas] Detect canvas

# In[20]:


# --- Load image ---
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not read: {img_path}")

# --- Grayscale ---
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Binary mask (ALL SCREEN WHITE, 255, 255) ---
ret, thresh = cv2.threshold(grey, 255, 255, cv2.THRESH_BINARY_INV)
# display(Image.fromarray(thresh))

# --- Find largest contour (corners of white screen) ---
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)  

# --- Draw largest contour on a 1-channel canvas ---
result_img = np.zeros_like(grey)
cv2.drawContours(result_img, [largest_contour], -1, (255,255,255), 2) # color=255 since canvas is 1-channel
display(Image.fromarray(result_img))


# --- Export scaled vertices (Nx1x2 -> list[[x/100, y/100], ...]) ---
canvas_vertices = [[point[0][0]/100, point[0][1]/100] for point in largest_contour]


# ## 🖼️  [Canvas] Export to .txt

# In[21]:


# # Save to file
# save_to_file(os.path.join(tmp_dir, "canvas_vertices"), canvas_vertices, True)


# # 🧱 [Wall] Detect Walls
# 
# ▭ → masked → 🧱

# ## ⚙️ 🧱 [Wall] Detect Functions

# ### ⚙️ 🧱 Wall Filter

# In[22]:


def wall_filter(gray):
    """
    Extract potential wall structures from a grayscale floorplan image.

    The function applies thresholding and morphological operations 
    to isolate wall-like features, while reducing noise and separating 
    uncertain regions. Intended as a preprocessing step before 
    contour extraction or polygon approximation.

    @Param gray: np.ndarray
        Single-channel grayscale image of the floorplan.
    @Return np.ndarray
        Binary mask (same shape as input) where 'unknown' regions 
        (likely wall candidates) are highlighted in white on black background.
        
    Requires:
        - numpy as np
        - cv2 (OpenCV)
        - from PIL import Image
        - from IPython.display import display 
               
    @Source 
        Adapted from: https://github.com/grebtsew/FloorplanToBlender3d/tree/master/FloorplanToBlenderLib
    """

    # Threshold image: invert so walls (dark lines) become white foreground.
    # Uses fixed high threshold (230) combined with Otsu's method
    # for robustness against varying scan intensities.
    ret, thresh = cv2.threshold(gray,230,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # print("threshold in wall_filter") 
    # display(Image.fromarray(thresh)) # visualization for debugging
    img_main = thresh # alias kept for clarity in intermediate steps


    # Noise removal via morphological opening.
    # Opening = erosion → dilation. Removes isolated white pixels and 
    # thin noise while keeping larger wall structures intact.
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)

    # Dilate result to ensure wall structures are well-connected. 
    # Expands white regions, filling small gaps.
    sure_bg = cv2.dilate(opening,kernel,iterations=1)
    # display(Image.fromarray(sure_bg))

    # Distance transform: compute distance to nearest black pixel.
    # Highlights the centerlines of walls more strongly than edges.
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)


    # Threshold the distance transform to separate "sure foreground" 
    # (strong wall centers) from weaker wall areas.
    # The scaling factors (0.5, 0.2) control strictness.
    ret, sure_fg = cv2.threshold(0.5*dist_transform,0.2*dist_transform.max(),255,0)
    
    # Convert sure foreground to binary image (uint8).
    sure_fg = np.uint8(sure_fg)

    # Subtract strong foreground from background to obtain "unknown" regions:
    # areas where wall presence is uncertain (edges, junctions, gaps).
    unknown = cv2.subtract(sure_bg,sure_fg)
    # print("unknown in wall_filter")
    # display(Image.fromarray(unknown)) # visualization for debugging

    return unknown


# ### ⚙️ 🧱 ▢ create n x 4 vertices and faces
# 
# Generate 3D wall quads (Nx4 vertices) and a shared face index from 2D floorplan contours.
# 

# In[23]:


def create_nx4_verts_and_faces(boxes, height = 1, scale = 1, ground = 0):
    '''
    Create verts and faces
    @Param boxes,
    @Param height,
    @Param scale,
    @Return verts - as [[wall1],[wall2],...] numpy array, faces - as array to use on all boxes, wall_amount - as integer
    
    Use the result by looping over boxes in verts, and create mesh for each box with same face and pos
    See create_custom_mesh in floorplan code.

    Requires:
    - numpy as np
    - cv2 (OpenCV)
    - Blender mesh utility: `create_custom_mesh` (if exporting to Blender)

    @source https://github.com/grebtsew/FloorplanToBlender3d/blob/master/FloorplanToBlenderLib/detect.py
    '''
    wall_counter = 0
    verts = []

    for box in boxes:
        box_verts = []
        for index in range(0, len(box) ):
            temp_verts = []
            # Get current
            curr = box[index][0];

            # Next point (wrap around at the end)
            if(len(box)-1 >= index+1):
                next = box[index+1][0];
            else:
                next = box[0][0]; # link to first pos

            # Create 4 vertices for the wall quad:
            # (curr ground, curr top, next ground, next top)
            temp_verts.extend([(curr[0]/scale, curr[1]/scale, ground)])
            temp_verts.extend([(curr[0]/scale, curr[1]/scale, height)])
            temp_verts.extend([(next[0]/scale, next[1]/scale, ground)])
            temp_verts.extend([(next[0]/scale, next[1]/scale, height)])

            # add wall verts to verts
            box_verts.extend([temp_verts])

            # wall counter
            wall_counter += 1

        verts.extend([box_verts])

    # All wall quads share the same face index pattern
    faces = [(0, 1, 3, 2)]
    return verts, faces, wall_counter


# ### ⚙️ 🧱 + 💠 Detect Precise Boxes  Post-processing and Straightenning the walls
# 
# Detect and rectify polygonal boxes with iterative rectilinear snapping for high-precision floorplan corner extraction.
# 

# In[24]:


def detectPreciseBoxes(detect_img, output_img = None, color = [100,100,0]):
    """
    Detect corners with boxes in image with high precision
    @Param detect_img image to detect from @mandatory
    @Param output_img image for output
    @Param color to set on output
    @Return corners(list of boxes), output image

    Requires:
    - numpy as np
    - cv2 (OpenCV)
    - matplotlib.pyplot as plt
    - from PIL import Image  
    - from IPython.display import display
    - helper functions: straighten_rectilinear(contour, length_threshold)
    
    Notes:
        - This function is adapted and extended from FloorplanToBlender3d 
          (original: detect.py). 
        - Modified here to use a **two-stage rectilinear snapping** pipeline 
          for robustness in scanned/noisy floorplans.

    @source https://github.com/grebtsew/FloorplanToBlender3d/blob/master/FloorplanToBlenderLib/detect.py
    """
    res = []

    contours, hierarchy = cv2.findContours(detect_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #area = sorted(contours, key=cv2.contourArea, reverse=True)

    largest_contour_area = 0
    for cnt in contours:
        largest_contour = cnt

        epsilon = 0.001*cv2.arcLength(largest_contour,True)
        approx = cv2.approxPolyDP(largest_contour,epsilon,True)
        if output_img is not None:
            final = cv2.drawContours(output_img, [approx], 0, color)
        res.append(approx)

    res1 = []
    for r in res:
        res1.append(straighten_rectilinear(r,2))
    # display(Image.fromarray(res2))
    
    res2 = []
    for r in res1:
        res2.append(straighten_rectilinear(r,1))
    
    res = res2

    return res, output_img


# ## 🧱[Wall] Detect Application

# In[25]:


'''
    Generate wall data file for floorplan.

    This pipeline extracts the outer boundary (mask), filters potential walls, 
    detects precise wall contours, and converts them into 3D-ready 
    vertices and faces for later mesh generation.

    @Param img_path str 
        Path to the input floorplan image (PNG/JPG).
    @Param info bool
        If True, intermediate data and steps are printed/displayed.
    @Return shape 
        Processed contour and wall geometry suitable for export.

    Requires:
        - numpy as np
        - cv2 (OpenCV)
        - from PIL import Image
        - from IPython.display import display
        - helper functions: wall_filter(), detectPreciseBoxes(), 
          create_nx4_verts_and_faces()
'''


# Read floorplan image
img = cv2.imread(img_path)



# ================================ MASK ================================
# Create a binary mask from the initial contour
mask = np.zeros((height, width), dtype=np.uint8)

# Get the floorData contour
contour = contour  # (assumed already available from prior step)
# contour = get_floor_boundary(img)  # alternative approach if automated

print("contour",contour)

# Slightly expand the contour outward by shifting relative to centroid.
# This helps close small gaps and ensures mask covers full boundary.
scaled_contour = contour + np.sign(contour - np.mean(contour, axis=0)) * np.array([[[2, 2]]]) 
scaled_contour = np.array(scaled_contour, dtype=np.int32)  # Convert to int32

print("scaled_contour",scaled_contour)

# Draw filled contour on mask
cv2.drawContours(mask, [scaled_contour], -1, (255), -1)  # -1 means "fill inside"

# Apply mask to original image
masked_img = cv2.bitwise_and(img, img, mask=mask)
# Make outside white
white_background = np.full_like(img, 255)  # full white canvas
inv_mask = cv2.bitwise_not(mask)  # invert mask (background now white)
white_outside = cv2.bitwise_and(white_background, white_background, mask=inv_mask)
masked_img = cv2.add(masked_img, white_outside)  # combine masked floorplan with white outside


# Optional: Display the mask and masked image
display(Image.fromarray(mask))  # Show mask
floor_mask_test = mask
display(Image.fromarray(masked_img))  # Show masked image
print("MASK Done")

# ================================ GRAYSCALE ================================
# Convert masked image to grayscale for morphology/contours
gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
display(Image.fromarray(gray))

# ================================ CLOSE ================================
# Morphological closing: dilation followed by erosion
# Purpose: fills small holes and gaps in the walls
kernel = np.ones((5,5),np.uint8)
closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
display(Image.fromarray(closed))



# ================================ WALL FILTER ================================
# Extract walls using custom wall_filter (threshold + morphology + distance transform)
# Helps remove small artifacts and emphasize wall structures
wall_img = wall_filter(closed)
# display(Image.fromarray(wall_img))


# ================================ WALL DETECTION ================================
# Detect contours of walls with high precision
wall_boxes, img = detectPreciseBoxes(wall_img)

display(Image.fromarray(wall_img))


# ================================ VERTS & FACES ================================
# Create 3D-ready vertices and faces from detected wall boxes

verts = []   # list of wall vertices
faces = []   # list of face definitions
wall_height = 1  # extrusion height of walls
scale = 100    # pixel-to-world scaling factor

# Convert boxes (2D polygons) into 3D quads (nx4 faces)
verts, faces, wall_amount = create_nx4_verts_and_faces(wall_boxes, wall_height, scale)

print("verts",verts)
print("wall_amount",wall_amount)  
print("wall_boxes",wall_boxes)


# Debug loop through wall boxes
box = []
wall = []
for box in wall_boxes:
    print("box",box)


# ## 🧱[Wall]: clean small walls

# In[26]:


from shapely.geometry import Polygon

def wall_edges_to_polygon(wall):
    """
        Convert a wall (list of edges) into a Shapely polygon.

        Each wall is represented as a list of edges, where an edge is
        [[x0, y0], [x1, y1]]. This function collects the starting
        point of each edge, then appends the final edge's endpoint to
        close the polygon.

        @Param wall list of edges (each edge = [[x0,y0],[x1,y1]]) @mandatory
        @Return shapely.geometry.Polygon object representing the wall shape

        @Requires:
            - shapely.geometry.Polygon
    """
    # Collect the first point of each edge
    points = [edge[0] for edge in wall]
     # Add the last endpoint to close the polygon loop
    points.append(wall[-1][1])
    # Convert list of points into a Shapely polygon
    return Polygon(points)

def clean_small_walls_shapely(wall_vertices, min_area=0.012):
    """
    Remove small wall polygons based on area threshold using Shapely.

    Each wall is represented as a list of edges. The function converts
    these edges into a polygon (via wall_edges_to_polygon), computes the
    polygon's area, and keeps only those walls with area larger than
    `min_area`. Useful to clean noise or very small artifacts detected
    as walls in floorplan parsing.

    @Param wall_vertices list of walls, where each wall is represented 
           as a list of edges (edge = [[x0,y0],[x1,y1]]) @mandatory
    @Param min_area float, minimum polygon area threshold (default: 0.012) @optional
    @Return list of walls that passed the area threshold

    @Requires:
        - shapely.geometry.Polygon
        - wall_edges_to_polygon(wall) helper function
    """
    cleaned_walls = []  # store filtered walls
    for wall in wall_vertices:
        # Convert wall edges to a polygon
        poly = wall_edges_to_polygon(wall)
        area = poly.area

        # Keep only if larger than threshold
        if area > min_area:
            cleaned_walls.append(wall)
            print(f"Wall kept - Area: {area:.4f}")
        else:
            print(f"Wall removed - Area: {area:.4f}")
    return cleaned_walls

# Preview the input wall image
display(Image.fromarray(wall_img))

# Create Wall Vertices
wall_vertices = create_vertices(wall_boxes)
# Clean Small Wall (noise)
cleaned_walls = clean_small_walls_shapely(wall_vertices)

# Preview the cleaned wall image
plot_vertices(cleaned_walls,[])

# now we have the cleaned wall vertices
wall_vertices = cleaned_walls


# ### 🧱[Wall] Export to .txt

# In[27]:


# wall_vertices = create_vertices(wall_boxes)

print("wall_vertices:", wall_vertices )

print("len(wall_vertices)",len(wall_vertices))



# save_to_file(os.path.join(tmp_dir, "walls_vertices"), wall_vertices, True)


# # 🤖 CubiCasa Pre-Trained Object/Room Segmentation          
# # (Deep Neural Network)

# ## 🤖 CubiCasa Run The Model

# In[28]:


'''
    Floorplan Semantic Segmentation Pipeline (Furukawa-based Model)

    This script:
        - Loads a pre-trained floorplan segmentation model (modified Furukawa HG architecture).
        - Prepares input floorplan images for the model (resizing, normalization, tensor conversion).
        - Runs predictions across 4 rotations to increase robustness.
        - Splits outputs into junctions, rooms, and icons.
        - Extracts polygons for each predicted type (walls, rooms, windows, etc.).
        - Converts polygons into vertices/faces for 3D reconstruction.

    Requires:
        - Python 3.x
        - numpy as np
        - matplotlib (pyplot, image)
        - cv2 (OpenCV)
        - torch (PyTorch)
        - torch.nn.functional as F
        - from torch.utils.data import DataLoader
        - utils.loaders: FloorplanSVG, DictToTensor, Compose, RotateNTurns
        - utils.plotting: segmentation_plot, polygons_to_image, draw_junction_from_dict, discrete_cmap
        - utils.post_prosessing: split_prediction, get_polygons, split_validation
        - model.get_model
        - mpl_toolkits.axes_grid1.AxesGrid
        - helper: create_nx4_verts_and_faces, scale_point_to_vector (for 3D conversion)

'''
import sys
sys.path.append("..") # Adds higher directory to python modules path.

# Import library
# from utils.FloorplanToBlenderLib import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn.functional as F
import cv2 
from torch.utils.data import DataLoader

from model import get_model
from utils.loaders import FloorplanSVG, DictToTensor, Compose, RotateNTurns
from utils.plotting import segmentation_plot, polygons_to_image, draw_junction_from_dict,           discrete_cmap
discrete_cmap()
from utils.post_prosessing import split_prediction, get_polygons, split_validation
from mpl_toolkits.axes_grid1 import AxesGrid

# clever trick to rotate image and make model's prediction more robust
rot = RotateNTurns()
# telling model what to look for in the image
room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath",
                "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink",
                "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]


# prepare the model----------------------------
# 1. Get base architecture (empty/random weights)
model = get_model('hg_furukawa_original', 51) # Load the base model architecture
n_classes = 44 # Total classes model can predict (44 classes at the end of Neural Network)
split = [21, 12, 11]  #  split this 44 classes into 3 parts: (21 junctions (where the walls meet) + 12 for rooms + 11 icons)

# 2. Modify architecture by = adding/changing layers (conv4_ and upsample)
model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1) # added convolution layer to the model to make the model's 256 channels to 44 channels that I want to predict
model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4) # added upsampling layer all this to make the results x4 bigger to fit the image original size
# 3. THEN load pre-trained weights file (from Furukawa's model). run by CPU only
checkpoint = torch.load('model_best_val_loss_var.pkl', map_location=torch.device('cpu')) # CPU only


# Input Image
# [512x512x3]
#     ↓
# Furukawa Model
# (Many internal layers)
#     ↓
# Feature Maps
# [128x128x256]
#     ↓
# model.conv4_  👈 We added this!
# Input: 256 channels
# Output: 44 channels (our classes)
# [128x128x44]
#     ↓
# model.upsample 👈 We added this!
# Input: 44 channels
# Output: 44 channels (but 4x bigger)
# [512x512x44]

# Input:                  Output:
# [1 2]       →          [1 * * 2]
# [3 4]                  [* * * *]
#                        [* * * *]
#                        [3 * * 4]


# the reason for model.conv4_ =  torch.nn.Conv2d(....)
# ... and model.upsample = torch.nn.ConvTranspose2d(....)
# ... is to sure that we go through the Furukawa's model and IFFFF we got some new layer (conv4_ or upsample), we go through THAT TOO.

# 4. Load weights from Furukawa's model to the neural network edges
model.load_state_dict(checkpoint['model_state']) 

# 5. Set the model to evaluation mode (ready to predict from given input image)
model.eval()
# ------------------------------------------------------------





# prepare the image--------------------------------------------
# 1. get the image from path
# img_path = 'Images/my_example3.png' 

# 2. Create tensor for pytorch
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # correct color channels (RGB → BGR)
# Converts BGR → RGB (OpenCV uses BGR by default) BUT our model (Neural network) expect RGB format 
# Like flipping the image's color channels to the right order


# 3. Image Normalization to range (-1,1)
# Original → Divided by 255 → Multiply by 2 → Subtract 1
# 255      → 1.0            → 2.0           → 1.0
img = 2 * (img / 255.0) - 1


# Rearrange Dimensions
# from (h,w,3)--->(3,h,w) as model input dimension is defined like this
img = np.moveaxis(img, -1, 0)
# img = np.moveaxis(img, source=-1, destination=0)
# -1 means "last axis" (the 3 RGB channels)
# 0 means "move it to first position"
# so 
# Before: (height=400, width=600, channels=3)
# After:  (channels=3, height=400, width=600)


# ALL THIS IS BECAUSE pytorch expects the image in this format: (channels(3 RGB), height, width)

# Convert to pytorch, enable cuda (CPU only)
img = torch.tensor([img.astype(np.float32)]) # CPU only
# This line does THREE important things:
# 1. Before: Could be uint8 (0-255) or float64
# img.astype(np.float32)  # Convert to 32-bit float (bcz Why? PyTorch prefers float32 for: Memory efficiency + GPU compatibility + Numerical stability)

# 2. Add batch dimension (    [img]    ):
# Before: (3, height, width)
# After:  (1, 3, height, width)  # Added batch dimension
# Why? Neural networks expect batches:
# Even for one image, need batch format
# Batch size of 1 means one image at a time


# 3. Convert from NumPy array to PyTorch tensor
# numpy_array = [img.astype(np.float32)]
# torch_tensor = torch.tensor(numpy_array)


# It's like preparing food for a restaurant:
# 1. astype(np.float32)  → Use standard measuring units
# 2. [img]               → Put on serving tray (even for one dish)
# 3. torch.tensor()      → Transfer to restaurant's kitchen equipment


# So if we had a 400x600 image:
# Original:        (400, 600, 3)      # NumPy array, uint8
# After moveaxis:  (3, 400, 600)      # NumPy array, uint8
# After tensor:    (1, 3, 400, 600)   # PyTorch tensor, float32
#                  ↑  ↑  ↑    ↑
#                  │  │  │    └── Width
#                  │  │  └─────── Height
#                  │  └────────── Channels (RGB)
#                  └──────────── Batch size (1)

# this is perfect for PyTorch. (bcz our model was trained on PyTorch. + Weights are saved in PyTorch format)


# img = torch.tensor([img.astype(np.float32)]).cuda() # CUDA version
n_rooms = len(room_classes) # 12
n_icons = len(icon_classes) # 11

# 4. Predict the image
with torch.no_grad():  # Tell PyTorch we don't need gradients (Its FORMALITY ALWAYS 1st thing to do)
    # (bcz we are not training the model. we are just predicting.)

    # Check if shape of image is odd or even
    size_check = np.array([img.shape[2],img.shape[3]])%2
    height = img.shape[2] - size_check[0]
    width = img.shape[3] - size_check[1]
    # Example:
    # img.shape = (1, 3, 401, 601)  # Odd dimensions
    # size_check = [1, 1]           # Both odd
    # height = 401 - 1 = 400        # Make even
    # width = 601 - 1 = 600         # Make even

    # Why do we need EVEN dimensions?
    # 1. Neural Network Requirements
    # Some operations (like certain strides and pooling) work better with even dimensions
    # Prevents rounding issues during upsampling/downsampling

    # 2. Rotation Consistency
    # With odd dimensions:
    # 401 x 601 → rotate 90° → 601 x 401 → rotate back
    # Might lose or shift a pixel!
   
    # With even dimensions:
    #  400 x 600 → rotate 90° → 600 x 400 → rotate back

    # Clean rotations, no pixel issues
    # Odd dimensions:  7x7
    # ⬜⬜⬜⬜⬜⬜⬜
    # ⬜⬜⬜⬜⬜⬜⬜
    # ⬜⬜⬜⬜⬜⬜⬜
    # ⬜⬜⬜⬛⬜⬜⬜  <- Center pixel causes issues in rotation
    # ⬜⬜⬜⬜⬜⬜⬜
    # ⬜⬜⬜⬜⬜⬜⬜
    # ⬜⬜⬜⬜⬜⬜⬜

    # Even dimensions:  6x6
    # ⬜⬜⬜⬜⬜⬜
    # ⬜⬜⬜⬜⬜⬜
    # ⬜⬜⬛⬛⬜⬜  <- Clean center rotation
    # ⬜⬜⬛⬛⬜⬜
    # ⬜⬜⬜⬜⬜⬜
    # ⬜⬜⬜⬜⬜⬜
    img_size = (height, width)


    # ------------Rotation Loop------------

    rotations = [(0, 0),   # No rotation (original)
             (1, -1),  # 90° clockwise, then back -90°
             (2, 2),   # 180°, then back 180°
             (-1, 1)]  # 270° clockwise, then back -270°
    pred_count = len(rotations) # 4

    prediction = torch.zeros([pred_count, n_classes, height, width])
    # Create Empty Prediction Storage
    # Shape: (4, 44, height, width)
    #        ↑  ↑   ↑       ↑
    #        |  |   |       └── Image width
    #        |  |   └────────── Image height
    #        |  └──────────────  44 classes (21 junctions + 12 rooms + 11 icons)
    #        └─────────────────  4 rotations

    # The Rotation Loop (4 times)
    for i, r in enumerate(rotations):
        forward, back = r
        # 1. Rotate Image
        rot_image = rot(img, 'tensor', forward)
        # Example: 400x600 image rotated 90° becomes 600x400

        # 2. [!!!!!] RUN Model
        pred = model(rot_image)
        # pred shape: (1, 44, height, width)
        #              ↑  ↑   ↑       ↑
        #              │  │   │       └── Each pixel gets predictions (44 classes)
        #              │  │   └────────── For each pixel in height
        #              │  └──────────────  44 predictions per pixel:
        #              │                   - 21 for junctions
        #              │                   - 12 for room types
        #              │                   - 11 for icons (doors, windows)
        #              └─────────────────  Batch size (1)

        # 3. Rotate Predictions Back
        pred = rot(pred, 'tensor', back)  # Rotate whole tensor
        pred = rot(pred, 'points', back)  # Rotate point coordinates in heatmaps


        # 4. Doublecheck Size Matches still
        pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
        # 5. Store Prediction
        prediction[i] = pred[0]

    # ------------Rotation Loop------------

# example for image size 400x600 and 44 classes:
 #  [!!!] the prediction tensor has shape [4, 44, 400, 600]   (4 rotations, 44 classes, 400 height, 600 width)
# print("before mean len(prediction)",len(prediction))

# Combines all 4 rotation predictions into 1 best prediction (by averaging)
prediction = torch.mean(prediction, 0, True)
# print("after mean len(prediction)",len(prediction))
#  [!!!] the prediction tensor has shape [1, 44, 400, 600]

# It's A 4D TENSOR:
# [1, 44, 400, 600]
# ↑  ↑   ↑       ↑
# |  |   |       └── Each pixel gets predictions (44 classes)
# |  |   └────────── For each pixel in height
# |  └──────────────  44 predictions per pixel:
# └─────────────────  Batch size (1)

# 44 prediction layers (1 prediction FOR EACH entire 400x600 image) 
# (1 only Walls prediction of 400x600 image, another 1 is for rooms for 400x600 image, another 1 is for icons for 400x600 image,....)
#       ↗                             
#     ↗     ┌────────────────────┐ 
#   ↗      /│                   /│
#         / │                  / │
#        /  │                 /  │
#       /   │                /   │
#      /    │               /    │
#     ┌────────────────────┐     │
#     │     │              │     │
#     │     │              │     │ 400 pixels
#     │     │              │     │  height
#     │     │              │     │
#     │     │              │     │
#     │     └──────────────│─────┘
#     │    /               │    /
#     │   /                │   /
#     │  /                 │  /
#     │ /                  │ /
#     │/                   │/
#     └────────────────────┘
#          600 pixels
#           width

#  44 layers (types)
#     ↓
#     [Layer 0: Junction type 0]        z
#     ┌──────────────────┐     ↑    
#     │0.99998 0.99998...│     │   
#     │0.99998 0.99998...│ 400 │   
#     │...     ...    ...│     │   
#     └──────────────────┘     ↓    
#          600 pixels →     

#     [Layer 1: Junction type 1]
#     ┌──────────────────┐
#     │0.89123 0.88234...│
#     │0.87654 0.89012...│
#     │...     ...    ...│
#     └──────────────────┘

#     [Layer 21: Room - Background]
#     ┌──────────────────┐
#     │0.77123 0.78901...│
#     │0.76543 0.77890...│
#     │...     ...    ...│
#     └──────────────────┘

#     [Layer 23: Room - Wall]
#     ┌──────────────────┐
#     │0.95432 0.96789...│
#     │0.94567 0.95678...│
#     │...     ...    ...│
#     └──────────────────┘

#     [Layer 34: Icon - Window]
#     ┌──────────────────┐
#     │0.12345 0.13456...│
#     │0.11234 0.12345...│
#     │...     ...    ...│
#     └──────────────────┘

#     [Layer 35: Icon - Door]
#     ┌──────────────────┐
#     │0.23456 0.24567...│
#     │0.22345 0.23456...│
#     │...     ...    ...│
#     └──────────────────┘
    
#     ...and so on until Layer 43

# Each layer is a complete 400x600 prediction map:
# - Layer 0-20: Junction predictions
# - Layer 21-32: Room predictions
# - Layer 33-43: Icon predictions


# =========== EXAMPLE OF rooms and icons ==============================
# For Rooms (12 types)
# Each pixel gets votes:
# Pixel(x,y) → [0.1, 0.8, 0.1] → argmax → "This is room type 1!"
rooms_pred = F.softmax(prediction[0, 21:21+12], 0).cpu().data.numpy()
rooms_pred = np.argmax(rooms_pred, axis=0)
# print("len(rooms_pred)",len(rooms_pred))
# print("rooms_pred",rooms_pred[100])
# REMEMBER:
# room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room", "Bath",
#                 "Entry", "Railing", "Storage", "Garage", "Undefined"]
#    Index:       0          1        2        3           4          5         6
#                 7          8        9        10         11
# 100th row of pixels show here
# [0 0 0 ... 2 2 2 ... 1 1 1 ... 2 2 2 ... 0 0 0]
# # Background(0) → Wall(2) → Outdoor(1) → Wall(2) → Background(0)
# i.e.
# Background  Wall        Outdoor Space           Wall    Background
# [0000000] [22222222] [111111111111111...] [22222222] [0000000]
# ------------------------------------------------------------
# i.e.
# Think of it like a horizontal slice through a floor plan:
# Background  Wall   Outdoor Space   Wall  Background
#     ↓        ↓          ↓           ↓       ↓
#     0000  ████████  ░░░░░░░░░░  ████████  0000
#     (0)     (2)        (1)         (2)     (0)

# For Icons (11 types)
icons_pred = F.softmax(prediction[0, 21+12:], 0).cpu().data.numpy()
icons_pred = np.argmax(icons_pred, axis=0)
# print("len(icons_pred)",len(icons_pred))
# print("icons_pred",icons_pred[170])
# REMEMBER:
# icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink",
#                 "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]
#    Index:       0          1        2        3           4          5         6
#                 7          8        9        10         11
# 170th row of pixels show here
# [0 0 0 ... 2 2 2 ... 1 1 1 ... 2 2 2 ... 0 0 0]
# SAME SAME SAME SAME SAME SAME SAME SAME   
# ================================================================ 



# We split these predictions into 3 parts:
heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
# # Remember the shapes:
# heatmaps: (21, height, width)  # 21 junction types
# rooms:    (12, height, width)  # 12 room types
# icons:    (11, height, width)  # 11 icon types
# print("heatmaps",heatmaps[20]) # 0-20 (21 junction types)
# print("rooms",rooms[11]) # 0-11 (12 room types)
# print("icons",icons[10]) # 0-10 (11 icon types)
# ================================================================ 




# We use these heatmaps, rooms, icons to get the polygons locations of them on the image []
polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), 0.2, [1, 2])

# REMEMBER:
# room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room", "Bath",
#                 "Entry", "Railing", "Storage", "Garage", "Undefined"]
# icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink",
#                 "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]


# print("polygons",polygons[0])
# polygons[0] = [[x1,y1], [x2,y2], [x3,y3], ...]  # List of coordinates forming a polygon
# print("types",types[0])
# types[0] = {'type': 'wall', 'class': 2}  # Dictionary describing what this polygon represents

# print("room_polygons",room_polygons[0])
# room_polygons[0] = "POLYGON ((124 211, 134 211, 136 211, ..., 124 211))"
# # This is a Shapely Polygon object represented as a string of coordinates
# # Each pair of numbers represents (x, y) coordinates that form the polygon
# print("room_types",room_types[0])
# room_types[0] = {'type': 'room', 'class': 2}  # Dictionary describing what this polygon represents
# ================================================================ 




# ## 🤖 ALL Labels Learnings
# Now we need an example image to work with.

# In[29]:


"""
    Visualize polygon-to-image segmentation results for rooms and icons.

    Uses `polygons_to_image` output (segmentation maps) and displays them 
    with discrete colormaps (`rooms`, `icons_furu`) registered beforehand 
    in plotting.py. Room segmentation is labeled with room_classes, 
    icon segmentation with icon_classes.

    @Requires:
        - matplotlib.pyplot as plt
        - numpy as np
        - polygons_to_image(polygons, types, room_polygons, room_types, height, width)
        - discrete_cmap() already called to register custom colormaps
        - room_classes, icon_classes lists
"""

# Convert polygons into segmentation masks
pol_room_seg, pol_icon_seg = polygons_to_image(polygons, types, room_polygons, room_types, height, width)


# ---------------- Room Segmentation ----------------
plt.figure(figsize=(12,12))
ax = plt.subplot(1, 1, 1)
ax.axis('off')

# Display room segmentation with registered 'rooms' colormap
rseg = ax.imshow(pol_room_seg, cmap='rooms', vmin=0, vmax=n_rooms-0.1)

# Add colorbar with room class labels
cbar = plt.colorbar(rseg, ticks=np.arange(n_rooms) + 0.5, fraction=0.046, pad=0.01)
cbar.ax.set_yticklabels(room_classes, fontsize=20)
plt.tight_layout()

# Save the plot as an image file
# plt.savefig('room_segmentation.png', dpi=300, bbox_inches='tight')

# # Display the saved image using cv2
# import cv2
# saved_img = cv2.imread('room_segmentation.png')
# cv2.imshow('Room Segmentation', saved_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.show()

# ---------------- Icon Segmentation ----------------
plt.figure(figsize=(12,12))
ax = plt.subplot(1, 1, 1)
ax.axis('off')

# Display icon segmentation with registered 'icons_furu' colormap
iseg = ax.imshow(pol_icon_seg, cmap='icons_furu', vmin=0, vmax=n_icons-0.1)

# Add colorbar with icon class labels
cbar = plt.colorbar(iseg, ticks=np.arange(n_icons) + 0.5, fraction=0.046, pad=0.01)
cbar.ax.set_yticklabels(icon_classes, fontsize=20)
plt.tight_layout()
# plt.show()  # Commented out for non-interactive execution


# ## 🤖 → get 🪟+🚪+🧱[window + door + wall] (unclean)

# In[30]:


"""
    Extract and convert polygons for windows, doors, and walls from detected polygons.

    Workflow:
        1. Get indices of polygons matching "Window" or "Door" from `icon_classes`.
        2. Collect those polygons into `boxes` format for mesh generation.
        3. Convert boxes into vertices, faces, and counts with `create_nx4_verts_and_faces`.
        4. Convert boxes into simplified 2D vertices using `create_vertices`.
        5. Save results for downstream usage (e.g., export to 3D viewer).
        6. Repeat same process for wall polygons (`types == 'wall'`).

    @Requires:
        - numpy as np
        - polygons (list of detected polygons)
        - types, room_types (class/type metadata for polygons)
        - icon_classes (list of icon labels)
        - wall_height (scalar, extrusion height for meshes)
        - scale (scalar, pixel-to-world scaling factor)
        - create_nx4_verts_and_faces(), create_vertices(), save_to_file()
"""
# ======= get windows polygons ====================================
# icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink",
#                 "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

# Find the index of "Window" in the icon_classes list
window_class_number = icon_classes.index("Window")
print(f"Window icon class number: {window_class_number}")

# Collect indices of polygons labeled as Window icons
window_polygon_numbers=[i for i,j in enumerate(types) if j['class']==icon_classes.index("Window")and (j['type']=='icon')]
print("window_polygon_numbers",window_polygon_numbers)


# Collect polygons into "boxes" format for mesh generation
boxes=[]
for i,j in enumerate(polygons):
    if i in window_polygon_numbers:
        temp=[]
        for k in j:
            temp.append(np.array([k]))
        boxes.append(np.array(temp))

# Generate 3D vertices/faces for window meshes
verts, faces, window_amount = create_nx4_verts_and_faces(boxes, wall_height, scale)


# print("verts",verts)
# print("boxes",boxes)
# print("len(boxes)",len(boxes)) 
# print("faces",faces)
# print("window_amount",window_amount)

# Convert polygons into simplified vertex edges
window_vertices = create_vertices(boxes)
print("window_vertices:", window_vertices )
print("len(window_vertices)",len(window_vertices))


# Save extracted window vertices to file for later use
# save_to_file(os.path.join(tmp_dir, "windows_vertices"), window_vertices, True)
print("--------------------------------")


# ---- scale point to vector ------
# # Create windows verts
# verts = []
# for box in boxes:
#     verts.extend([scale_point_to_vector(box, scale, 0)])

# # create faces
# faces = []
# for room in verts:
#     count = 0
#     temp = ()
#     for _ in room:
#         temp = temp + (count,)
#         count += 1
#     faces.append([(temp)])



# ======= get doors polygons ====================================
# icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink",
#                 "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

# print(icon_classes.index("Door"))
# Collect indices of polygons labeled as Door icons
door_polygon_numbers=[i for i,j in enumerate(types) if (j['class']==icon_classes.index("Door")) and (j['type']=='icon')]
print("door_polygon_numbers",door_polygon_numbers)



boxes=[]
for i,j in enumerate(polygons):
    if i in door_polygon_numbers:
        temp=[]
        for k in j:
            temp.append(np.array([k]))
        boxes.append(np.array(temp))

# Generate door meshes
verts, faces, door_amount = create_nx4_verts_and_faces(boxes, wall_height, scale)


# print("verts",verts)
# print("boxes",boxes)
# print("len(boxes)",len(boxes))
# print("faces",faces)
# print("door_amount",door_amount)

# Convert polygons into simplified vertex edges
door_vertices = create_vertices(boxes)

print("door_vertices:", door_vertices )

print("len(door_vertices)",len(door_vertices))



# ======= get walls polygons ====================================

# Collect indices of polygons labeled as walls
wall_type_polygon_numbers=[i for i,j in enumerate(types) if j['type']=='wall']
print("wall_type_polygon_numbers",wall_type_polygon_numbers)


boxes=[]
for i,j in enumerate(polygons):
    if i in wall_type_polygon_numbers:
        temp=[]
        for k in j:
            temp.append(np.array([k]))
        boxes.append(np.array(temp))


# Generate wall meshes 
verts, faces, wall_amount = create_nx4_verts_and_faces(boxes, wall_height, scale)



# print("verts",verts)
# print("boxes",boxes)
# print("len(boxes)",len(boxes))
# print("faces",faces)
# print("window_amount",wall_amount)

# ---- scale point to vector ------
# Create top walls verts
# verts = []
# for box in boxes:
#     verts.extend([scale_point_to_vector(box, scale, 0)])

# # create faces
# faces = []
# for room in verts:
#     count = 0
#     temp = ()
#     for _ in room:
#         temp = temp + (count,)
#         count += 1
#     faces.append([(temp)])


# # 💠 Post-Processing

# ## ⚙️💠 Post-Processing Helper Functions

# In[31]:


# Helper functions
def euclidean_distance(p1, p2):
    """
    Compute the straight-line (Euclidean) distance between two 2D points.

    @Param p1 list/tuple [x, y] @mandatory
        First point coordinates.
    @Param p2 list/tuple [x, y] @mandatory
        Second point coordinates.
    @Return float
        Distance between p1 and p2.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# For edge = [[x0, y0], [x1, y1]], 
def is_horizontal_edge(edge):
    """
    Check if an edge is approximately horizontal.

    @Param edge list [[x0, y0], [x1, y1]] @mandatory
        Two endpoints defining the edge.
    @Return bool
        True if the y-coordinates differ by less than 0.01 (≈ horizontal).
    """
    # distance btwn y0 - y1 
    return abs(edge[0][1] - edge[1][1]) < 0.01

def is_vertical_edge(edge):
    """
    Check if an edge is approximately vertical.

    @Param edge list [[x0, y0], [x1, y1]] @mandatory
        Two endpoints defining the edge.
    @Return bool
        True if the x-coordinates differ by less than 0.01 (≈ vertical).
    """
    # distance btwn x0 - x1
    return abs(edge[0][0] - edge[1][0]) < 0.01

def check_obj_alignment(door):
    """
    Determine whether a rectangular door structure is aligned horizontally or vertically.

    Logic:
        - A door is expected to have 4 edges.
        - Compares the lengths of the first two edges.
        - The longer edge determines the dominant orientation.
        - Falls back to 'neither' if not 4 edges are given.

    @Param door list of 4 edges [[[x0,y0],[x1,y1]], ...] @mandatory
    @Return str
        'horizontal', 'vertical', or 'neither'
    """
    if len(door) != 4:
        return 'neither'
    dist1 = euclidean_distance(door[0][0], door[0][1])
    dist2 = euclidean_distance(door[1][0], door[1][1])
    if dist1 > dist2:
        return 'horizontal' if is_horizontal_edge(door[0]) else 'vertical'
    else:
        return 'vertical' if is_vertical_edge(door[1]) else 'horizontal'
    


# ### ⚙️💠 ▭ → ▢ scale object down Function

# In[32]:


def scale_object(door, scale_factor=0.5):
    """
    Scale a rectangular door inward while keeping its center fixed.

    @Param door list of edges [[[x0,y0],[x1,y1]], ...] (expected 4 edges) @mandatory
    @Param scale_factor float factor to shrink along door's alignment (default=0.5) @optional
    @Return list of edges (scaled version of the door with same center alignment)
    """
    door_alignment = check_obj_alignment(door)
    center_x = sum(point[0] for edge in door for point in edge) / 8  # Average x of all points
    center_y = sum(point[1] for edge in door for point in edge) / 8  # Average y of all points

    scaled_door = []
    for edge in door:
        new_edge = []
        for point in edge:
            if door_alignment == "horizontal":
                # Only scale x-coordinates (scale down)
                new_x = (point[0] - center_x) * scale_factor + center_x
                new_edge.append([new_x, point[1]])
            else:  # vertical
                # Only scale y-coordinates (scale down)
                new_y = (point[1] - center_y) * scale_factor + center_y
                new_edge.append([point[0], new_y])
        scaled_door.append(new_edge)
    return scaled_door


# ## 💠 Post-Processing Functions
# 
# →  🔨 To detach the intersecting objects with Walls Segments 
# 
# →  📏 Snap the objects to closest Wall opening
# 
# →  X2 Clean Duplicates 
# 
# →  🚪→🪟 Replace intersecting door-windows with windows only
# 
# →  ✂️ Cutting the connected objects in equal parts.

# ## 💠 → 🔨 Detach intersection
# 
# Detach the Objects intersecting with Walls

# ### ⚙️💠🔨 Detach Function

# In[33]:


def detach_intersecting_objects(obj_vertices, wall_vertices):
    """
    Detach intersecting objects (windows/doors) from walls by snapping edges
    to intersections and reconstructing them as proper rectangles.

    The function:
      - Iterates through all detected objects
      - Uses door/window alignment (horizontal/vertical) to decide which edges 
        should be checked for intersections with wall edges
      - Adjusts object edge coordinates so that they snap to nearby wall intersections
      - Reconstructs valid rectangles from intersecting edges
      - Returns only objects that actually intersect walls

    @Param obj_vertices list of objects, each as 4 edges [[(x0,y0),(x1,y1)], ...]
    @Param wall_vertices list of wall segments, each segment is a list of edges
    @Return list of object edge groups, each snapped and reconstructed

    @Requires:
        - math (for copysign, sqrt via helper euclidean_distance)
        - shapely.geometry.LineString (to compute intersections)
        - Helper functions:
            - check_obj_alignment(obj)   → 'horizontal' / 'vertical' / 'neither'
            - is_horizontal_edge(edge)     → bool
            - is_vertical_edge(edge)       → bool
            - euclidean_distance(p1, p2)   → float
    """
    intersecting_objects = []   # Store only objects that intersect with walls

    # Loop over each object polygon (list of 4 edges)
    for obj in obj_vertices:
        alignment = check_obj_alignment(obj)  # Determine if object is horizontal/vertical
        intersecting_edges = []  # Collect snapped edges here

        # Loop through each edge of the object
        for edge in obj:
            if alignment == "horizontal":
                # Only consider horizontal edges for snapping
                if is_horizontal_edge(edge):
                    edge_line = LineString(edge)  # Convert edge to shapely line
                    intersected = False
                    
                    # Check for intersections with every wall edge
                    for wall_segment in wall_vertices:
                        for wall_edge in wall_segment:
                            wall_line = LineString(wall_edge)
                            if edge_line.intersects(wall_line):  # If they intersect
                                intersection = edge_line.intersection(wall_line)
                                # print("wall_line", wall_edge)
                                # print("edge_line", edge)
                                # print("intersection", intersection)

                                # Extract intersection point
                                intersection_x , intersection_y = intersection.coords[0][0], intersection.coords[0][1]
                                # print("intersection_x", intersection_x)
                                # print("intersection_y", intersection_y)
                                # print("edgex1[0][0]", edge[0][0])
                                # print("edgex2[1][0]", edge[1][0])
                                # print(" ")

                                # Decide which end of edge to move closer to intersection
                                if abs(intersection_x - edge[0][0]) < abs(intersection_x - edge[1][0]):
                                    edge[0][0] = intersection_x + (math.copysign(1, intersection_x - edge[0][0]) * 0.05)
                                else:
                                    edge[1][0] = intersection_x + (math.copysign(1, intersection_x - edge[1][0]) * 0.05)

                                intersected = True

                                # break
                            
                        
                        # if intersected:
                        #     break

                    # if intersected:
                    # Keep the adjusted edge if intersected
                    intersecting_edges.append(edge)
                        # break

                        
                # elif is_vertical_edge(edge):
                #     # print("intersecting_edges", intersecting_edges[-1])
                    
                #     print("intersecting_edges H-> Vertical EDGE", intersecting_edges)
                #     print("edge", edge)
                #     print("edge[0][0]", edge[0][0])
                #     print("edge[1][0]", edge[1][0])
                #     print(" ")
                #     edge[0][0] = intersecting_edges[-1][0][0]
                #     edge[1][0] = intersecting_edges[-1][1][0]
                #     # print("edge", edge)
                #     # intersecting_edges.append(edge)
            

            # SAME EXACT THING but for Vertical Alignment
            elif alignment == "vertical":
                # Only consider vertical edges for snapping
                if is_vertical_edge(edge):
                    edge_line = LineString(edge)
                    intersected = False

                    # Check for intersections with every wall edge
                    for wall_segment in wall_vertices:
                        for wall_edge in wall_segment:
                            wall_line = LineString(wall_edge)
                            if edge_line.intersects(wall_line):
                                intersection = edge_line.intersection(wall_line)
                                # print("wall_line", wall_edge)
                                # print("edge_line", edge)
                                # print("intersection", intersection)
                                
                                intersection_x , intersection_y = intersection.coords[0][0], intersection.coords[0][1]
                                # print("intersection_x", intersection_x)
                                # print("intersection_y", intersection_y)
                                # print("edgex1[0][1]", edge[0][1])
                                # print("edgex2[1][1]", edge[1][1])
                                # print(" ")

                                # Snap Y-coordinate to wall intersection
                                if abs(intersection_y - edge[0][1]) < abs(intersection_y - edge[1][1]):
                                    edge[0][1] = intersection_y + (math.copysign(1, intersection_y - edge[0][1]) * 0.05)
                                else:
                                    edge[1][1] = intersection_y + (math.copysign(1, intersection_y - edge[1][1]) * 0.05)

                                intersected = True

                                # break

                        # if intersected:
                        #     break

                    # if intersected:
                    intersecting_edges.append(edge)
                        # break
        

        # If no edges intersect walls, skip this object
        if len(intersecting_edges) == 0:
            # Either skip this object or handle non-intersecting case
            # Option 1: Skip
            continue
            # Option 2: Add the original object edges
            # intersecting_objects.append(obj)
                
       
        # Set the edges to be a rectangle
        
        # USE the shortest edge only
        # then rectangulate the edges
        # (i.e. If at least 2 edges intersect → reconstruct rectangle)
        if len(intersecting_edges) >= 2: 
            if is_horizontal_edge(intersecting_edges[0]):
                # Pick shortest horizontal edge as baseline
                if euclidean_distance(intersecting_edges[0][0], intersecting_edges[0][1]) <= euclidean_distance(intersecting_edges[1][0], intersecting_edges[1][1]):
                    # set shortest edge
                    # Align second edge with first, then rectangulate
                    intersecting_edges[1][1][0] = intersecting_edges[0][0][0]
                    intersecting_edges[1][0][0] = intersecting_edges[0][1][0]

                    # Rectangulate
                    intersecting_edges.append([intersecting_edges[0][1], intersecting_edges[1][0]])
                    intersecting_edges.append([intersecting_edges[1][1], intersecting_edges[0][0]])

                    # print("distance 0 <= 1")
                    # print("intersecting_edges[0][0]", intersecting_edges[0][0])
                    # print("intersecting_edges[0][1]", intersecting_edges[0][1])
                    # print("intersecting_edges[1][0]", intersecting_edges[1][0])
                    # print("intersecting_edges[1][1]", intersecting_edges[1][1])
                    
                    # print(" ")
                    # print("intersecting_edges", intersecting_edges)
                
                else:
                     # Align first edge with second
                    intersecting_edges[0][0][0] = intersecting_edges[1][1][0]
                    intersecting_edges[0][1][0] = intersecting_edges[1][0][0]

                    # Rectangulate
                    intersecting_edges.append([intersecting_edges[0][1], intersecting_edges[1][0]])
                    intersecting_edges.append([intersecting_edges[1][1], intersecting_edges[0][0]])
                    
            else:   # Vertical alignment
                if euclidean_distance(intersecting_edges[0][0], intersecting_edges[0][1]) <= euclidean_distance(intersecting_edges[1][0], intersecting_edges[1][1]):
                    intersecting_edges[1][1][1] = intersecting_edges[0][0][1]
                    intersecting_edges[1][0][1] = intersecting_edges[0][1][1]
                    intersecting_edges.append([intersecting_edges[0][1], intersecting_edges[1][0]])
                    intersecting_edges.append([intersecting_edges[1][1], intersecting_edges[0][0]])
                else:                
                    intersecting_edges[0][0][1] = intersecting_edges[1][1][1]
                    intersecting_edges[0][1][1] = intersecting_edges[1][0][1]
                    intersecting_edges.append([intersecting_edges[0][1], intersecting_edges[1][0]])
                    intersecting_edges.append([intersecting_edges[1][1], intersecting_edges[0][0]])
                
            if intersecting_edges:
                # Store reconstructed rectangle
                intersecting_objects.append(intersecting_edges)

    return intersecting_objects


# ### 💠🔨🪟 
# [Windows] Detach Application 

# In[34]:


# ============================================================
# Process detected window vertices: intersection cleanup & rectification
# ============================================================
print("INITIAL windows", window_vertices)

# Visualize current windows with walls
plot_vertices(wall_vertices, window_vertices)

# Example of detected window vertices before processing
# window_vertices = [[[[4.71, 2.02], [5.47, 2.02]], [[5.47, 2.02], [5.47, 2.2]], [[5.47, 2.2], [4.71, 2.2]], [[4.71, 2.2], [4.71, 2.02]]], [[[4.02, 2.02], [4.7, 2.02]], [[4.7, 2.02], [4.7, 2.2]], [[4.7, 2.2], [4.02, 2.2]], [[4.02, 2.2], [4.02, 2.02]]], [[[1.56, 2.02], [3.04, 2.02]], [[3.04, 2.02], [3.04, 2.2]], [[3.04, 2.2], [1.56, 2.2]], [[1.56, 2.2], [1.56, 2.02]]]]

print("window_vertices", window_vertices)


if window_vertices:
    # Step 1: Iteratively detach window edges intersecting with wall edges
    intersecting_win_vertices = detach_intersecting_objects(window_vertices, wall_vertices)
    intersecting_win_vertices = detach_intersecting_objects(intersecting_win_vertices, wall_vertices)
    intersecting_win_vertices = detach_intersecting_objects(intersecting_win_vertices, wall_vertices)

    # Step 2: If intersections were found → fix rectangles to ensure consistent shape
    if intersecting_win_vertices:
        Detached_windows = intersecting_win_vertices
    Detached_windows = fix_rectangles(Detached_windows)




    # Step 3: Plot walls + detached windows for visual verification
    plot_vertices(wall_vertices, Detached_windows)

    # Step 4: Save cleaned windows as updated window set
    print("Detached_windows", Detached_windows)
    window_vertices = Detached_windows





# ### 💠🔨🚪 
# [Doors-Shapely] Detach Application 

# In[35]:


# ============================================================
# Process detected door vertices: intersection cleanup & rectification
# ============================================================
print("INITIAL Doors", door_vertices)
# Visualize current door with walls
plot_vertices(wall_vertices, door_vertices)

print("door_vertices", door_vertices)

if door_vertices:
    # Step 1: Iteratively detach door edges intersecting with wall edges
    intersecting_door_vertices = detach_intersecting_objects(door_vertices, wall_vertices)
    intersecting_door_vertices = detach_intersecting_objects(intersecting_door_vertices, wall_vertices)
    intersecting_door_vertices = detach_intersecting_objects(intersecting_door_vertices, wall_vertices)

    # Step 2: If intersections were found → fix rectangles to ensure consistent shape
    if intersecting_door_vertices:
        Detached_doors = intersecting_door_vertices
    Detached_doors = fix_rectangles(Detached_doors)




    # Step 3: Plot walls + detached doors for visual verification
    plot_vertices(wall_vertices, Detached_doors)

    # Step 4: Save cleaned doors as updated doors set
    print("Detached_doors", Detached_doors)
    door_vertices = Detached_doors




# ## 💠 → 📏 Snapping
# 
# Snap the objects to closest Wall opening

# ### ⚙️💠📏 Snapping Functions

# In[36]:


def snap_edges_separately(edges, alignment, walls, obj_range):
    """
    Snap a pair of parallel edges (left–right or top–bottom) of a rectangular 
    object (obj/window) to the nearest wall segments in the floorplan.

    @Param edges       tuple of two edges [[x,y],[x,y]], representing either 
                       the left/right or top/bottom edges of the object
    @Param alignment   str: 'horizontal' (object spans left–right) OR 
                            'vertical' (object spans top–bottom)
    @Param walls       list of wall segments, where each wall is a list of edges
    @Param obj_range  tuple (min, max) range along the perpendicular axis to 
                       limit snapping search (ensures only relevant walls are used)

    @Return (edge1, edge2) new edges snapped to closest walls. If no suitable 
            wall found, original edges are returned.


    @Requires:
        - is_vertical_edge() and is_horizontal_edge() helpers must exist.
        - Input edges must represent a rectangle's parallel sides.
        - walls must be a list of segments with coordinates structured like [[x0,y0],[x1,y1]].

    @Note:
        - Within one rectangle: prevents snapping both edges to the same wall (via coord1 check).
        - Across rectangles: no global tracking, so walls can still be reused.
    """
    edge1, edge2 = edges
    obj_max = obj_range[1] + 0.05 # allow tolerance above range
    obj_min = obj_range[0] - 0.05 # allow tolerance below range
    if alignment == 'horizontal':
        # Horizontal: snap to vertical walls (x changes)
        a1, b1 = edge1[0]
        a2, b2 = edge2[0]

        # --- First edge snapping ---
        best1 = None
        min_dist1 = float('inf')  # track closest wall
        for wall_segment in walls:
            for wall_edge in wall_segment:
                if is_vertical_edge(wall_edge):
                    wall_coord = wall_edge[0][0]   # x-coordinate of vertical wall
                    wall_min = min(wall_edge[0][1], wall_edge[1][1])
                    wall_max = max(wall_edge[0][1], wall_edge[1][1])
                    # Check if wall spans the door's y-range
                    if wall_min <= obj_max and wall_max >= obj_min:
                        dist = abs(a1 - wall_coord)
                        direction = np.sign(a1 - wall_coord)
                        if dist < min_dist1:    # choose closest wall
                            min_dist1 = dist
                            best1 = (wall_edge, wall_coord, direction)
        if best1 is None:   # no wall found
            return edge1, edge2
        wall1, coord1, dir1 = best1
        new_edge1 = [[coord1, b1], [coord1, b2]]    # snap edge1 to wall x


        # --- Second edge snapping ---
        best2 = None
        min_dist2 = float('inf')
        for wall_segment in walls:
            for wall_edge in wall_segment:
                if is_vertical_edge(wall_edge):
                    wall_coord = wall_edge[0][0]
                    wall_min = min(wall_edge[0][1], wall_edge[1][1])
                    wall_max = max(wall_edge[0][1], wall_edge[1][1])
                    if wall_min <= obj_max and wall_max >= obj_min:
                        dist = abs(a2 - wall_coord)
                        direction = np.sign(a2 - wall_coord)
                        # ensure it's not the same wall and on opposite side
                        if wall_coord != coord1 and direction == -dir1:
                            if dist < min_dist2:
                                min_dist2 = dist
                                best2 = (wall_edge, wall_coord)
        if best2 is None:  # only one wall found
            return new_edge1, edge2
        wall2, coord2 = best2
        new_edge2 = [[coord2, b1], [coord2, b2]]
        return new_edge1, new_edge2
    
    
    elif alignment == 'vertical':
        # Vertical: snap to horizontal walls (y changes)
        a1, b1 = edge1[0]
        a2, b2 = edge2[0]

        # --- First edge snapping ---
        best1 = None
        min_dist1 = float('inf')
        for wall_segment in walls:
            for wall_edge in wall_segment:
                if is_horizontal_edge(wall_edge):
                    wall_coord = wall_edge[0][1]  # y-coordinate of horizontal wall
                    wall_min = min(wall_edge[0][0], wall_edge[1][0])
                    wall_max = max(wall_edge[0][0], wall_edge[1][0])
                    if wall_min <= obj_max and wall_max >= obj_min:
                        dist = abs(b1 - wall_coord)
                        direction = np.sign(b1 - wall_coord)
                        if dist < min_dist1:
                            min_dist1 = dist
                            best1 = (wall_edge, wall_coord, direction)
        if best1 is None:
            return edge1, edge2
        wall1, coord1, dir1 = best1
        new_edge1 = [[a1, coord1], [a2, coord1]]    # snap edge1 to wall y


        # --- Second edge snapping ---
        best2 = None
        min_dist2 = float('inf')
        for wall_segment in walls:
            for wall_edge in wall_segment:
                if is_horizontal_edge(wall_edge):
                    wall_coord = wall_edge[0][1]
                    wall_min = min(wall_edge[0][0], wall_edge[1][0])
                    wall_max = max(wall_edge[0][0], wall_edge[1][0])
                    if wall_min <= obj_max and wall_max >= obj_min:
                        dist = abs(b2 - wall_coord)
                        direction = np.sign(b2 - wall_coord)
                        if wall_coord != coord1 and direction == -dir1:
                            if dist < min_dist2:
                                min_dist2 = dist
                                best2 = (wall_edge, wall_coord)
        if best2 is None:
            return new_edge1, edge2
        wall2, coord2 = best2
        new_edge2 = [[a1, coord2], [a2, coord2]]
        return new_edge1, new_edge2
    
    # FALLBACK
    else:
        # If alignment is unknown, return edges unchanged
        return edge1, edge2


# ================== SNAP OBJECTS TO WALL OPENINGS ==================
def snap_object_sides_to_walls(object_vertices, wall_vertices):
    """
    Align rectangular objects (doors, windows, etc.) to the nearest wall openings
    by snapping their parallel edges (top-bottom or left-right) to detected wall lines.

    @Param object_vertices  list of objects, each represented as 4 edges 
                            [[[x0,y0],[x1,y1]], ...] (closed rectangle)
    @Param wall_vertices    list of walls, where each wall is a list of edges

    @Return list of updated objects (with edges snapped and reconstructed).

    @Requires:
        - `check_obj_alignment(obj)` → must return "horizontal", "vertical", or "neither"
        - `snap_edges_separately()` function → performs snapping of edge pairs
        - Object input must be 4 edges forming a rectangle
        - `numpy` (np) for math operations if used internally
    """
    updated_objs = []   # store aligned objects
    
    # Process each object
    for obj_no, obj in enumerate(object_vertices, 1):
        alignment = check_obj_alignment(obj)  # detect orientation of rectangle
        if alignment == 'neither':
            updated_objs.append(obj)  # keep unchanged if orientation unknown
            continue

        # Convert edges into editable lists
        new_obj = [list(edge) for edge in obj]
        
        # ---------------- Vertical Alignment ----------------
        # Calculate Object's range
        if alignment == 'vertical':
            obj_range = (
                min(obj[0][0][0], obj[0][1][0]),  # left-most x
                max(obj[0][0][0], obj[0][1][0])   # right-most x
            )
            # Snap bottom and top edges together
            bottom_edge, top_edge = obj[0], obj[2]
            new_bottom, new_top = snap_edges_separately(
                (bottom_edge, top_edge), 
                'vertical', 
                wall_vertices, 
                obj_range
            )
            new_obj[0] = new_bottom
            new_obj[2] = new_top
            
            
            # Reconstruct missing vertical edges from updated horizontals
            new_obj[1] = [new_obj[0][1], new_obj[2][0]]  # Right edge
            new_obj[3] = [new_obj[2][1], new_obj[0][0]]  # Left edge
            print("OUTPUT", new_obj)
            
        # ---------------- Horizontal Alignment ----------------
        else:  # horizontal
            # Range along y-axis (span of object vertically)
            obj_range = (
                min(obj[0][0][1], obj[0][1][1]),  # lowest y
                max(obj[0][0][1], obj[0][1][1])   # highest y
            )

            # Snap left and right edges to walls
            left_edge, right_edge = obj[1], obj[3]
            new_left, new_right = snap_edges_separately(
                (left_edge, right_edge),
                'horizontal',
                wall_vertices,
                obj_range
            )
            new_obj[1] = new_left
            new_obj[3] = new_right
            
            # Reconstruct missing horizontal edges from updated verticals
            new_obj[0] = [new_obj[1][1], new_obj[3][1]]  # Top edge
            new_obj[2] = [new_obj[3][0], new_obj[1][0]]  # Bottom edge
            print("OUTPUT", new_obj)
        
        updated_objs.append(new_obj)  # save updated rectangle
    
    return updated_objs







# ### 💠📏🪟 
# [Windows] Snapping Application 

# In[37]:


# ============================================================
# Process detected window vertices: scale → snap → rectify
# ============================================================

# Step 0: Visualize initial windows before modification
print("FIRST PLOT of Detached_windows")
plot_vertices(wall_vertices, window_vertices)

# Detached_windows = fix_rectangles(window_vertices)
# print("FIX RECTANGLE")
# plot_vertices(wall_vertices, Detached_windows)

# ------------------------------------------------------------
# Step 1: Scale windows slightly inward (shrink around center)
# ------------------------------------------------------------
for i, window in enumerate(window_vertices):

    # Apply uniform inward scaling to each window
    scaled_window = scale_object(window, 0.8)
    
    # Build updated edge list for this window
    new_window = []
    for window_edge in scaled_window:
        # DEBUG [scale]:
        new_window.append(window_edge)
                
    # Replace original window with its scaled version
    window_vertices[i] = new_window

# Save scaled set
scaled_windows = window_vertices
print("scaled_windows")
plot_vertices(wall_vertices, scaled_windows)


# ------------------------------------------------------------
# Step 2: Snap scaled windows to nearest wall segments
# ------------------------------------------------------------
snapped_windows = snap_object_sides_to_walls(scaled_windows, wall_vertices)

# Ensure snapped windows are consistent rectangles
snapped_windows = fix_rectangles(snapped_windows)
# snapped_windows = window_vertices
# print("\nUpdated window vertices:")
# for i, door in enumerate(snapped_windows, 1):
#     print(f"Window {i}: {door}")
print("SNAP TO WALLS")
plot_vertices(wall_vertices, snapped_windows)
# # print("snapped_windows", snapped_windows)

# ------------------------------------------------------------
# Step 3: Save final snapped windows
# ------------------------------------------------------------
window_vertices = snapped_windows


# ### 💠📏🚪 
# [Doors] Snapping Application 

# In[38]:


# ============================================================
# Process detected door vertices: scale → snap → rectify
# ============================================================

# Step 0: Visualize initial doors before modification
print("FIRST PLOT of Detached_doors")
plot_vertices(wall_vertices, door_vertices)

# Detached_doors = fix_rectangles(door_vertices)
# print("FIX RECTANGLE")
# plot_vertices(wall_vertices, Detached_doors)

# ------------------------------------------------------------
# Step 1: Scale doors slightly inward (shrink around center)
# ------------------------------------------------------------
for i, door in enumerate(door_vertices):

    # Apply uniform inward scaling to each door
    scaled_door = scale_object(door, 0.8)
    
    # Build updated edge list for this door
    new_door = []
    for door_edge in scaled_door:
        # DEBUG [scale]:
        new_door.append(door_edge)
                
    # Replace original door with its scaled version
    door_vertices[i] = new_door

# Save scaled set
scaled_doors = door_vertices
print("scaled_doors")
plot_vertices(wall_vertices, scaled_doors)


# ------------------------------------------------------------
# Step 2: Snap scaled doors to nearest wall segments
# ------------------------------------------------------------
snapped_doors = snap_object_sides_to_walls(scaled_doors, wall_vertices)

# Ensure snapped doors are consistent rectangles
snapped_doors = fix_rectangles(snapped_doors)
# snapped_doors = door_vertices
# print("\nUpdated door vertices:")
# for i, door in enumerate(snapped_doors, 1):
#     print(f"Door {i}: {door}")
print("SNAP TO WALLS")
plot_vertices(wall_vertices, snapped_doors)
# # print("snapped_windows", snapped_windows)

# ------------------------------------------------------------
# Step 3: Save final snapped doors
# ------------------------------------------------------------
door_vertices = snapped_doors


# ## 💠 → X2 Clean Duplicates 
# 
# Clean Duplicates of Objects as the result of snapping objects to same openings

# ### ⚙️💠X2 Clean Duplicates Functions

# In[39]:


def are_similar(set1, set2, tolerance):
    """
    Check if two sets of vertices (rectangles/polygons) are similar 
    by comparing their corner points within a distance tolerance.

    @Param set1, set2  list of edges (each edge = [[x0,y0],[x1,y1]]) @mandatory
    @Param tolerance   float max allowed distance between corresponding corners @mandatory
    @Return bool True if shapes are similar, False otherwise

    @Requires:
        - euclidean_distance(p1, p2) must exist to compute distances.
        - Both inputs must be lists of edges with consistent ordering.
    """
    # Extract corner points (first point of each edge)
    points1 = [edge[0] for edge in set1]
    points2 = [edge[0] for edge in set2]
    
    # Early exit: must have the same number of points
    if len(points1) != len(points2):
        return False
    
    # Compare each corner pair; if any are too far → not similar
    for p1, p2 in zip(points1, points2):
        if euclidean_distance(p1, p2) > tolerance:
            return False
    return True


# ============================================================
#                       clean_duplicates
# ============================================================
def clean_duplicates(door_vertices, tolerance=0.2):
    """
    Remove duplicate doors/objects from a list by checking similarity.

    @Param door_vertices  list of doors, where each door is a list of edges @mandatory
    @Param tolerance      float distance threshold for considering two doors duplicates @optional
    @Return list cleaned set of doors with duplicates removed

    @Requires:
        - are_similar(set1, set2, tolerance) must exist.
        - door_vertices must be structured consistently (list of list of edges).
    """
    door_no_duplicates = []
    for door in door_vertices:
        # Check if the current door is similar to any existing one
        for existing in door_no_duplicates:
            if are_similar(door, existing, tolerance):
                break
        else:
            # If no similar set is found, add the door
            door_no_duplicates.append(door)
    return door_no_duplicates


# ### 💠X2🪟
# [Windows] Clean duplicates Application 

# In[40]:


# Clean duplicates
no_dup_windows = clean_duplicates(window_vertices)
# Preview Results
print("CLEAN DUPLICATES")
plot_vertices(wall_vertices, no_dup_windows)
print("no_dup_windows\n", no_dup_windows)

# Fix Rectangles
no_dup_windows = fix_rectangles(no_dup_windows)
# Preview Results
print("FIX RECTANGLE")
plot_vertices(wall_vertices, no_dup_windows)


# save_to_file(os.path.join(tmp_dir, "windows_vertices"), snapped_windows, True)

window_vertices = no_dup_windows


# ### 💠X2🚪
# [Doors] Clean duplicates Application 

# In[41]:


# Clean duplicates
no_dup_doors = clean_duplicates(door_vertices)
# Preview Results
print("CLEAN DUPLICATES")
plot_vertices(wall_vertices, no_dup_doors)
print("no_dup_doors\n", no_dup_doors)

# Fix Rectangle
no_dup_doors = fix_rectangles(no_dup_doors)
# Preview Results
print("FIX RECTANGLE")
plot_vertices(wall_vertices, no_dup_doors)


# save_to_file(os.path.join(tmp_dir, "doors_vertices"), snapped_doors, True)
door_vertices = no_dup_doors


# ## 💠→ 🚪→🪟 
# 
# Replace intersecting door-windows with only windows 

# In[42]:


# ============================================================
# Detect and resolve collisions between doors and windows
# ============================================================

# Visualize current placement of doors and windows
plot_vertices(door_vertices, window_vertices)

def edges_to_polygon(edges):
    """
    Convert a list of edges (each edge = [[x0,y0],[x1,y1]]) into 
    a Shapely Polygon object.

    @Param edges list of edges forming a closed shape @mandatory
    @Return Polygon shapely polygon object

    @Requires:
        - shapely.geometry.Polygon must be imported.
        - Edges must form a closed loop or near-closed loop.
    """
    # Convert list of edges into a polygon by flattening and removing duplicates while keeping order
    coords = [tuple(pt) for edge in edges for pt in edge]
    
    # The coords list might repeat points for each edge, so we deduplicate in sequence
    unique_coords = []
    for c in coords:
        if not unique_coords or unique_coords[-1] != c:
            unique_coords.append(c)
    # Close polygon if not closed
    if unique_coords[0] != unique_coords[-1]:
        unique_coords.append(unique_coords[0])
    return Polygon(unique_coords)



# Convert all windows and doors into shapely Polygons
window_polygons = [edges_to_polygon(w) for w in window_vertices]
door_polygons = [edges_to_polygon(d) for d in door_vertices]


print("door_vertices", door_vertices)
print("window_vertices", window_vertices)

# ============================================================
# Collision detection between windows and doors
# ============================================================
collisions = []
for wi, w_poly in enumerate(window_polygons):
    for di, d_poly in enumerate(door_polygons):
        # Check if the window and door polygons intersect with a threshold of 0.5
        if w_poly.intersects(d_poly):
            collisions.append(door_vertices[di])
            # door_vertices.remove(door_vertices[di])

# Remove all door_vertices that are in collisions
door_vertices = [d for d in door_vertices if d not in collisions]



# save_to_file(os.path.join(tmp_dir, "doors_vertices"), door_vertices, True)


# save_to_file(os.path.join(tmp_dir, "windows_vertices"), window_vertices, True)


plot_vertices(door_vertices, window_vertices)


# ## 💠 →  ✂️ Cut equally
# 
# Cutting the connected objects in equal parts.

# ### ⚙️💠✂️ Cut equally Functions

# In[43]:


def split_large_window(window_vert, width_threshold=1.0):
    """
    Split a rectangular window into smaller equal-width windows if its 
    span exceeds a given width threshold.

    @Param window_vert   list of 4 edges representing the window rectangle @mandatory
    @Param width_threshold float width threshold for splitting (default: 1.0) @optional
    @Return list of windows, each in edge-list form. If no split needed → 
            returns the original window inside a list.

    @Requires:
        - check_obj_alignment(window) must exist to detect 'vertical' or 'horizontal' orientation.
        - euclidean_distance(p1,p2) helper must exist.
        - window_vert must represent a consistent closed rectangle of 4 edges.
    """
    alignment = check_obj_alignment(window_vert)
    
    # --- Step 1: Calculate width of window based on orientation ---
    if alignment == 'vertical':
        # Measure vertical window width along its right edge
        width = euclidean_distance(window_vert[1][0], window_vert[1][1])  # right edge
        is_wide = width > width_threshold
    else:  # horizontal
        # Measure horizontal window width along its top edge
        width = euclidean_distance(window_vert[0][0], window_vert[0][1])  # top edge
        is_wide = width > width_threshold
        
    if not is_wide:
        return [window_vert]   # Already within threshold → no split
    
    # --- Step 2: Decide how many splits needed ---
    n_splits = int(width / width_threshold)
    if width % width_threshold > 0:  # If there's a remainder, add one more split
        n_splits += 1  # Handle leftover by adding one extra split
        
    split_windows = []
    segment_width = width / n_splits  # Equal size segments
    
    # --- Step 3: Build each split window geometry ---
    for i in range(n_splits):
        if alignment == 'vertical':
            # Vertical → split along Y-axis
            y_top = window_vert[0][0][1]
            y_bottom = window_vert[2][0][1]
            segment_height = (y_top - y_bottom) / n_splits
            
            # Calculate y coordinates for this segment
            seg_top = y_bottom + (i + 1) * segment_height
            seg_bottom = y_bottom + i * segment_height
            
            # Recreate rectangle edges for this split
            new_window = [
                [[window_vert[0][0][0], seg_top], [window_vert[0][1][0], seg_top]],  # top edge
                [[window_vert[1][0][0], seg_top], [window_vert[1][1][0], seg_bottom]],  # right edge
                [[window_vert[2][0][0], seg_bottom], [window_vert[2][1][0], seg_bottom]],  # bottom edge
                [[window_vert[3][0][0], seg_bottom], [window_vert[3][1][0], seg_top]]  # left edge
            ]
            
        else:  # horizontal
            # For horizontal windows, we split along the x-axis
            x_left = window_vert[3][0][0]
            x_right = window_vert[1][0][0]
            
            # Calculate x coordinates for this segment
            seg_left = x_left + i * segment_width
            seg_right = x_left + (i + 1) * segment_width
            
            # Recreate rectangle edges for this split
            new_window = [
                [[seg_left, window_vert[0][0][1]], [seg_right, window_vert[0][1][1]]],  # top edge
                [[seg_right, window_vert[1][0][1]], [seg_right, window_vert[1][1][1]]],  # right edge
                [[seg_right, window_vert[2][0][1]], [seg_left, window_vert[2][1][1]]],  # bottom edge
                [[seg_left, window_vert[3][0][1]], [seg_left, window_vert[3][1][1]]]  # left edge
            ]
            
        split_windows.append(new_window)
    
    return split_windows


# ============================================================
# Function to process all windows and split if necessary
# ============================================================
def process_windows(window_list, width_threshold=1.0):
    """
    Apply window splitting to a list of windows. 
    Each wide window is divided into smaller windows.

    @Param window_list   list of windows (each window = list of 4 edges) @mandatory
    @Param width_threshold float splitting threshold (default: 1.0) @optional
    @Return list of windows, where oversized windows are replaced by multiple splits.

    @Requires:
        - split_large_window() must exist.
    """
    processed_windows = []
    for window in window_list:
        split_windows = split_large_window(window, width_threshold)
        processed_windows.extend(split_windows)  # append all splits (or the original)

    return processed_windows



# # Test with your example
# test_windows = [
    
#   [[[6.0800815, 0.88], [7.7099185, 0.88]], [[7.7099185, 0.88], [7.7099185, 0.76]], [[7.7099185, 0.76], [6.0800815, 0.76]], [[6.0800815, 0.76], [6.0800815, 0.88]]]
# , [[[10.23, 3.2598735], [10.33, 3.2598735]], [[10.33, 3.2598735], [10.33, 0.7301264999999999]], [[10.33, 0.7301264999999999], [10.23, 0.7301264999999999]], [[10.23, 0.7301264999999999], [10.23, 3.2598735]]]
# , [[[8.000107, 0.88], [10.139893, 0.88]], [[10.139893, 0.88], [10.139893, 0.76]], [[10.139893, 0.76], [8.000107, 0.76]], [[8.000107, 0.76], [8.000107, 0.88]]]
# , [[[8.16, 7.979945000000001], [8.26, 7.979945000000001]], [[8.26, 7.979945000000001], [8.26, 6.880055]], [[8.26, 6.880055], [8.16, 6.880055]], [[8.16, 6.880055], [8.16, 7.979945000000001]]]
# , [[[5.760054, 2.89], [6.839945999999999, 2.89]], [[6.839945999999999, 2.89], [6.839945999999999, 2.79]], [[6.839945999999999, 2.79], [5.760054, 2.79]], [[5.760054, 2.79], [5.760054, 2.89]]]
# , [[[10.23, 4.2699419999999995], [10.33, 4.2699419999999995]], [[10.33, 4.2699419999999995], [10.33, 3.110058]], [[10.33, 3.110058], [10.23, 3.110058]], [[10.23, 3.110058], [10.23, 4.2699419999999995]]]
# , [[[10.23, 7.979906000000001], [10.33, 7.979906000000001]], [[10.33, 7.979906000000001], [10.33, 6.1000939999999995]], [[10.33, 6.1000939999999995], [10.23, 6.1000939999999995]], [[10.23, 6.1000939999999995], [10.23, 7.979906000000001]]]
# , [[[9.370038999999998, 6.02], [10.149961000000001, 6.02]], [[10.149961000000001, 6.02], [10.149961000000001, 5.92]], [[10.149961000000001, 5.92], [9.370038999999998, 5.92]], [[9.370038999999998, 5.92], [9.370038999999998, 6.02]]]
# , [[[8.270038999999999, 6.03], [9.049961000000001, 6.03]], [[9.049961000000001, 6.03], [9.049961000000001, 5.91]], [[9.049961000000001, 5.91], [8.270038999999999, 5.91]], [[8.270038999999999, 5.91], [8.270038999999999, 6.03]]]

# ]





# ### 💠✂️🪟  
# [Windows] Cut Application 

# In[44]:


# ============================================================
# Final window post-processing: split, clean, and plot
# ============================================================

# Step 1: Split large windows into smaller ones if above threshold
processed_windows = process_windows(window_vertices, 1.0)


plot_vertices(wall_vertices, processed_windows)

# Print results
# print(f"Original number of windows: {len(test_windows)}")
print(f"Number of windows after processing: {len(processed_windows)}")


# save_to_file(os.path.join(tmp_dir, "windows_vertices"), processed_windows, True)


# Step 2: Save processed set as current windows
print("processed_windows")
print(processed_windows)
window_vertices = processed_windows

# Step 3: Ensure all windows are valid rectangles
window_vertices = fix_rectangles(window_vertices)

# Step 4: Final visualization of walls + cleaned windows
plot_vertices(wall_vertices, window_vertices)


# ### 💠✂️🚪  
# [Doors] Cut Application 

# In[45]:


# ============================================================
# Final window post-processing: split, clean, and plot
# ============================================================

# Step 1: Split large windows into smaller ones if above threshold
processed_doors = process_windows(door_vertices, 1.0)


plot_vertices(wall_vertices, processed_doors)

# Print results
# print(f"Original number of windows: {len(test_windows)}")
print(f"Number of windows after processing: {len(processed_doors)}")


# save_to_file(os.path.join(tmp_dir, "doors_vertices"), processed_doors, True)


# Step 2: Save processed set as current windows
print("processed_doors")
print(processed_doors)
door_vertices = processed_doors
# Step 3: Ensure all windows are valid rectangles
door_vertices = fix_rectangles(door_vertices)

# Step 4: Final visualization of walls + cleaned windows
plot_vertices(wall_vertices, door_vertices)


# # 💾🚪🪟🧱🖼️ ▭ Exporting post-processed data to .txt files
# 

# In[46]:


# 🚪
save_to_file(os.path.join(tmp_dir, "doors_vertices"), door_vertices, True)

# 🪟
save_to_file(os.path.join(tmp_dir, "windows_vertices"), window_vertices, True)

# 🧱
save_to_file(os.path.join(tmp_dir, "walls_vertices"), wall_vertices, True)

# 🖼️
save_to_file(os.path.join(tmp_dir, "canvas_vertices"), canvas_vertices, True)

# ▭
save_to_file(os.path.join(tmp_dir, "floor_vertices"), floor_boxes, True)


# Later will output it as a json file to test with the 3D viewer:

# output_data = {
#     "doors": processed_doors,
#     "floor": floor_boxes,
#     "walls": wall_vertices,
#     "windows": processed_windows,
#     "canvas": canvas_vertices
# }


# # 💾 RESULT JSON

# In[47]:


import json

output_data = {
    "doors": door_vertices,
    "floor": floor_boxes,
    "walls": wall_vertices,
    "windows": window_vertices,
    "canvas": canvas_vertices
}

with open(os.path.join(tmp_dir, "floorplan_data.json"), "w") as f:
    json.dump(output_data, f)


# # 🔬 IoU Testing 
# 
# Uncomment to see the results
# 
# 
# (as far as /Annotation Folder annotated images existing we can do the IoU Testing)
# 
# ![input](Images/iou-precision-recall.png)

# In[48]:


# import numpy as np
# import cv2
# import json

# import re
# import glob


# # example8_file now contains the path to Images/example8.png (or similar)

# # name = "flrpln1"
# name = main_img_path.split("Images/")[1].split(".")[0]
# print(name)


# # Load annotation
# with open(f"Images/annotation/{name}.json", "r") as f:
#     data = json.load(f)

# width = data["width"]
# height = data["height"]

# # Initialize masks
# wall_mask = np.zeros((height, width), dtype=np.uint8)
# door_mask = np.zeros((height, width), dtype=np.uint8)
# window_mask = np.zeros((height, width), dtype=np.uint8)
# floor_mask = np.zeros((height, width), dtype=np.uint8)

# # Process each object
# for obj in data.get("boxes", []):
#     label = obj.get("label")
    
#     # If it's a polygon (walls)
#     if obj.get("type") == "polygon" and "points" in obj and len(obj["points"]) >= 3:
#         points = np.array(obj["points"], dtype=np.int32).reshape((-1, 1, 2))
#         if label == "wall":
#             cv2.fillPoly(wall_mask, [points], color=255)
#         elif label == "floor":
#             cv2.fillPoly(floor_mask, [points], color=255)
#         elif label == "window":
#             cv2.fillPoly(window_mask, [points], color=255)
#         elif label == "door":
#             cv2.fillPoly(door_mask, [points], color=255)
#     elif all(k in obj for k in ["x", "y", "width", "height"]):
#         x = int(float(obj["x"]))
#         y = int(float(obj["y"]))
#         w = int(float(obj["width"]))
#         h = int(float(obj["height"]))
#         if label == "door":
#             cv2.rectangle(door_mask, (x, y), (x + w, y + h), 255, thickness=-1)
#         elif label == "window":
#             cv2.rectangle(window_mask, (x, y), (x + w, y + h), 255, thickness=-1)



# # Save masks
# cv2.imwrite(f"Images/annotation/{name}_wall.png", wall_mask)
# cv2.imwrite(f"Images/annotation/{name}_door.png", door_mask)
# cv2.imwrite(f"Images/annotation/{name}_window.png", window_mask)
# cv2.imwrite(f"Images/annotation/{name}_floor.png", floor_mask)

# display(Image.fromarray(wall_mask))
# display(Image.fromarray(door_mask))
# display(Image.fromarray(window_mask))
# display(Image.fromarray(floor_mask))


# In[49]:


# def compute_iou(pred, gt):
#     pred_bin = pred > 0
#     gt_bin = gt > 0
#     intersection = np.logical_and(pred_bin, gt_bin)
#     union = np.logical_or(pred_bin, gt_bin)
#     intersection_sum = intersection.sum()
#     union_sum = union.sum()
#     return intersection_sum / union_sum if union_sum != 0 else 0


# def compute_precision(pred, gt):
#     pred_bin = pred > 0
#     gt_bin = gt > 0
#     intersection = np.logical_and(pred_bin, gt_bin)
#     intersection_sum = intersection.sum()
#     pred_sum = pred_bin.sum()
#     return intersection_sum / pred_sum if pred_sum != 0 else 0


# def compute_recall(pred, gt):
#     pred_bin = pred > 0
#     gt_bin = gt > 0
#     intersection = np.logical_and(pred_bin, gt_bin)
#     intersection_sum = intersection.sum()
#     gt_sum = gt_bin.sum()
#     return intersection_sum / gt_sum if gt_sum != 0 else 0







# ## Wall Test

# In[50]:


# # Convert outline to filled polygon
# contours, _ = cv2.findContours(wall_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# filled_wall_img = np.zeros_like(wall_img)
# cv2.drawContours(filled_wall_img, contours, -1, 255, thickness=cv2.FILLED)


# # iou_score = compute_iou(wall_mask, filled_wall_img)
# iou_score = compute_iou(filled_wall_img, wall_mask)
# print(f"IoU Score: {iou_score:.4f}")
# # precision_score = compute_precision(wall_mask, filled_wall_img)
# precision_score = compute_precision(filled_wall_img, wall_mask)
# print(f"Precision Score: {precision_score:.4f}")
# # recall_score = compute_recall(wall_mask, filled_wall_img)
# recall_score = compute_recall(filled_wall_img, wall_mask)
# print(f"Recall Score: {recall_score:.4f}")

# # Show intersection overlay
# # Create a 3-channel RGB image for visualization
# overlay = np.zeros((wall_mask.shape[0], wall_mask.shape[1], 3), dtype=np.uint8)

# # Red for prediction only
# overlay[(wall_mask > 0) & (filled_wall_img == 0)] = [255, 0, 0]
# # Blue for ground truth only
# overlay[(filled_wall_img > 0) & (wall_mask == 0)] = [0, 0, 255]
# # Green for intersection
# overlay[(wall_mask > 0) & (filled_wall_img > 0)] = [0, 255, 0]

# display(Image.fromarray(overlay))


# ## Floor Test

# In[51]:


# # === Floor ===
# iou_score = compute_iou(floor_mask, floor_mask_test)
# print(f"IoU Score: {iou_score:.4f}")
# precision_score = compute_precision(floor_mask, floor_mask_test)
# print(f"Precision Score: {precision_score:.4f}")
# recall_score = compute_recall(floor_mask, floor_mask_test)
# print(f"Recall Score: {recall_score:.4f}")

# # Show intersection overlay
# # Create a 3-channel RGB image for visualization
# overlay = np.zeros((floor_mask.shape[0], floor_mask.shape[1], 3), dtype=np.uint8)

# # Red for prediction only
# overlay[(floor_mask > 0) & (floor_mask_test == 0)] = [255, 0, 0]
# # Blue for ground truth only
# overlay[(floor_mask_test > 0) & (floor_mask == 0)] = [0, 0, 255]
# # Green for intersection
# overlay[(floor_mask > 0) & (floor_mask_test > 0)] = [0, 255, 0]

# display(Image.fromarray(overlay))


# ## Windows Test
# 
# Vertices to image

# In[52]:


# import copy


# # 1. Find min and max x
# all_xs = [pt[0] for rect in processed_windows for line in rect for pt in line]
# min_x, max_x = min(all_xs), max(all_xs)

# # 2. Flip x for each point
# flipped_windows = copy.deepcopy(processed_windows)
# for rect in flipped_windows:
#     for line in rect:
#         for pt in line:
#             pt[0] = min_x + max_x - pt[0]

# # Now flipped_windows contains the horizontally flipped coordinates
# print(flipped_windows)



# In[53]:


# import numpy as np
# import cv2

# img = cv2.imread(main_img_path)

# img_height, img_width = img.shape[:2]
# window_mask_test = np.zeros((img_height, img_width), dtype=np.uint8)

# scale = 100  # adjust as needed

# for rect in processed_windows:
#     # Extract the 4 corner points (first point of each line)
#     points = [line[0] for line in rect]
#     # Convert to pixel coordinates
#     pixel_points = np.array([[int(round(x * scale)), int(round(y * scale))] for x, y in points], dtype=np.int32)
#     # Reshape for fillPoly
#     pixel_points = pixel_points.reshape((-1, 1, 2))
#     # Draw filled polygon
#     cv2.fillPoly(window_mask_test, [pixel_points], 255)

# # # print(window_mask_test)
# # # If using Jupyter, use:
# # # from matplotlib import pyplot as plt
# # # plt.imshow(window_mask_test, cmap='gray')
# # # plt.show()
# # display(Image.fromarray(window_mask_test))
# # display(Image.fromarray(window_mask))

# kernel = np.ones((3, 3), np.uint8)
# window_mask_test = cv2.dilate(window_mask_test, kernel, iterations=2)

# # === Windows ===
# # iou_score = compute_iou(window_mask, window_mask_test)
# iou_score = compute_iou(window_mask_test, window_mask)
# print(f"IoU Score: {iou_score:.4f}")
# # precision_score = compute_precision(window_mask, window_mask_test)
# precision_score = compute_precision(window_mask_test, window_mask)
# print(f"Precision Score: {precision_score:.4f}")
# # recall_score = compute_recall(window_mask, window_mask_test)
# recall_score = compute_recall(window_mask_test, window_mask)
# print(f"Recall Score: {recall_score:.4f}")

# # Show intersection overlay
# # Create a 3-channel RGB image for visualization
# overlay = np.zeros((window_mask.shape[0], window_mask.shape[1], 3), dtype=np.uint8)

# # Red for prediction only
# overlay[(window_mask > 0) & (window_mask_test == 0)] = [255, 0, 0]
# # Blue for ground truth only
# overlay[(window_mask_test > 0) & (window_mask == 0)] = [0, 0, 255]
# # Green for intersection
# overlay[(window_mask > 0) & (window_mask_test > 0)] = [0, 255, 0]

# display(Image.fromarray(overlay))


# ## Doors Test

# In[54]:


# import copy


# # 1. Find min and max x
# all_xs = [pt[0] for rect in processed_doors for line in rect for pt in line]
# min_x, max_x = min(all_xs), max(all_xs)

# # 2. Flip x for each point
# flipped_doors = copy.deepcopy(processed_doors)
# for rect in flipped_doors:
#     for line in rect:
#         for pt in line:
#             pt[0] = min_x + max_x - pt[0]

# # Now flipped_doors contains the horizontally flipped coordinates
# print(flipped_doors)


# In[55]:


# import numpy as np
# import cv2

# img = cv2.imread(img_path)

# img_height, img_width = img.shape[:2]
# door_mask_test = np.zeros((img_height, img_width), dtype=np.uint8)

# scale = 100  # adjust as needed

# for rect in processed_doors:
#     # Extract the 4 corner points (first point of each line)
#     points = [line[0] for line in rect]
#     # Convert to pixel coordinates
#     pixel_points = np.array([[int(round(x * scale)), int(round(y * scale))] for x, y in points], dtype=np.int32)
#     # Reshape for fillPoly
#     pixel_points = pixel_points.reshape((-1, 1, 2))
#     # Draw filled polygon
#     cv2.fillPoly(door_mask_test, [pixel_points], 255)

# # display(Image.fromarray(door_mask_test))
# # display(Image.fromarray(door_mask))

# kernel = np.ones((3, 3), np.uint8)
# door_mask_test = cv2.dilate(door_mask_test, kernel, iterations=2)

# # === Doors ===
# # iou_score = compute_iou(door_mask, door_mask_test)
# iou_score = compute_iou(door_mask_test, door_mask)
# print(f"IoU Score: {iou_score:.4f}")
# # precision_score = compute_precision(door_mask, door_mask_test)
# precision_score = compute_precision(door_mask_test, door_mask)
# print(f"Precision Score: {precision_score:.4f}")
# # recall_score = compute_recall(door_mask, door_mask_test)
# recall_score = compute_recall(door_mask_test, door_mask)
# print(f"Recall Score: {recall_score:.4f}")

# # Show intersection overlay
# # Create a 3-channel RGB image for visualization
# overlay = np.zeros((window_mask.shape[0], window_mask.shape[1], 3), dtype=np.uint8)

# # Red for prediction only
# overlay[(door_mask > 0) & (door_mask_test == 0)] = [255, 0, 0]
# # Blue for ground truth only
# overlay[(door_mask_test > 0) & (door_mask == 0)] = [0, 0, 255]
# # Green for intersection
# overlay[(door_mask > 0) & (door_mask_test > 0)] = [0, 255, 0]

# display(Image.fromarray(overlay))




# In[56]:


# import matplotlib.pyplot as plt
# import numpy as np

# # Normalize masks to 0-1 for stacking
# door = (door_mask_test > 0).astype(np.uint8)
# # door = (door_mask > 0).astype(np.uint8)
# window = (window_mask_test > 0).astype(np.uint8)
# # window = (window_mask > 0).astype(np.uint8)
# wall = (filled_wall_img > 0).astype(np.uint8)
# # wall = (wall_mask > 0).astype(np.uint8)

# # Create an RGB image: walls=gray, doors=red, windows=blue (overlap: magenta/cyan/white)
# merged = np.zeros((door.shape[0], door.shape[1], 3), dtype=np.uint8)
# # Walls: gray
# merged[wall > 0] = [180, 180, 180]
# # Doors: add red
# merged[door > 0] = [255, 0, 0]
# # Windows: add blue
# merged[window > 0] = [0, 0, 255]
# # If overlap, combine colors
# merged[(wall > 0) & (door > 0)] = [255, 100, 100]
# merged[(wall > 0) & (window > 0)] = [100, 100, 255]
# merged[(door > 0) & (window > 0)] = [255, 0, 255]
# merged[(wall > 0) & (door > 0) & (window > 0)] = [255, 255, 255]

# plt.figure(figsize=(8, 8))
# plt.imshow(merged)
# plt.title('Walls, Doors, and Windows Merged')
# plt.axis('off')
# plt.show()


# # ❗🤖 New Findings – Wall Channel Fixed
# 
# Simpler Pipeline helped to fix the wall Channels
# Future Explorations

# ### Key Finding
# Our **original pipeline** gave poor wall masks because it:
# - Used **argmax** across all room classes → walls lost to background/outdoor.
# - Applied **rotate + resize (align_corners=True)** → blurred thin wall lines.
# - Accidentally applied a **double rotation correction** (`'tensor'` + `'points'`).
# 
# ### What Worked Better
# The **simpler pipeline** produced clean walls by:
# - Taking **softmax probability** of the **Wall channel** only, then thresholding (e.g. `>0.8`).
# - **Padding** inputs to multiples of 4 instead of resizing (kept pixel alignment).
# - Skipping fragile polygonization and looking directly at the wall mask.
# 
# ### Why It Matters
# - **Thin structures** like walls are best recovered from **probability maps**, not hard labels.
# - **Geometry alignment** (no resampling, no odd rotations) keeps edges sharp.
# - **Post-processing** should be minimal or carefully tuned (binary closing, higher thresholds).
# 
# ✅ **Rule of thumb:** *Always extract the class probability you care about, avoid unnecessary interpolation, and fix rotations before averaging.*

# ## 🧱 ▢ Walls only contour
# 
# Using the walls heatmap (not accurate for walls) as the contour around the walls
# 
# Used to remove the windows

# ### 🤖🧱 → Seperated Cubi - Walls

# In[57]:


# ▓▓▓  Wall-only inference with the pre-trained Furukawa checkpoint  ▓▓▓
# Works on CPU, Apple-M-series (mps) or CUDA automatically.

import torch
from model import get_model   # make sure your repo is on PYTHONPATH

# ───────────────────────────────────────────────────────────
# 0.  pick a device
# ───────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")   # Apple Silicon (M1/M2)
else:
    device = torch.device("cpu")

print("Running on:", device)

# ───────────────────────────────────────────────────────────
# 1.  Build the network exactly like in the checkpoint
# ───────────────────────────────────────────────────────────
N_CLASSES         = 44
ROOM_OFFSET       = 21      # first 21 are junctions
WALL_ROOM_INDEX   = 2       # index 2 inside the 12-room block
WALL_CHANNEL      = ROOM_OFFSET + WALL_ROOM_INDEX # 21 + 2

net = get_model('hg_furukawa_original', 51)
net.conv4_   = torch.nn.Conv2d(256, N_CLASSES, 1, bias=True)
net.upsample = torch.nn.ConvTranspose2d(N_CLASSES, N_CLASSES, 4, 4)

ckpt = torch.load('model_best_val_loss_var.pkl', map_location='cpu')
net.load_state_dict(ckpt['model_state'])
net.to(device).eval()

# ───────────────────────────────────────────────────────────
# 2.  Utility: run one RGB numpy image → wall prob / mask
# ───────────────────────────────────────────────────────────
@torch.no_grad()
def wall_mask_from_numpy(rgb_np, thresh=0.35):
    """
    rgb_np : H×W×3 uint8 (0–255)  in RGB order.
    returns : (prob, mask)        both H×W   float32 / bool
    """
    rgb = 2*(rgb_np/255.0) - 1           # [-1,1] like training
    t   = torch.from_numpy(rgb).permute(2,0,1)[None].float().to(device)

    logits = net(t)                      # B×44×H×W
    room_logits = logits[:, ROOM_OFFSET:ROOM_OFFSET+12]
    wall_prob   = torch.softmax(room_logits, 1)[:, WALL_ROOM_INDEX]  # B×H×W
    wall_prob   = wall_prob[0].cpu()      # remove batch, back to CPU
    wall_mask   = (wall_prob > thresh)

    return wall_prob.numpy(), wall_mask.numpy()

import cv2, matplotlib.pyplot as plt, numpy as np
from PIL import Image

# Utility: pad to multiple of 4 for safe inference
def pad_to_multiple_of_4(img):
    h, w = img.shape[:2]
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                    borderType=cv2.BORDER_REFLECT)
    return img_padded, pad_h, pad_w

# ───────────────────────
# Load and enhance image
# ───────────────────────
# img_path = "Images/example8.png"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
display(Image.fromarray(gray))
img = cv2.detailEnhance(img, sigma_s=100, sigma_r=10)
display(Image.fromarray(img))

# Convert to RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Pad to avoid resolution loss from network stride
rgb_padded, pad_h, pad_w = pad_to_multiple_of_4(rgb)

# Inference
prob_padded, mask_padded = wall_mask_from_numpy(rgb_padded, thresh=0.8)

# Crop back to original size
if pad_h > 0:
    prob_padded = prob_padded[:-pad_h, :]
    mask_padded = mask_padded[:-pad_h, :]
if pad_w > 0:
    prob_padded = prob_padded[:, :-pad_w]
    mask_padded = mask_padded[:, :-pad_w]

prob, mask = prob_padded, mask_padded

# Visualize
plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.title("wall probability"); plt.axis('off')
plt.imshow(prob, cmap='hot'); plt.colorbar(fraction=0.046)
plt.subplot(1,2,2); plt.title("binary wall mask"); plt.axis('off')
plt.imshow(mask, cmap='gray')
# display(Image.fromarray((mask * 255).astype(np.uint8)))
display(Image.fromarray(mask))
plt.tight_layout(); # plt.show()  # Commented out for non-interactive execution

# =================================================

print("mask", mask)





# In[58]:


def mask_to_image_cv2(mask, output_path='mask_image.png'):
    # Ensure mask is a boolean NumPy array
    if not isinstance(mask, np.ndarray) or mask.dtype != bool:
        raise ValueError("Mask must be a boolean NumPy array")

    # Convert boolean mask to uint8 (0 for False, 255 for True)
    img = (mask.astype(np.uint8) * 255)

    # Convert to 3-channel RGB (replicate grayscale across R, G, B)
    img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img_3ch


mask_img = mask_to_image_cv2(mask)

gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)



contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
contour = largest_contour


epsilon = epsilon_mult * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)
contour = approx
vis_image = np.zeros_like(mask_img)
contour = straighten_rectilinear(contour)
contour = straighten_rectilinear(contour)

cv2.drawContours(vis_image, [contour], -1, (255, 255, 255), 2)
display(Image.fromarray(vis_image))




def contour_to_list(contour):
    """
    Convert a contour of shape (n, 1, 2) to a list of [x, y] points.
    
    Parameters:
    - contour: NumPy array of shape (n, 1, 2) or similar
    
    Returns:
    - List of [x, y] coordinates, e.g., [[a, b], [c, d], ...]
    """
    # Ensure contour is a NumPy array
    if not isinstance(contour, np.ndarray):
        contour = np.array(contour)
    
    # Check shape and reshape if necessary
    if len(contour.shape) == 3 and contour.shape[1] == 1:
        # Reshape from (n, 1, 2) to (n, 2)
        contour = contour.reshape(-1, 2)
    
    # Convert to list of [x, y] pairs
    return contour.tolist()

# Convert the contour
outter_boundaries = contour_to_list(contour)

for boundary in outter_boundaries:
    boundary[0] = (boundary[0] / 100)
    boundary[1] = (boundary[1] / 100)

print("outter_boundaries", outter_boundaries)


# ### Test Cubi-walls vs cv2-walls

# In[59]:


import numpy as np
from PIL import Image


# Convert outline to filled polygon
contours, _ = cv2.findContours(wall_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled_wall_img = np.zeros_like(wall_img)
cv2.drawContours(filled_wall_img, contours, -1, 255, thickness=cv2.FILLED)

print("filled_wall_img", filled_wall_img.shape)
display(Image.fromarray(filled_wall_img))

print("mask", mask.shape)
display(Image.fromarray(mask))

print("wall_img", wall_img.shape)
print("img", wall_img)

# Show intersection overlay
# Create a 3-channel RGB image for visualization
overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

# Red for prediction only
overlay[(mask > 0) & (filled_wall_img == 0)] = [255, 0, 0]
# Blue for ground truth only
overlay[(filled_wall_img > 0) & (mask == 0)] = [0, 0, 255]
# Green for intersection
overlay[(mask > 0) & (filled_wall_img > 0)] = [0, 255, 0]

display(Image.fromarray(overlay))


# ## Seperated Cubi - Windows

# In[60]:


# ▓▓▓  Window-only inference with the pre-trained Furukawa checkpoint  ▓▓▓
# Works on CPU, Apple-silicon (mps) or CUDA automatically.

import torch
from model import get_model                      # your repo factory

# ─── 0. pick device ──────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")                 # Apple M-series
else:
    device = torch.device("cpu")
print("Running on:", device)

# ─── 1. build network exactly like checkpoint ───────────────────────
N_CLASSES         = 44
ROOM_OFFSET       = 21                              # 0-20 junctions
ICON_OFFSET       = ROOM_OFFSET + 12                # 33-43  icons
WINDOW_ICON_INDEX = 1                               # ["NoIcon", *"Window"*, …]
WINDOW_CHANNEL    = ICON_OFFSET + WINDOW_ICON_INDEX # 33 + 1 = 34

net = get_model('hg_furukawa_original', 51)
net.conv4_   = torch.nn.Conv2d(256, N_CLASSES, 1, bias=True)
net.upsample = torch.nn.ConvTranspose2d(N_CLASSES, N_CLASSES, 4, 4)

ckpt = torch.load('model_best_val_loss_var.pkl', map_location='cpu')
net.load_state_dict(ckpt['model_state'])
net.to(device).eval()

# ─── 2. helper : RGB numpy  →  window prob / mask ───────────────────
@torch.no_grad()
def window_mask_from_numpy(rgb_np, thresh=0.50):
    """
    rgb_np : H×W×3 uint8 in RGB.
    returns : (prob, mask)   both H×W  float32 / bool
    """
    rgb   = 2*(rgb_np/255.0) - 1                      # training normalisation
    t     = torch.from_numpy(rgb).permute(2,0,1)[None].float().to(device)

    logits      = net(t)                              # B×44×H×W
    icon_logits = logits[:, ICON_OFFSET:ICON_OFFSET+11]  # 11 icon channels
    win_prob    = torch.softmax(icon_logits, 1)[:, WINDOW_ICON_INDEX]  # B×H×W
    win_prob    = win_prob[0].cpu()                   # to CPU, drop batch
    win_mask    = (win_prob > thresh)
    return win_prob.numpy(), win_mask.numpy()

# ─── 3. demo ────────────────────────────────────────────────────────
import cv2, matplotlib.pyplot as plt, numpy as np
from PIL import Image                                   # for display()

# img_path = "Images/atlantic-floor-plan.png"              # ← your raster
# bgr      = cv2.imread(img_path)
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
display(Image.fromarray(gray))
img = cv2.detailEnhance(img, sigma_s=100, sigma_r=10)
display(Image.fromarray(img))

rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

prob, win_mask = window_mask_from_numpy(rgb, thresh=0.05)    # tweak threshold!

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.title("window probability"); plt.axis('off')
plt.imshow(prob, cmap='hot'); plt.colorbar(fraction=0.046)
plt.subplot(1,2,2); plt.title("binary window mask"); plt.axis('off')
plt.imshow(win_mask, cmap='gray')
plt.tight_layout(); plt.show()


# ## Seperated Cubi - Doors

# In[61]:


# ▓▓▓  Window-only inference with the pre-trained Furukawa checkpoint  ▓▓▓
# Works on CPU, Apple-silicon (mps) or CUDA automatically.

import torch
from model import get_model                      # your repo factory

# ─── 0. pick device ──────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")                 # Apple M-series
else:
    device = torch.device("cpu")
print("Running on:", device)

# ─── 1. build network exactly like checkpoint ───────────────────────
N_CLASSES         = 44
ROOM_OFFSET       = 21                              # 0-20 junctions
ICON_OFFSET       = ROOM_OFFSET + 12                # 33-43  icons
DOOR_ICON_INDEX = 2                               # ["NoIcon", *"Window"*, …]
DOOR_CHANNEL    = ICON_OFFSET + DOOR_ICON_INDEX # 33 + 1 = 34

net = get_model('hg_furukawa_original', 51)
net.conv4_   = torch.nn.Conv2d(256, N_CLASSES, 1, bias=True)
net.upsample = torch.nn.ConvTranspose2d(N_CLASSES, N_CLASSES, 4, 4)

ckpt = torch.load('model_best_val_loss_var.pkl', map_location='cpu')
net.load_state_dict(ckpt['model_state'])
net.to(device).eval()

# ─── 2. helper : RGB numpy  →  window prob / mask ───────────────────
@torch.no_grad()
def window_mask_from_numpy(rgb_np, thresh=0.50):
    """
    rgb_np : H×W×3 uint8 in RGB.
    returns : (prob, mask)   both H×W  float32 / bool
    """
    rgb   = 2*(rgb_np/255.0) - 1                      # training normalisation
    t     = torch.from_numpy(rgb).permute(2,0,1)[None].float().to(device)

    logits      = net(t)                              # B×44×H×W
    icon_logits = logits[:, ICON_OFFSET:ICON_OFFSET+11]  # 11 icon channels
    win_prob    = torch.softmax(icon_logits, 1)[:, DOOR_ICON_INDEX]  # B×H×W
    win_prob    = win_prob[0].cpu()                   # to CPU, drop batch
    win_mask    = (win_prob > thresh)
    return win_prob.numpy(), win_mask.numpy()

# ─── 3. demo ────────────────────────────────────────────────────────
import cv2, matplotlib.pyplot as plt, numpy as np
from PIL import Image                                   # for display()

# img_path = "Images/atlantic-floor-plan.png"              # ← your raster
# bgr      = cv2.imread(img_path)
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
display(Image.fromarray(gray))
img = cv2.detailEnhance(img, sigma_s=100, sigma_r=100)
display(Image.fromarray(img))

rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


prob, door_mask = window_mask_from_numpy(rgb, thresh=0.25)    # tweak threshold!

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.title("window probability"); plt.axis('off')
plt.imshow(prob, cmap='hot'); plt.colorbar(fraction=0.046)
plt.subplot(1,2,2); plt.title("binary window mask"); plt.axis('off')
plt.imshow(door_mask, cmap='gray')
plt.tight_layout(); # plt.show()  # Commented out for non-interactive execution


def main_floorplan_processing(img_path, tmp_dir):
    """
    Main function to process a floorplan image and generate all the required outputs.
    
    Args:
        img_path (str): Path to the input floorplan image
        tmp_dir (str): Directory to save output files
    
    Returns:
        dict: Dictionary containing all processed data
    """
    # Create tmp directory if it doesn't exist
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    # Set the main image path
    main_img_path = img_path
    
    # Import all required libraries
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    import io 
    
    # Define a dummy display function for non-Jupyter environments
    try:
        from IPython.display import display
    except ImportError:
        def display(*args, **kwargs):
            pass  # Do nothing in non-Jupyter environments
    
    from PIL import Image
    import math
    import json
    from shapely.geometry import Polygon, LineString, Point
    from shapely.ops import unary_union
    import shapely
    from shapely.geometry import LineString, Point
    from shapely.geometry import LineString
    
    # Check for model file
    model_path = 'model/model_1427.pth'
    if not os.path.exists(model_path):
        # Try alternative path
        model_path = 'model_best_val_loss_var.pkl'
        if not os.path.exists(model_path):
            import subprocess
            subprocess.run(["gdown", "https://drive.google.com/uc?id=1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK"])
    
    # Read and process the image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    
    # Create blank canvas for drawing
    height, width, channels = img.shape
    blank_image = np.zeros((height,width,3), np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Detect outer contour
    contour, img, perp_score = detectOuterContours(gray, blank_image, color=(255,255,255))
    
    # ROUND 2 - Clean up contour
    bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    
    # Refine approximation
    if perp_score >= 0.99:
        epsilon_mult = 0.0001
    elif perp_score >= 0.9:
        epsilon_mult = 0.0035
    elif perp_score >= 0.8:
        epsilon_mult = 0.008
    else:
        epsilon_mult = 0.005
    
    epsilon = epsilon_mult * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    contour = approx
    
    # Apply straightening
    contour = straighten_rectilinear(contour,15)
    contour = straighten_rectilinear(contour,8)
    
    # Scale and export floor points
    floor_boxes = [[point[0][0]/100, point[0][1]/100] for point in contour]
    
    # Save floor vertices
    save_to_file(os.path.join(tmp_dir, "floor_vertices"), floor_boxes, True)
    
    # Canvas detection
    img = cv2.imread(img_path)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey, 255, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    canvas_vertices = [[point[0][0]/100, point[0][1]/100] for point in largest_contour]
    save_to_file(os.path.join(tmp_dir, "canvas_vertices"), canvas_vertices, True)
    
    # Wall detection
    img = cv2.imread(img_path)
    mask = np.zeros((height, width), dtype=np.uint8)
    scaled_contour = contour + np.sign(contour - np.mean(contour, axis=0)) * np.array([[[2, 2]]]) 
    scaled_contour = np.array(scaled_contour, dtype=np.int32)
    cv2.drawContours(mask, [scaled_contour], -1, (255), -1)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    white_background = np.full_like(img, 255)
    inv_mask = cv2.bitwise_not(mask)
    white_outside = cv2.bitwise_and(white_background, white_background, mask=inv_mask)
    masked_img = cv2.add(masked_img, white_outside)
    
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    wall_img = wall_filter(closed)
    wall_boxes, img = detectPreciseBoxes(wall_img)
    
    # Create wall vertices
    wall_vertices = create_vertices(wall_boxes)
    cleaned_walls = clean_small_walls_shapely(wall_vertices)
    wall_vertices = cleaned_walls
    save_to_file(os.path.join(tmp_dir, "walls_vertices"), wall_vertices, True)
    
    # Neural network processing for windows and doors
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from model import get_model
    from utils.loaders import FloorplanSVG, DictToTensor, Compose, RotateNTurns
    from utils.plotting import segmentation_plot, polygons_to_image, draw_junction_from_dict, discrete_cmap
    from utils.post_prosessing import split_prediction, get_polygons, split_validation
    from mpl_toolkits.axes_grid1 import AxesGrid
    
    # Model setup
    rot = RotateNTurns()
    room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath",
                    "Entry", "Railing", "Storage", "Garage", "Undefined"]
    icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink",
                    "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]
    
    model = get_model('hg_furukawa_original', 51)
    n_classes = 44
    split = [21, 12, 11]
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    
    # Try to load the model with different possible paths
    model_loaded = False
    for model_file in ['model/model_1427.pth', 'model_best_val_loss_var.pkl']:
        if os.path.exists(model_file):
            try:
                checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
                if 'model_state' in checkpoint:
                    model.load_state_dict(checkpoint['model_state'])
                else:
                    model.load_state_dict(checkpoint)
                model_loaded = True
                break
            except Exception as e:
                print(f"Failed to load model from {model_file}: {e}")
                continue
    
    if not model_loaded:
        raise FileNotFoundError("Could not load model from any available path")
    
    model.eval()
    
    # Image preparation
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = 2 * (img / 255.0) - 1
    img = np.moveaxis(img, -1, 0)
    img = torch.tensor([img.astype(np.float32)])
    n_rooms = len(room_classes)
    n_icons = len(icon_classes)
    
    # Model prediction
    with torch.no_grad():
        size_check = np.array([img.shape[2],img.shape[3]])%2
        height = img.shape[2] - size_check[0]
        width = img.shape[3] - size_check[1]
        img_size = (height, width)
        
        rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
        pred_count = len(rotations)
        prediction = torch.zeros([pred_count, n_classes, height, width])
        
        for i, r in enumerate(rotations):
            forward, back = r
            rot_image = rot(img, 'tensor', forward)
            pred = model(rot_image)
            pred = rot(pred, 'tensor', back)
            pred = rot(pred, 'points', back)
            pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
            prediction[i] = pred[0]
        
        prediction = torch.mean(prediction, 0, True)
        
        rooms_pred = F.softmax(prediction[0, 21:21+12], 0).cpu().data.numpy()
        rooms_pred = np.argmax(rooms_pred, axis=0)
        icons_pred = F.softmax(prediction[0, 21+12:], 0).cpu().data.numpy()
        icons_pred = np.argmax(icons_pred, axis=0)
        
        heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
        polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), 0.2, [1, 2])
    # Extract windows
    window_class_number = icon_classes.index("Window")
    window_polygon_numbers=[i for i,j in enumerate(types) if j['class']==icon_classes.index("Window")and (j['type']=='icon')]
    boxes=[]
    for i,j in enumerate(polygons):
        if i in window_polygon_numbers:
            temp=[]
            for k in j:
                temp.append(np.array([k]))
            boxes.append(np.array(temp))
    
    window_vertices = create_vertices(boxes)
    
    # Extract doors
    door_polygon_numbers=[i for i,j in enumerate(types) if (j['class']==icon_classes.index("Door")) and (j['type']=='icon')]
    boxes=[]
    for i,j in enumerate(polygons):
        if i in door_polygon_numbers:
            temp=[]
            for k in j:
                temp.append(np.array([k]))
            boxes.append(np.array(temp))
    
    door_vertices = create_vertices(boxes)
    
    # Post-processing for windows
    if window_vertices:
        intersecting_win_vertices = detach_intersecting_objects(window_vertices, wall_vertices)
        intersecting_win_vertices = detach_intersecting_objects(intersecting_win_vertices, wall_vertices)
        intersecting_win_vertices = detach_intersecting_objects(intersecting_win_vertices, wall_vertices)
        
        if intersecting_win_vertices:
            Detached_windows = intersecting_win_vertices
            Detached_windows = fix_rectangles(Detached_windows)
            window_vertices = Detached_windows
    
    # Post-processing for doors
    if door_vertices:
        intersecting_door_vertices = detach_intersecting_objects(door_vertices, wall_vertices)
        intersecting_door_vertices = detach_intersecting_objects(intersecting_door_vertices, wall_vertices)
        intersecting_door_vertices = detach_intersecting_objects(intersecting_door_vertices, wall_vertices)
        
        if intersecting_door_vertices:
            Detached_doors = intersecting_door_vertices
            Detached_doors = fix_rectangles(Detached_doors)
            door_vertices = Detached_doors
    
    # Scale and snap windows
    for i, window in enumerate(window_vertices):
        scaled_window = scale_object(window, 0.8)
        new_window = []
        for window_edge in scaled_window:
            new_window.append(window_edge)
        window_vertices[i] = new_window
    
    snapped_windows = snap_object_sides_to_walls(window_vertices, wall_vertices)
    snapped_windows = fix_rectangles(snapped_windows)
    window_vertices = snapped_windows
    
    # Scale and snap doors
    for i, door in enumerate(door_vertices):
        scaled_door = scale_object(door, 0.8)
        new_door = []
        for door_edge in scaled_door:
            new_door.append(door_edge)
        door_vertices[i] = new_door
    
    snapped_doors = snap_object_sides_to_walls(door_vertices, wall_vertices)
    snapped_doors = fix_rectangles(snapped_doors)
    door_vertices = snapped_doors
    
    # Clean duplicates
    no_dup_windows = clean_duplicates(window_vertices)
    no_dup_windows = fix_rectangles(no_dup_windows)
    window_vertices = no_dup_windows
    
    no_dup_doors = clean_duplicates(door_vertices)
    no_dup_doors = fix_rectangles(no_dup_doors)
    door_vertices = no_dup_doors
    
    # Handle door-window collisions
    window_polygons = [edges_to_polygon(w) for w in window_vertices]
    door_polygons = [edges_to_polygon(d) for d in door_vertices]
    
    collisions = []
    for wi, w_poly in enumerate(window_polygons):
        for di, d_poly in enumerate(door_polygons):
            if w_poly.intersects(d_poly):
                collisions.append(door_vertices[di])
    
    door_vertices = [d for d in door_vertices if d not in collisions]
    
    # Process windows and doors
    processed_windows = process_windows(window_vertices, 1.0)
    window_vertices = processed_windows
    window_vertices = fix_rectangles(window_vertices)
    
    processed_doors = process_windows(door_vertices, 1.0)
    door_vertices = processed_doors
    door_vertices = fix_rectangles(door_vertices)
    
    # Save all results
    save_to_file(os.path.join(tmp_dir, "doors_vertices"), door_vertices, True)
    save_to_file(os.path.join(tmp_dir, "windows_vertices"), window_vertices, True)
    save_to_file(os.path.join(tmp_dir, "walls_vertices"), wall_vertices, True)
    save_to_file(os.path.join(tmp_dir, "canvas_vertices"), canvas_vertices, True)
    save_to_file(os.path.join(tmp_dir, "floor_vertices"), floor_boxes, True)
    
    # Create JSON output
    output_data = {
        "doors": door_vertices,
        "floor": floor_boxes,
        "walls": wall_vertices,
        "windows": window_vertices,
        "canvas": canvas_vertices
    }
    
    with open(os.path.join(tmp_dir, "floorplan_data.json"), "w") as f:
        json.dump(output_data, f)
    
    return output_data

# If this file is run directly, execute the main processing
if __name__ == "__main__":
    # This allows the file to be run directly for testing
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        tmp_dir = "tmp"
        result = main_floorplan_processing(img_path, tmp_dir)
        print("Processing completed successfully!")
        print(f"Results saved to {tmp_dir}/")
    else:
        print("Usage: python flooplan_detector.py <image_path>")

