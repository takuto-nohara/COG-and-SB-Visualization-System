# Ground plane estimation references (reviewed)

## Purpose of this note
The 3D visualization layer in this repository adjusts the world-to-screen projection so that the Y axis is aligned to a physically plausible ground-up direction.  The references below describe representative peer-reviewed approaches for monocular/image-based ground plane estimation and robust plane fitting that were used as design references.

## References

1. Dragon, Ralf and Luc Van Gool. 2014, CVPR.
   - Title: **Ground Plane Estimation using a Hidden Markov Model**
   - Link: https://openaccess.thecvf.com/content_cvpr_2014/html/Dragon_Ground_Plane_Estimation_2014_CVPR_paper.html
   - Summary:
     - Estimates ground plane orientation and location from monocular video in a moving-observer setting.
     - Uses a geometric state formulation and enforces temporal consistency through an HMM-like model.
     - Focus point is stable estimation of plane normal and distance across frames, useful when direct per-frame geometry is noisy.
   - Relation to this project:
     - Our implementation applies frame-wise alignment with a fallback of robustness checks; this paper is a reference for temporal/robust handling when ground cues are noisy.

2. Liu, Chen et al. 2019, CVPR.
   - Title: **PlaneRCNN: 3D Plane Detection and Reconstruction from a Single Image**
   - Link: https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_PlaneRCNN_3D_Plane_Detection_and_Reconstruction_From_a_Single_Image_CVPR_2019_paper.html
   - Summary:
     - Detects and reconstructs arbitrary plane instances from a single RGB image.
     - Outputs explicit planar regions and plane parameters, then reconstructs depth geometry.
     - Demonstrates that images contain strong piecewise-planar priors that can be used to recover world-like orientation cues.
   - Relation to this project:
     - Not directly integrated yet, but motivates extending from landmark-only ground estimation to image-derived planar priors.

3. Chakraborty, Suryansh et al. (GroundNet).
   - Title: **GroundNet: Monocular Ground Plane Normal Estimation with Geometric Consistency**
   - Link: https://arxiv.org/abs/1811.07222
   - Summary:
     - Estimates ground plane normal from a single image using multi-stream geometric consistency (depth + normal cues).
     - Uses an explicit geometric consistency objective so depth and normal signals reinforce each other.
     - One of the common baselines for single-image ground orientation estimation.
   - Relation to this project:
     - Relevant as a future upgrade path: combine model-based geometric consistency with existing body-support landmark estimation.

4. Fischler, R. C., Bolles, R. C. 1981, CACM.
   - Title: **Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography**
   - Link: https://cacm.acm.org/research/random-sample-consensus/
   - Summary:
     - Introduces RANSAC, a robust estimator for model fitting under outliers.
     - Useful for plane fitting when inliers are sparse and corrupted by outliers.
   - Relation to this project:
     - Practical reference if we later switch from pure PCA fitting to a robust ground plane fit.

## Implementation note
The current fix in `src/cogsb/visualization/draw.py` uses a lightweight geometry-based alignment (support-point PCA and axis-aligned rotation), not a full deep monocular network.
This file documents the papers used as design reference for deciding ground-axis correction direction and robustness strategy.
