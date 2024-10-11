# triangles
Visualization Suite for Semantic Segmentation Model Outputs

The primary goal of this work is to provide easy (hopefully) tooling to compare semantic segmentation model outputs, including metrics, confusion matrices, and triangles plots to better identify model bottlenecks and facilitate model selection. This is extensible to any segmentation tasks. 

# Installation 

1. `git clone https://github.com/noellelaw/triangles`
2. `conda activate -n triangles python=3.12`
3. `conda activate triangles`
4. `python -m pip install -r requirements.txt`
5. `python triangles.py`

# Overview

1. Upload a zip file containing the following subdirectories: (1) ground_truth and (2) 1+ model prediction output folders, in which each model output has its own subdirectory and the prediction image name maps to a corresponding ground truth image.
2. (Optional) Define classes and categories. If you do not input these, classes will automatically be determined for you and plots that rely on catrgories cannot be created :broken-heart:. If you include the void class 0 (grayscale) or (0,0,0) RGB, it must be the first label input.
3. Select the Run Evaluator button to visualize your results! This will take longer if step 2 was not taken. All metrics and plotly graphs will be saved to {ROOT_DIR}/{ZIP_FILE}/.

# Eventual video walk through and discussion on 
why the triangles plot helps better visualize model results 
