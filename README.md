**RECAP — Rapid Event-level Classification of Affected Properties**
=======================================================================

**Group Information**
---------------------

**Aryan Anand** Email: a.aryansurya007@gmail.com

**Yatin Raju** Email: yatinrcb@gmail.com

**Sri Satishkumar** Email: @gmail.com

**Project Summary / Abstract**
------------------------------

Natural disasters cause widespread damage, making rapid assessment essential for response and recovery. Traditional manual inspection of satellite imagery is slow and resource‑intensive.

This project introduces **RECAP**, a two‑stage deep‑learning pipeline for **Rapid Event-level Classification of Affected Properties** using pre‑ and post‑disaster satellite images from the xBD dataset.

Our pipeline consists of:

1.  **Building Segmentation Model**A FCN‑ResNet50 network extracts building footprints from disaster imagery.
    
2.  **Polygon‑aware Siamese Damage Classifier**A ResNet‑34 Siamese encoder learns changes between pre/post images and assigns each building one of four xBD damage categories:
    
    *   No Damage
        
    *   Minor Damage
        
    *   Major Damage
        
    *   Destroyed
        

We benchmark model performance and provide research‑ready evaluation plots (confusion matrix, ROC/PR curves, calibration, etc.).This study contributes a reproducible, end‑to‑end framework capable of scaling to real‑world disaster‑response workflows.

**Problem Description**
-----------------------

### **Problem Summary**

Disaster‑response teams need fast, reliable methods to analyze satellite images and estimate building damage. The challenge lies in:

*   Variability across disaster types (floods, hurricanes, wildfires)
    
*   Inconsistent lighting, shadows, and viewing angles
    
*   Dense urban environments requiring precise building extraction
    
*   Highly imbalanced damage categories
    

### **Motivation**

*   Reduce human workload in large‑scale disaster sites
    
*   Provide consistent, automated, and explainable predictions
    
*   Explore deep-learning strategies for polygon‑level classification
    
*   Deliver tools that can visualize results on an interactive map
    

### **Challenges**

*   Noisy or incomplete building outlines
    
*   Domain shift across geographic regions
    
*   Large dataset size (~40k high‑res tiles)
    
*   Aggregating pixel‑level features into building polygons
    
*   Balancing accuracy across rare classes (e.g., major damage)
    

**Contributions**
-----------------

### **\[Novel System\]** + **\[Extension of Prior Work\]**

This project builds on concepts from prior segmentation and change‑detection research but introduces a unique, fully reproducible workflow:

### **Our Key Contributions**

*   ✔ **A two‑stage vision system** combining semantic segmentation + polygon‑level damage classification
    
*   ✔ **A ResNet‑34 Siamese change‑encoder** that learns building‑level differences
    
*   ✔ **Feature pooling inside polygons**, not bounding boxes — increasing accuracy
    
*   ✔ **A full evaluation suite** generating:
    
    *   Confusion matrix
        
    *   ROC curves
        
    *   PR curves
        
    *   Calibration curves
        
    *   Per-class F1
        
    *   Class distribution plots
        
*   ✔ **A partial HuggingFace dataset downloader** (public xBD mirror) for easy setup
    
*   ✔ **A Streamlit map interface** for visualizing building damage on an interactive world map
    

**References**
--------------

1.  Gupta et al. _xBD: A Dataset for Assessing Building Damage from Satellite Imagery_ (2019).
    
2.  He et al. _Deep Residual Learning for Image Recognition_. CVPR (2016).
    
3.  Long et al. _Fully Convolutional Networks for Semantic Segmentation_. CVPR (2015).
    
4.  Spall, J.C. _Simultaneous Perturbation Stochastic Approximation_ (1998).
    
5.  More coming from your academic paper.
    

**Dataset Download Instructions**
=================================

We use a **partial downloader** for the public HuggingFace dataset:

**Repo:** aryananand/xBD

**Downloader script:** /src/download\_xbd\_partial.py

### **Run the downloader:**
```bash
python3 src/download_xbd_partial.py
```
### **Downloaded dataset structure:**

```
data/xBD/
    train/
        images/
        labels/
        target/
    test/
        images/
        labels/
        target/
```
Each image tile includes _pre‑_ and _post‑disaster_ PNGs with matching label JSONs.

**Directory Structure**
=======================

```
├── data/                                   # Dataset downloaded from HuggingFace
│   └── xBD/
│       ├── train/
│       │   ├── images/
│       │   ├── labels/
│       │   └── target/
│       └── test/
│           ├── images/
│           ├── labels/
│           └── target/
│
├── models/                                 # Saved models
│   └── polygon_siamese_best.pt
│
├── artifacts/
│   ├── plots/                              # Evaluation figures
│   └── demo_overlays/                      # Sample predicted overlay images
│
├── src/
│   ├── train_building_seg.py
│   ├── train_polygon_siamese.py
│   ├── eval_full_model.py
│   ├── model_polygon_siamese.py
│   ├── building_seg_model.py
│   ├── dataset_tiles.py
│   ├── app_streamlit_map.py
│   └── common.py
│
├── HuggingFace/
│   └── get_xbd.py                           # Partial dataset downloader
│
├── README.md
└── requirements.txt
```

**Dependencies**
================

### **Core Dependencies**

*   Python 3.11+
    
*   PyTorch
    
*   Torchvision
    
*   Numpy
    
*   Matplotlib
    
*   Pandas
    
*   Scikit‑learn
    
*   Pillow
    
*   tqdm
    
*   shapely
    
*   huggingface\_hub
    
*   hf\_transfer (optional, for faster downloads)
    
*   Streamlit (for frontend)
    
*   pydeck (for map visualization)

**How to Run**
==============

### **1\. Download Dataset**
```bash
python3 src/download_xbd_partial.py   
```
### **2\. Train Building Segmentation**
```bash
python3 src/train_building_seg.py   
```
### **3\. Train Polygon Siamese Classifier**
```bash
python3 src/train_polygon_siamese.py   
```
### **4\. Evaluate Full Model**

(Generates all research‑quality plots)
```bash
python3 src/eval_full_model.py   
```

### **5\. Generate Overlays for Demo**
```bash
python3 -m src.demo_random_tiles   
```

### **6\. Select Top‑20 Best Predictions**
```bash
python3 -m src.select_best_tiles   
```

### **7\. Launch the Streamlit App**
```bash
streamlit run src/app_streamlit_map.py   
```

This will display:

*   A world map of all tiles
    
*   Pre/post images
    
*   Damage overlays
    
*   Best‑20 curated examples
    

**Demo Video**
===================

Link:
