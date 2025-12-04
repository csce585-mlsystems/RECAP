**Project Title**
=================

**RECAP: Automated Building Damage Detection Using Pre/Post‑Disaster Satellite Imagery**
----------------------------------------------------------------------------------------

**Group Info**
--------------

*   **Aryan Anand**
    
    *   Email: a.aryansurya007@gmail.com
        
*   **Yatin Raju**
    
    *   Email: yatinrcb@gmail.com
*   **Sri Satishkumar**
    
    *   Email: @gmail.com
        

**Project Summary / Abstract**
------------------------------

Natural disasters cause extensive structural damage that must be assessed quickly to guide emergency response. Satellite imagery provides large‑scale coverage, but manually analyzing thousands of buildings is slow and error‑prone.

In this project, we develop **RECAP — a two‑stage deep learning pipeline** that automatically detects buildings and classifies their damage severity using _pre‑ and post‑disaster_ satellite imagery. Our model integrates:

1.  **FCN‑ResNet50 building segmentation**
    
2.  **Polygon‑aware Siamese damage classifier (ResNet‑34 backbone)**
    

We evaluate RECAP on the **xBD disaster damage dataset** and generate research‑quality visualizations: confusion matrix, per‑class F1 plot, ROC curves, PR curves, calibration curve, and class support. Additionally, we build an **interactive Streamlit demo** that displays all disaster locations on a world map and allows users to visualize pre/post-image pairs and predicted damage overlays.

This system demonstrates a scalable, automated approach for rapid post‑disaster assessment.

**Problem Description**
-----------------------

### **Problem**

After disasters such as hurricanes, earthquakes, floods, and fires, governments must rapidly determine:

*   Where buildings are located
    
*   How severely each building is damaged
    
*   How damage is distributed across regions
    

Manual inspection is slow and expensive. Fully automatic systems must:

1.  Detect buildings in remote-sensing imagery
    
2.  Compare pre- and post-disaster images
    
3.  Identify damage types accurately (no-damage → destroyed)
    

### **Motivation**

*   Improve **speed** and **accuracy** of damage assessment
    
*   Support emergency agencies with **data-driven maps**
    
*   Demonstrate a **full ML pipeline** (training → evaluation → visualization → interactive UI)
    
*   Apply modern vision techniques to real-world humanitarian use cases
    

### **Challenges**

*   Highly imbalanced dataset (majority “no-damage”)
    
*   Wide variation in disasters, lighting, and image quality
    
*   Need to track **individual buildings** consistently across time
    
*   Processing thousands of polygons per tile efficiently
    

**Contribution**
----------------

### \[Novel System\] \[Extension of Existing Work\]

Our contributions include:

### **1\. Two‑Stage Deep Learning Pipeline**

*   **BuildingSegModel**
    
    *   FCN‑ResNet50 (offline weights)
        
    *   Predicts building masks for each tile
        
*   **PolygonSiamese Model**
    
    *   Siamese ResNet‑34
        
    *   Performs polygon mask pooling
        
    *   Classifies damage into four categories
        

### **2\. Polygon‑Aware Feature Engineering**

We compute:

\[vpre,vpost,vpost−vpre,∣vpost−vpre∣\]\[_v_pre​,_v_post​,_v_post​−_v_pre​,∣_v_post​−_v_pre​∣\]

This 4× feature concatenation gives strong temporal change representation.

### **3\. Full Evaluation Framework**

We implemented:

*   Confusion Matrix
    
*   Per-Class F1
    
*   ROC Curves
    
*   Precision‑Recall Curves
    
*   Calibration Curve
    
*   Class Support Distribution
    
*   CSV summary of all metrics
    

### **4\. Interactive Streamlit Front-End**

*   Load all xBD disaster coordinates
    
*   Place clickable pins on a global map
    
*   Show pre-image, post-image, predicted overlays, and ground truth
    
*   Supports selection of “20 best images” for demo
    

### **5\. Reproducible Pipeline & Scripts**

*   train\_polygon\_siamese.py
    
*   train\_building\_seg.py
    
*   eval\_full\_model.py
    
*   select\_best\_tiles.py
    
*   app\_streamlit\_map.py
    

**References**
--------------

(You can replace these with your actual citations.)

1.  Gupta, R., et al. **xBD: A Dataset for Assessing Building Damage from Satellite Imagery.**
    
2.  He, K., et al. **ResNet: Deep Residual Learning for Image Recognition.**
    
3.  Long, J., et al. **Fully Convolutional Networks for Semantic Segmentation.**
    
4.  Zhan, Y., et al. **Damage Assessment from Pre/Post-Disaster Imagery Using CNNs.**
    

**Reproducing Code for Milestone P1**
=====================================

1.  Install dependencies using pip or uv
    
2.  data/xBD Dataset/
    
3.  python3 -m src.train\_building\_seg
    
4.  python3 -m src.train\_polygon\_siamese
    
5.  python3 -m src.demo\_random\_tiles
    
6.  python3 -m src.eval\_full\_model
    

**Dependencies**
================

### Python Libraries

*   Python 3.10+
    
*   PyTorch
    
*   Torchvision
    
*   Numpy
    
*   Matplotlib
    
*   Scikit‑Learn
    
*   Pillow
    
*   Shapely
    
*   Streamlit
    
*   PyDeck
    

Ensure your repo includes:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`pyproject.toml    uv.lock`  

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

**How to Run**
==============

### **1\. Train the models**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python3 -m src.train_building_seg  python3 -m src.train_polygon_siamese   `

### **2\. Evaluate**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python3 -m src.eval_full_model   `

### **3\. Generate best demo tiles**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python3 -m src.select_best_tiles   `

### **4\. Launch Streamlit UI**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   streamlit run src/app_streamlit_map.py   `

**Demo**
========

Your project must include:

*   **Video walkthrough of the front-end**
    
*   **Slides (10 slide deck)**
    
*   **Live map demo (Streamlit)**
    
*   **Predicted overlays + ground truth**
