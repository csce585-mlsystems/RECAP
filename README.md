# Project Title  
## RECAP: Rapid Event-level Classification of Affected Properties

## Group Info  
- Aryan Anand  
  - Email: asanand@email.sc.edu  
- Sri Satishkumar  
  - Email: satishks@email.sc.edu 
- Yatin Raju  
  - Email: bhupatir@email.sc.edu  

## Project Summary/Abstract  
RECAP is an end-to-end prototype that converts paired pre- and post-disaster satellite images into building-level damage labels with calibrated confidence. The model is trained on the public xView2/xBD dataset and outputs four categories: no damage, minor, major, destroyed. Results are presented on a simple Streamlit map for quick situational awareness. For class demos, predictions will be precomputed for one or two events so the app runs smoothly on a standard laptop.

## Problem Description  
- **Problem description (2–3 sentences):** After disasters, responders need a quick, trustworthy view of which buildings are safe, damaged, or destroyed. Field surveys are slow and risky, and raw satellite imagery is not directly actionable. We aim to automatically convert paired pre/post imagery into building-level assessments with usable confidence.  
- **Motivation**  
  - Faster triage: focus inspections where damage is most likely  
  - Smarter resource use: guide shelters, supplies, and aid  
  - Accessible tool: a clear map that non-experts can use  
- **Challenges**  
  - Class imbalance and label noise (minor vs major)  
  - Preventing leakage with event-level train/test splits  
  - Calibrating confidence for predictable threshold behavior  


## Contribution  
- [`Replication of existing work`]
- [`Extension of existing work`]

**Summary:** We will reproduce a standard Siamese CNN baseline on xView2/xBD, then extend it with probability calibration and a lightweight map UI.  
- Contribution 1: Event-split training and evaluation with a Siamese ResNet-18 baseline (image-only)  
- Contribution 2: Post-hoc calibration and an interactive Streamlit map with precomputed predictions  

## References  
### BibTeX of all references used in the project (will also be included as `references.bib`)

@inproceedings{gupta2019xbd,  
  title = {Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery},  
  author = {Gupta, Rohit and Goodman, Benjamin and Patel, Nilesh and others},  
  booktitle = {CVPR Workshops},  
  year = {2019}  
}

@inproceedings{daudt2018siamese,  
  title = {Fully Convolutional Siamese Networks for Change Detection},  
  author = {Daudt, Rodrigo Caye and Le Saux, Bertrand and Boulch, Alexandre},  
  booktitle = {IEEE ICIP},  
  year = {2018}  
}

@inproceedings{alam2018crisismmd,  
  title = {CrisisMMD: Multimodal Twitter Datasets from Natural Disasters},  
  author = {Alam, Firoj and Ofli, Ferda and Imran, Muhammad},  
  booktitle = {AAAI ICWSM},  
  year = {2018}  
}

---

# < The following is only applicable for the final project submission >  

## Dependencies  
None

## Directory Structure  
None

## How to Run  
None

## Demo  
None
