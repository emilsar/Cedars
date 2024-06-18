# Project: Automated Bladder Cancer Screening with Deep Learning Algorithms

## Project Title
Automated Bladder Cancer Screening with Deep Learning Algorithms

### Problem Statement
Bladder cancer's high incidence and recurrence rates necessitate enhanced screening methods. This project seeks to address this by developing algorithms for analyzing cell imagery from urine specimens to distinguish between cell nucleus and cytoplasm as a quantitative marker of malignancy.

### Summary of Learning Objectives
- Understanding image segmentation techniques.
- Application of deep learning models like UNET for medical image analysis.
- Quantitative analysis for cancer risk assessment.
- Practical implementation of segmentation and scoring systems using real-world medical data.

## Short Description
This project focuses on enhancing bladder cancer screening methods by developing algorithms to analyze cell imagery from urine specimens. The goal is to distinguish between cell nucleus and cytoplasm, using various image segmentation algorithms, as a quantitative marker of malignancy.

## Detailed Description

### Project Goals
- Learn and compare various image segmentation algorithms including intensity thresholding, clustering techniques, texture analysis, and neural network approaches like UNET.
- Deepen understanding of segmentation techniques in image processing and their application in medical diagnostics, particularly in improving the accuracy, reliability, and efficiency of bladder cancer screening.

### ML/Data Science Methods
- **Image Segmentation**: Techniques such as intensity thresholding, clustering, texture analysis, and neural networks (e.g., UNET) for identifying and segmenting nucleus and cytoplasm in cell images.
- **Deep Learning Algorithms**: Implementation of advanced neural networks to automate the detection and analysis of cellular features in medical images.
- **Quantitative Analysis**: Calculation of the nuclear-to-cytoplasmic (N/C) ratio as a key marker for malignancy, and development of scoring systems to assess cancer risk.

### Datasets
1. **Urothelial Cell Dataset (Model Training)**: Contains over 100 urothelial cells with ground truth segmentation masks for nucleus and cytoplasm. The goal is to train and compare various algorithms for segmenting these regions and calculating the N/C ratio.
2. **Specimen Cell Dataset**: Contains 25 cells per patient, with patients categorized by diagnosis. The aim is to study the predictive power of the developed biomarker (N/C ratio) and its distribution across different patient diagnoses.

### Getting Started Guide
#### Prerequisites
- Basic understanding of Python, machine learning, and image processing.

#### Environment Setup
1. **Install Python and necessary libraries**:
   ```bash
   pip install opencv-python pandas numpy torch matplotlib seaborn
   ```
2. **Recommended IDE**: Jupyter Notebook or VS Code.

#### Data Preparation
1. **Urothelial Cell Dataset**:
   - Download the dataset from [Urothelial Cell Dataset](TBD).
   - Run 1_prepdata.py to generate a pickle containing dictionary of images and segmentation masks.
   - Load and preprocess the images using OpenCV. See 1_load_data.ipynb for an example of intensity based methods. 
   - Construct segmentation masks for training models, should already exist for uploaded images.
   - See Project2/CedarsAI_team2_progress folder for example of the KMeans clustering approach as well as loading custom texture analysis approaches (GLCM, Gabor filters, first/second order features, random forest for feature selection, etc)
   - Train models, see 2_image_segmentation.ipynb.

2. **Specimen Cell Dataset**:
   - Download the dataset from [Specimen Cell Dataset](TBD).
   - Load and preprocess the images using developed algorithms.
   - Analyze the N/C ratio across different patient diagnoses.

#### Model Implementation
1. **Image Segmentation**:
   - Implement traditional segmentation methods like intensity thresholding and clustering.
   - Develop and train neural networks such as UNET for advanced segmentation.
2. **Quantitative Analysis**:
   - Calculate the N/C ratio for each cell. 
   - Develop a scoring system to assess cancer risk based on cellular features.
3. **Evaluation**:
   - Compare the performance of different segmentation algorithms. 
   - Calculate classification report for pixel-wise segmentation.
   - Correlate true versus predicted NC Ratio. 
   - Validate the predictive power of the N/C ratio with patient diagnoses.

### Suggested Readings and References
- [Bladder Cancer Screening and Diagnosis](https://pubmed.ncbi.nlm.nih.gov/37377320/)
- [Introduction to Image Segmentation](https://www.geeksforgeeks.org/image-segmentation-in-deep-learning/)
- [Deep Learning in Medical Imaging](https://www.sciencedirect.com/science/article/pii/S1361841518303005)

### Recommended Videos
- [Introduction to Image Processing](TBD)
- [Deep Learning with UNET](TBD)

### Data Download Links
- [Urothelial Cell Dataset](TBD)
- [Specimen Cell Dataset](TBD)

### Project Files
- [Download Project Files (ZIP)](TBD)

## Author Information
Joshua Levy - [GitHub](https://github.com/jlevy44) | [LinkedIn](https://www.linkedin.com/in/joshua-levy-87044913b) | [LevyLab](https://levylab.host.dartmouth.edu/)

