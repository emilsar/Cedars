# Project: Graph Neural Networks for Social and Biological Network Analysis

## Project Title
Graph Neural Networks for Social and Biological Network Analysis

## Short Description
This project introduces students to the cutting-edge field of Graph Neural Networks (GNNs) with a focus on understanding how GNNs can capture and represent complex relationships in non-Euclidean data, such as social and biological networks. Students will explore how GNN-derived numerical embeddings can effectively represent the intricate connections and interactions between individuals or entities for predictive analytics. The use of GNNs will be compared to other state-of-the-art community detection, social influence, and biological predictive models, with interpretation of important features and relationships.

## Detailed Description

### Project Goals
- Introduce students to the basics and advanced concepts of Graph Neural Networks (GNNs).
- Analyze and model relationships in both social and biological networks using GNNs.
- Compare the effectiveness of GNNs with traditional network analysis models like community detection algorithms and logistic regression.
- Provide hands-on experience with various GNN models, including Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs).

### Summary of Learning Objectives
- Graph theory fundamentals.
- Python programming for network analysis.
- Machine learning techniques for network data.
- Traditional and deep learning network analysis methods.
- Practical application of GNNs to real-world datasets in social and biological contexts.

### ML/Data Science Methods
- **Graph Theory**: Learn the basics of nodes, edges, and attributes, and how they represent real-world entities and their relationships.
- **Traditional Network Analysis**: Apply methods like centrality measures, community detection, and network visualization.
- **Deep Learning Network Analysis**: Implement and compare Graph Neural Networks such as GCNs and GATs.
- **Comparison with Traditional Methods**: Use logistic regression and other algorithms that do not account for network structure to highlight the benefits of GNNs.

### Datasets
1. **Physician Sharing Network**: Derived from Medicare claims data, this dataset details interactions between physicians based on shared patients.
2. **Lymphocyte Detection in Tissue Slides**: Contains tissue slide images with localized immune cells tagged based on their imaging features.
3. **Fake News Detection on Twitter**: Contains data on fake news propagation patterns on Twitter.
4. **Neuroscience Connectome Analysis**: Extract time-dependent correlations in fMRI to understand how connectivity evolves over time.

### Getting Started Guide
#### Prerequisites
- Basic understanding of Python, machine learning, and network analysis.

#### Environment Setup
1. **Install Python and necessary libraries**:
   ```bash
   pip install networkx pandas numpy torch torch_geometric opencv-python matplotlib seaborn
   ```
   [Install Torch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
2. **Recommended IDE**: Jupyter Notebook or VS Code.

#### Data Preparation
1. **Physician Sharing Network**:
   - Download the dataset from [Physician Sharing Network Dataset](https://example.com/physician-network-data).
   - Load and preprocess the data using pandas.
   - Construct the physician network graph using NetworkX.

2. **Lymphocyte Detection in Tissue Slides**:
   - Download the dataset from [Lymphocyte Detection Dataset](https://example.com/lymphocyte-data).
   - Load and preprocess the images using OpenCV.
   - Construct the spatial network of cells using NetworkX.

3. **Fake News Detection on Twitter**:
   - Download the dataset from [Fake News Detection Dataset](https://example.com/fake-news-data).
   - Load and preprocess the data using pandas.
   - Construct the social network graph using NetworkX.

#### Model Implementation
1. **Graph Neural Networks**:
   - Implement basic GNN models using PyTorch Geometric.
   - Compare GCNs and GATs in terms of performance and interpretability.
2. **Analysis and Visualization**:
   - Apply network analysis techniques and visualize the results using Matplotlib and Seaborn.

### Suggested Readings and References
- [Understanding Graph Neural Networks](https://medium.com/@ahmedmellit/understand-the-theoretical-foundations-of-graph-neural-networks-gnns-part-2-eb3a2a764e3e)
- [Graph Neural Networks in Healthcare](https://proceedings.mlr.press/v194/reddy22a.html)
- [FakeNewsNet: A Data Repository](https://github.com/KaiDMML/FakeNewsNet)
- [GNNs in Neuroscience: BrainGNN](https://medium.com/stanford-cs224w/gnns-in-neuroscience-graph-convolutional-networks-for-fmri-analysis-8a2e933bd802)

### Recommended Videos
- [Introduction to Graph Neural Networks](https://www.youtube.com/watch?v=36KmEod8pU8)
- [Deep Learning with Graphs](https://www.youtube.com/watch?v=jyxuY1h-hFo)
- [Introduction to Social Network Analysis](https://www.youtube.com/watch?v=36KmEod8pU8)

### Data Download Links
- [Physician Sharing Network Dataset](https://example.com/physician-network-data)
- [Lymphocyte Detection Dataset](https://example.com/lymphocyte-data)
- [Fake News Detection Dataset](https://example.com/fake-news-data)

### Project Files
- [Download Project Files (ZIP)](https://example.com/project-files.zip)

## Author Information
Joshua Levy - [GitHub](https://github.com/jlevy44) | [LinkedIn](https://www.linkedin.com/in/joshua-levy-87044913b) | [LevyLab](https://levylab.host.dartmouth.edu/)
