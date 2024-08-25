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
1. **Physician Sharing Network**: Two different networks. The tutorial focuses on physicians connected based on survey data describing their professional and personal interactions. https://schochastics.github.io/networkdata/reference/physicians.html [GNNs in Neuroscience: BrainGNN](https://medium.com/stanford-cs224w/gnns-in-neuroscience-graph-convolutional-networks-for-fmri-analysis-8a2e933bd802)
2. **Lymphocyte Detection in Tissue Slides**: Contains tissue histology images with localized immune cells tagged based on their imaging/immunofluorescence features. See https://proceedings.mlr.press/v194/reddy22a.html .
3. **Fake News Detection on Twitter**: Contains data on fake news propagation patterns on Twitter. See https://github.com/safe-graph/GNN-FakeNews https://www.sciencedirect.com/science/article/pii/S1568494623002533 .
4. **Neuroscience Connectome Analysis**: Extract time-dependent correlations in fMRI to understand how connectivity evolves over time.

### Getting Started Guide
#### Prerequisites
- Basic understanding of Python, machine learning, and network analysis. R is optional.
- See paper from: https://github.com/jlevy44/GCN4R

#### Environment Setup
1. **Install Python and necessary libraries**:
   ```bash
   pip install networkx pandas numpy torch opencv-python matplotlib seaborn scikit-learn plotly rpy2 cdlib libpysal spreg captum pysnooper fire
   ```
   [Install Torch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
2. **Recommended IDE**: Jupyter Notebook or VS Code.

#### Data Preparation and Tasks
1. **Physician Innovation Network**:
   - Download the dataset from [Physician Sharing Network Dataset](https://rdrr.io/github/schochastics/networkdata/man/physicians.html).
   - Load and preprocess the data using R/pandas. Already provided in 
   - Construct the physician network graph using NetworkX.
   - Task: Predict the year physicians adopted the new technology. Compare GNNs with classical logistic regression and other non-graph based methods.

2. **Lymphocyte Detection in Tissue Slides**:
   - Download the dataset from [Lymphocyte Detection Dataset](https://github.com/jlevy44/Cedars_AI_Campus_Tutorials/raw/main/Project7/lymphocyte_toy_data.pkl).
   - Load and preprocess the images using OpenCV.
   - Construct the spatial network of cells using NetworkX.
   - Description: Nodes: Cells, Attributes: CNN features, Edges: K-nearest neighboring cells.
   - Task: Predict whether a cell is immune or non-immune based on its spatial features. Compare GNNs with traditional image classification / MLP / etc. models.

3. **Fake News Detection on Twitter**:
   - Download the dataset from [Fake News Detection Dataset](https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.datasets.UPFD.html).
   - Load and preprocess the data using pandas.
   - Construct the social network graph using NetworkX.
   - Description: Nodes: Users, Edges: Retweets, Attributes: User profile and/or BERT-derived embeddings of tweets.
   - Task: Determine whether news is fake based on tweet text content and information from retweeting pattern. Does retweeting pattern inform whether news is fake? Compare GNNs with traditional text classification models on original tweet.
   
4. GNNs neuroscience:
   - Download the dataset from [GNNs in Neuroscience](https://colab.research.google.com/drive/16pZ3j3WZ5_E1oUa_70uz5Xb4ZVqHokMJ?usp=sharing)
   - Run through above notebook, may need to alter the code for latest package versions.
   - Description: fMRI signals within local brain regions of interest, which serve as the nodes. Edges are defined by the correlation between the fMRI signals within a fixed time period. Graph evolves over time as correlations within fixed intervals change.
   - Goal: Predict the age of the individual based on the evolving brain connectivity patterns. Compare GNNs with traditional time-series analysis methods, image analysis methods, average correlations, etc.. Interpret models with Captum, [GNNExplainer](https://pytorch-geometric.readthedocs.io/en/latest/modules/explain.html#philoshopy), etc. What are important time-points/connections?

#### Model Implementation
1. **Graph Neural Networks**:
   - Implement basic GNN models using PyTorch Geometric.
   - Compare GCNs and GATs in terms of performance and interpretability.
2. **Analysis and Visualization**:
   - Apply network analysis and compare predictive techniques and visualize the results using Matplotlib and Seaborn.

### Suggested Readings and References
- [Understanding Graph Neural Networks](https://medium.com/@ahmedmellit/understand-the-theoretical-foundations-of-graph-neural-networks-gnns-part-2-eb3a2a764e3e)
- [Graph Neural Networks in Healthcare](https://proceedings.mlr.press/v194/reddy22a.html)
- [FakeNewsNet: A Data Repository](https://github.com/KaiDMML/FakeNewsNet)
- [GNNs in Neuroscience: BrainGNN](https://medium.com/stanford-cs224w/gnns-in-neuroscience-graph-convolutional-networks-for-fmri-analysis-8a2e933bd802)

### Data Download Links
- [Physician Innovation Network Dataset, see X/A_physician.csv](https://github.com/jlevy44/Cedars_AI_Campus_Tutorials/tree/main/Project7) or load in R and convert to networkx via [R Data](https://rdrr.io/github/schochastics/networkdata/man/physicians.html)
- [Lymphocyte Detection Dataset](https://github.com/jlevy44/Cedars_AI_Campus_Tutorials/raw/main/Project7/lymphocyte_toy_data.pkl)
- [Fake News Detection Dataset](https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.datasets.UPFD.html)

### Project Files
- [Download Project Files (ZIP)](https://github.com/jlevy44/Cedars_AI_Campus_Tutorials/archive/refs/heads/main.zip)

### Getting Started
Check out tutorial notebooks here: [Tutorial Notebooks](https://github.com/jlevy44/Cedars_AI_Campus_Tutorials/tree/main/Project7)

## Author Information
Joshua Levy - [GitHub](https://github.com/jlevy44) | [LinkedIn](https://www.linkedin.com/in/joshua-levy-87044913b) | [LevyLab](https://levylab.host.dartmouth.edu/)
