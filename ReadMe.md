
# Visual Odometry Using Clique based Inlier Detection

This project mainly focuses on obtaining inlier points for estimating camera pose using maximal cliques technique involving bron-kerbrosch algorithm.

## Dependencies

The following librarires have been used and are quintessential in running the graphical node frameworks.
```bash
  pip3 install networkx
  pip3 install Collections
  pip install -U scikit-learn
```
Make sure you have the latest version of OpenCV (supported 3.4+) and Numpy installed prior to running the code.

KITTI DATASET CAN BE DOWNLOADED FROM : https://www.cvlibs.net/datasets/kitti/eval_odometry.php

## Instructons to run the code: 
- git clone the files from the repository.
- Run "python3 CliqueDetection(1).py" to run the code.
- Default methods : FAST feature detction and Seq6.
- For running method2 - run python3 AdjacencyMat.py on the terminal
- Default methods : FAST feature detction and Seq6.

## Clique Identifier 
The proposed method will receive **n** feature points from the FAST feature detection technique, and subsequently, it will perform the following steps to build cliques for consistent feature correspondences in subsequent frames:

1. **Dimensionality Reduction using PCA**: To handle a large number of feature points efficiently, the method employs Principal Component Analysis (PCA) to reduce the dimensionality of the feature vectors. This step helps in retaining the most informative dimensions of the features while reducing computational complexity. The number of components is set to a value n_components (usually a small value like 3) or the number of features points, whichever is smaller.

![sample graph](https://github.com/Achuthankrishna/Visual_Odom_Clique/blob/main/Results/sample%20graph.png)

2. **Graph Creation for Current and Next Frames**: The reduced feature points are used to create graphs for both the current frame (graph_T) and the next frame (graph_T_plus_1). Each node in the graph corresponds to a feature point, and edges between nodes are added if the distance between the corresponding feature vectors is below a certain threshold.

3. **Finding Consistent Cliques**: For each node in the graph_T, the algorithm identifies its neighbors in both graph_T and graph_T_plus_1. The common neighbors between these two frames are determined. Subsequently, cliques (complete subgraphs) are extracted from the subgraph formed by these common neighbors. The cliques are filtered to ensure their size does not exceed a predefined max_clique_size.

## Results
 