import cv2
from itertools import combinations
import networkx as nx

import numpy as np
from scipy.spatial import cKDTree
###################Approximate Distance ###############################
def cliqueInlierDetection(features, threshold):
    pairwise_distances = approximatePairwiseDistances(features, threshold)
    inliers = np.ones(len(features), dtype=bool)

    for i in range(len(features)):
        if inliers[i]:
            neighbors = np.where(pairwise_distances[i] <= threshold)[0]
            inliers[neighbors] = True

    return inliers

def approximatePairwiseDistances(features, threshold):
    kdtree = cKDTree(features)
    pairwise_distances = kdtree.sparse_distance_matrix(kdtree, max_distance=threshold).toarray()

    return pairwise_distances

# ##########################CLIQUE DETECTION########################################
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def cliqueBuilding(features_T, features_T_plus_1, threshold):
    # Create an empty graph
    n_components = min(1000, len(features_T[0]))
    #n_components = 3  # Default value
    if len(features_T) > 0:
        n_components = min(3, len(features_T[0]))
    pca = PCA(n_components=n_components)
    reduced_features_T = pca.fit_transform(features_T)
    reduced_features_T_plus_1 = pca.transform(features_T_plus_1)
    graph_T = createGraph(reduced_features_T, threshold)
    graph_T_plus_1 = createGraph(reduced_features_T_plus_1, threshold)

    # Find consistent cliques
    consistent_cliques = findCCliques(graph_T, graph_T_plus_1,4)
    # nx.draw(graph_T, with_labels=True)
    # plt.show()

    return consistent_cliques


def createGraph(features, threshold):
    graph = nx.Graph()

    # Add nodes to the graph
    for i, feature in enumerate(features):
        graph.add_node(i, feature=feature)
    # print("THESE ARE FEATURES",features)

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            distance = calculateDistance(features[i], features[j])
            if distance <= threshold:
                graph.add_edge(i, j)

    return graph

def calculateDistance(feature1, feature2):

    distance = np.sum((feature1 - feature2) ** 2)
    return distance

def findCCliques(graph_T, graph_T_plus_1, max_clique_size):
    consistent_cliques = []    
    for node in graph_T.nodes:
        neighbors_T = set(graph_T.neighbors(node))
        neighbors_T_plus_1 = set(graph_T_plus_1.neighbors(node))
        common_neighbors = neighbors_T.intersection(neighbors_T_plus_1)
        cliques = nx.find_cliques(graph_T.subgraph(common_neighbors))
        filtered_cliques = [clique for clique in cliques if len(clique) <= max_clique_size]
        consistent_cliques.extend(filtered_cliques)

    return consistent_cliques

def Scale(f, frame_id):
      print("FRAME ID",f.shape)
      x_pre, y_pre, z_pre = f[frame_id-1][3], f[frame_id-1][7], f[frame_id-1][11]
      x    , y    , z     = f[frame_id][3], f[frame_id][7], f[frame_id][11]
      scale = np.sqrt((x-x_pre)**2 + (y-y_pre)**2 + (z-z_pre)**2)
      return x, y, z, scale

def featureTracking(img_1, img_2, p1, threshold):
    lk_params = dict(winSize=(11, 11), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
    st = st.reshape(st.shape[0])
    
    # find good ones
    p1 = p1[st == 1]
    p2 = p2[st == 1]
    inliers = cliqueBuilding(p1, p2, threshold)
    indices = [idx for pair in inliers for idx in pair]
    # inliers = np.array(inliers, dtype=int)
    return p1[indices], p2[indices]

# def featureTracking(img_1, img_2, p1, threshold):
#     lk_params = dict(winSize=(11, 11), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

#     p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
#     st = st.reshape(st.shape[0])
    
#     # find good ones
#     p1 = p1[st == 1]
#     p2 = p2[st == 1]
#     inliers = cliqueInlierDetection(p1,threshold)
#     # _,inliers = RANSAC(p1, p2)
#     # extract individual indices from tuples
#     # indices = [idx for pair in inliers for idx in pair]
#     # inliers = np.array(inliers, dtype=int)
#     return p1[inliers], p2[inliers]


def featureDetection():
    thresh = dict(threshold=40, nonmaxSuppression=True);
    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    return fast

def getGroundTruth():
    file = 'C:/Users/achkr/OneDrive/Desktop/ENPM673/VisualOdom/dataset/poses/02.txt'
    return np.genfromtxt(file, delimiter=' ',dtype=None)

def getFrames(i):
    file_path = 'C:/Users/achkr/OneDrive/Desktop/ENPM673/VisualOdom/dataset/sequences/02/image_0/{0:06d}.png'.format(i)
    print(file_path)
    return cv2.imread(file_path, 0)

def getCameraCalib():
    return   np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
              [0, 7.188560000000e+02, 1.852157000000e+02],
              [0, 0, 1]])
    # return   np.array([[7.215377000000e+02, 0, 6.095593000000e+02],
    #           [0, 7.215377000000e+02, 1.728540000000e+02],
    #           [0, 0, 1]])
    # return   np.array([[ 7.070912000000e+02, 0, 6.018873000000e+02],
    #         [0, 7.070912000000e+02, 1.831104000000e+02],
    #         [0, 0, 1]])


#initialization
ground_truth = getGroundTruth()
video_name = 'img2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter(video_name, fourcc, 20, (1200,1200))
img_1 = getFrames(0)
print(img_1)
img_2 = getFrames(1)

# if len(img_1) == 3:
# 	gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
# 	gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
gray_1 = img_1
gray_2 = img_2
# cv2.imshow("GRAY",gray_1)

#find the detector
detector = featureDetection()
kp1      = detector.detect(img_1)
p1       = np.array([ele.pt for ele in kp1],dtype='float32')
threshold = 5.0  # Adjust the threshold value as needed
p1, p2   = featureTracking(gray_1, gray_2, p1,threshold)
# p3, p4   = featureTracking2(gray_1, gray_2, p1,threshold)
total_frames = 4000
processed_frames = 0

print(p1.shape)

#Camera parameters
fc = 718.8560
pp = (607.1928, 185.2157)
K  = getCameraCalib()
print(len(p1),len(p2))
E, mask = cv2.findEssentialMat(p2, p1, fc, pp, cv2.RANSAC,0.999,1.0); 
_, R, t, mask = cv2.recoverPose(E, p2, p1,focal=fc, pp = pp);


#initialize some parameters
MAX_FRAME 	  = 4660
MAX_FRAME2 	  = 2741
minimum  = 1500
MAX_FRAME6 	  = 1101
initfeature = p2
firstImg   = gray_2

rot_f = R
trans_f = t
# rot_f2 = R3
# trans_f2 = t3
print(t)
print(R)

traj = np.zeros((1200, 1200, 3), dtype=np.uint8)

maxError = 0
ate_errors=[]

#play image sequences
for numFrame in range(2, MAX_FRAME):
    print("Frame Number :",numFrame)
    if (len(initfeature) < minimum):
        feature   = detector.detect(firstImg)
        initfeature = np.array([ele.pt for ele in feature],dtype='float32')
    curImage = getFrames(numFrame)

    # if len(curImage_c) == 3:
    #     curImage = cv2.cvtColor(curImage_c, cv2.COLOR_BGR2GRAY)
    # else:
    #     curImage = curImage_c
    
    kp1 = detector.detect(curImage);
    initfeature, curFeature = featureTracking(firstImg, curImage, initfeature,threshold)
    initfeature2, curFeature2 = featureTracking(firstImg, curImage, initfeature,threshold)
    print("Done for",numFrame,numFrame+1)
    E, mask = cv2.findEssentialMat(curFeature, initfeature, fc, pp, cv2.LMEDS,0.999,1.0); 
    _, R, t, mask = cv2.recoverPose(E, curFeature, initfeature, focal=fc, pp = pp);
    
    print(t)
    gt_x, gt_y, gt_z, absolute_scale = Scale(ground_truth, numFrame)

    if absolute_scale > 0.1:  
        trans_f = trans_f + absolute_scale*rot_f.dot(t)
        rot_f = R.dot(rot_f)

    firstImg = curImage
    initfeature = curFeature
    

    ####Visualization of the result
    d_x, d_y = int(trans_f[0]) + 300, int(trans_f[2]) + 100;
    # d_x2, d_y2 = int(trans_f2[0]) + 300, int(trans_f2[2]) + 100;
    d_tx, d_ty = int(gt_x) + 300, int(gt_z) + 100
    print(d_x, d_y)
    curError = np.sqrt((trans_f[0]-gt_x)**2 + (trans_f[1]-gt_y)**2 + (trans_f[2]-gt_z)**2)
    print('Current Error: ', curError)
    # ate_errors.append(curError)
    reference_distance = np.sqrt(gt_x**2 + gt_y**2 + gt_z**2)
    normalized_error = curError / reference_distance
    
    # Append the normalized error to the list of normalized ATE errors
    ate_errors.append(normalized_error)

    if (curError > maxError):
        maxError = curError

    cv2.circle(traj, (d_x, d_y) ,1, (0,0,255), 2);
    # cv2.circle(traj, (d_x2, d_y2) ,1, (0,255,0), 2);
    cv2.circle(traj, (d_tx, d_ty) ,1, (255,0,0), 2);

    cv2.rectangle(traj, (10, 30), (700, 50), (0,0,0), cv2.FILLED);
    text = "Coordinates: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(trans_f[0]), float(trans_f[1]), float(trans_f[2]));
    cv2.putText(traj, text, (10,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8);
    

    processed_frames = 0

    
    # elapsed_time = time.time() - start

    # # Calculate the average time per frame
    # avg_time_per_frame = elapsed_time / processed_frames if processed_frames > 0 else 0

    # remaining_frames = total_frames - processed_frames
    # estimated_remaining_time = avg_time_per_frame * remaining_frames if avg_time_per_frame > 0 else 0
    # processed_frames+=1
    # # Print the ETA
    # print("Processed frames:", processed_frames)
    # print("ETA:", estimated_remaining_time, "seconds")


#   cv2.drawKeypoints(curImage, kp1, curImage_c)
    cv2.imshow('image', curImage)
    cv2.imshow( "Trajectory", traj );
    video_writer.write(traj)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

#Average Translation Error Calculation
average_normalized_ate = (sum(ate_errors) / len(ate_errors)) * 100
print("Average ATE (%):", average_normalized_ate)
print('Maximum Error: ', maxError)
cv2.imwrite('map.png', traj);
video_writer.release()
cv2.destroyAllWindows()