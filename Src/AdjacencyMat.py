import numpy as np
import cv2
from collections import Counter
def getGroundTruth():
    file = 'C:/Users/achkr/OneDrive/Desktop/ENPM673/VisualOdom/dataset/poses/04.txt'
    return np.genfromtxt(file, delimiter=' ',dtype=None)

def cliqueBuilding(features_T, features_T_plus_1, threshold):
    M = AdjMatrix(features_T, features_T_plus_1, threshold)
    consistent_cliques = findConsCliques(M, 4)
    return consistent_cliques

def calculateDistance(feature1, feature2):

    distance = np.sum((feature1 - feature2) ** 2)    
    return distance

def AdjMatrix(f_T, f_T_plus_1, threshold):
    dist = np.sum((f_T[:, np.newaxis] - f_T_plus_1) ** 2, axis=2)
    #dist = dist.astype(np.float16)
    M = dist <= threshold
    np.fill_diagonal(M, 0)
    return M.astype(int)

def findConsCliques(M, max_size):
    c_cliques = []

    npoints = M.shape[0]
    visited = set()

    for i in range(npoints):
        if i not in visited:
            clique = [i]
            find_clique(M, clique, visited, c_cliques, max_size)

    return c_cliques



def find_clique(M, clique, visited, c_cliques, max_size):
    if len(clique) <= max_size:
            c_cliques.append(clique)

    neighbors = check_neighbors(M, clique[-1])
    counter= Counter()
    for c in clique:
        for n in neighbors:
            if M[n][c] is True:
                counter[n] += 1

    ccliques= [n for n in neighbors if counter[n] == len(clique)]
    for candidate in ccliques:
        if candidate not in visited:
            visited.add(candidate)
            clique.append(candidate)
            find_clique(M, clique, visited, c_cliques, max_size)
            clique.pop()
            visited.remove(candidate)

def check_neighbors(M, node):
    return [i for i, j in enumerate(M[node]) if j]

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

    # if len(p1) and len(p2) < 2:
    #     return None, None
    
    inliers = cliqueBuilding(p1, p2, threshold)
    indices = [idx for pair in inliers for idx in pair]

    return p1[indices], p2[indices]

def featureDetection():
    fast = cv2.FastFeatureDetector_create(threshold=12, nonmaxSuppression=True)
    return fast

def getImages(i):
    file_path = 'C:/Users/achkr/OneDrive/Desktop/ENPM673/VisualOdom/dataset/sequences/04/image_0/{0:06d}.png'.format(i)
    return cv2.imread(file_path, cv2.IMREAD_COLOR)

# def drawCliquesOnImage(img, cliques, points):
#     for clique in cliques:
#         color = np.random.randint(0, 255, size=3).tolist()  # Random color for each clique
#         for i in range(len(clique) - 1):
#             pt1_idx, pt2_idx = clique[i], clique[i + 1]
#             pt1 = tuple(map(int, points[pt1_idx]))
#             pt2 = tuple(map(int, points[pt2_idx]))
#             cv2.line(img, pt1, pt2, [0,255,0], thickness=2)  # Increase the line thickness to 2 (or any other value)
# def drawCliquesOnImage(img, cliques, points):
#     # Generate a random color for each clique
#     colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(cliques))]

#     # Draw edges between points that are part of the same clique
#     for i, clique in enumerate(cliques):
#         color = colors[i]
#         for pt_idx in clique:
#             pt = tuple(map(int, points[pt_idx]))
#             cv2.circle(img, pt, 1, color, -1)  # Draw a circle at each point in the clique

#         # Draw edges between connected points in the clique
#         # for i in range(len(clique)):
#         #     pt1_idx, pt2_idx = clique[i], clique[(i + 1) % len(clique)]  # Connect the last point with the first point in the clique
#         #     pt1 = tuple(map(int, points[pt1_idx]))
#         #     pt2 = tuple(map(int, points[pt2_idx]))
#         #     cv2.line(img, pt1, pt2, [255,0,0], thickness=5)

#####################################DRAW#############################################
def buildCliques(p1, p2, threshold):
    # Here, we assume that p1 and p2 are the keypoints for two images, and inliers contains pairs of indices.
    # We want to form cliques from these inliers, based on a threshold distance.

    cliques = []

    # Create a graph where each point in p2 is connected to other points within the threshold distance.
    graph = {}
    for i, point in enumerate(p2):
        graph[i] = set(j for j, pt in enumerate(p2) if np.linalg.norm(pt - point) <= threshold)

    # Find cliques in the graph (connected components).
    visited = set()
    for node in graph:
        if node not in visited:
            clique = set()
            stack = [node]
            while stack:
                curr_node = stack.pop()
                if curr_node not in clique:
                    clique.add(curr_node)
                    stack.extend(graph[curr_node])
            cliques.append(clique)

    return cliques
def drawCliquesOnImage(img, cliques, points):
    # Generate a random color for each clique
    colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(cliques))]

    # Draw edges between points that are part of the same clique
    for i, clique in enumerate(cliques):
        color = colors[i]
        for pt_idx in clique:
            pt = tuple(map(int, points[pt_idx]))
            cv2.circle(img, pt, 3, color, -1)  # Draw a slightly larger circle at each point in the clique

        # Draw edges between connected points in the clique to form a subgraph
        color = [0, 255, 0]  # Green color for lines between points in the same clique
        clique_list = list(clique)  # Convert the set into a list
        for i in range(len(clique_list)):
            pt1_idx, pt2_idx = clique_list[i], clique_list[(i + 1) % len(clique_list)]  # Connect the last point with the first point in the clique
            pt1 = tuple(map(int, points[pt1_idx]))
            pt2 = tuple(map(int, points[pt2_idx]))
            cv2.line(img, pt1, pt2, color, thickness=2)
######################################################################################################################

# Initialization
ground_truth = getGroundTruth()
video_name = 'Sequenc46.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter(video_name, fourcc, 20, (1200, 1200))

# video_name2 = 'Cliquepoints2.mp4'
# video_writer2 = cv2.VideoWriter(video_name2, fourcc, 20, (1241, 376))

img_1 = getImages(0)
img_2 = getImages(1)
gray_1 = img_1
gray_2 = img_2
detector = featureDetection()
kp1 = detector.detect(img_1)
p1 = np.array([ele.pt for ele in kp1], dtype='float32')
threshold = 1.0
def CameraCalib():
    # return   np.array([[7.188560000000e+02, 0, 6.071928000000e+02], #For Seq 0 1 2
    #           [0, 7.188560000000e+02, 1.852157000000e+02],
    #           [0, 0, 1]])
    # return   np.array([[7.215377000000e+02, 0, 6.095593000000e+02], #For Seq 4 5 
    #           [0, 7.215377000000e+02, 1.728540000000e+02],
    #           [0, 0, 1]])
    return   np.array([[ 7.070912000000e+02, 0, 6.018873000000e+02], #For seq 6 7 8 9
              [0, 7.070912000000e+02, 1.831104000000e+02],
              [0, 0, 1]])


# Camera parameters
fc = 718.8560
pp = (607.1928, 185.2157)
K = CameraCalib()
p1, p2   = featureTracking(gray_1, gray_2, p1,threshold)

# Rot_f = np.eye(3)
# trans_f = np.zeros((3, 1))
K  = CameraCalib()
print(len(p1),len(p2))
E, mask = cv2.findEssentialMat(p2, p1, fc, pp, cv2.RANSAC,0.999,1.0); 
_, R, t, mask = cv2.recoverPose(E, p2, p1,focal=fc, pp = pp);
Rot_f=R
trans_f=t
maxError = 0

traj = np.zeros((1200, 1200, 3), dtype=np.uint8)
MAX_FRAME=271 #Change Accordingly
MAX_FRAME2=801
ate_errors=[]
# Play image sequences
for numFrame in range(2, MAX_FRAME):
    print("Frame:", numFrame)

    curImage_c = getImages(numFrame)
    # curImage_c = getImages(numFrame + 1)
    # curImage = cv2.cvtColor(curImage_c, cv2.COLOR_BGR2GRAY)

    kp1 = detector.detect(curImage_c)
    kp2 = detector.detect(curImage_c)
    p1, p2 = featureTracking(gray_1, curImage_c, p1, threshold)

    if p1 is None or p2 is None:
        print("Insufficient feature correspondences. Skipping frame.")
        continue

    E, mask = cv2.findEssentialMat(p2, p1, fc, pp, cv2.LMEDS, 0.999, 1.0)
    _, R, t, mask = cv2.recoverPose(E, p2, p1, focal=fc, pp=pp)

    gt_x, gt_y, gt_z, abs_scale = Scale(ground_truth, numFrame)

    if abs_scale > 0.1:
        trans_f = trans_f + abs_scale * Rot_f.dot(t)
        Rot_f = R.dot(Rot_f)

    gray_1 = curImage_c
    kp1 = detector.detect(gray_1)
    p1 = np.array([ele.pt for ele in kp1], dtype='float32')
    p2 = np.array([ele.pt for ele in kp2], dtype='float32')
    # inliers = cliqueBuilding(p1, p2, threshold)
    # indices = [idx for pair in inliers for idx in pair]

    # # Draw cliques on the image
    # img_cliques = curImage_c.copy()
    # cliques = buildCliques(p1, p2, threshold)
    # drawCliquesOnImage(img_cliques, cliques, p2)
    d_x, d_y = int(trans_f[0]) + 300, int(trans_f[2]) + 100
    d_tx, d_ty = int(gt_x) + 300, int(gt_z) + 100

    curError = np.sqrt((trans_f[0] - gt_x) ** 2 + (trans_f[1] - gt_y) ** 2 + (trans_f[2] - gt_z) ** 2)
    print('Current Error:', curError)
    # ate_errors.append(curError)
    reference_distance = np.sqrt(gt_x**2 + gt_y**2 + gt_z**2)
    normalized_error = curError / reference_distance
    
    # Append the normalized error to the list of normalized ATE errors
    ate_errors.append(normalized_error)


    if curError > maxError:
        maxError = curError

    cv2.circle(traj, (d_x, d_y), 1, (0, 0, 255), 2)
    cv2.circle(traj, (d_tx, d_ty), 1, (255, 0, 0), 2)

    cv2.rectangle(traj, (10, 30), (700, 50), (0, 0, 0), cv2.FILLED)
    # text = "Coordinates: x={:.2f}m y={:.2f}m z={:.2f}m".format(float(trans_f[0]), float(trans_f[1]), float(trans_f[2]))
    # cv2.putText(traj, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    cv2.imshow('image', curImage_c)
    # cv2.imshow('Cliques', img_cliques)
    cv2.imshow("Trajectory", traj)
    video_writer.write(traj)
    # video_writer2.write(img_cliques)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
average_normalized_ate = (sum(ate_errors) / len(ate_errors)) * 100
print("Average ATE (%):", average_normalized_ate)
print('Maximum Error:', maxError)
cv2.imwrite('map.png', traj)
video_writer.release()
# video_writer2.release()
cv2.destroyAllWindows()
