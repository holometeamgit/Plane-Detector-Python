import numpy as np
import cv2
import imutils
import math
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def get_matches(frames_descriptors, frames_keypoints):
    matches = []
    kps_inds = []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for i in range(1, len(frames_descriptors)):
        prev_des = frames_descriptors[i - 1]
        cur_des = frames_descriptors[i]

        if (prev_des is None) or (cur_des is None):
            matches.append([])
        else:
            kp1 = frames_keypoints[i - 1]
            kp2 = frames_keypoints[i]
            cur_matches = bf.match(prev_des, cur_des)
            cur_points_matches = np.int32([ [kp1[m.queryIdx].pt, kp2[m.trainIdx].pt] for m in cur_matches ])
            cur_kps = np.int32([ [m.queryIdx, m.trainIdx] for m in cur_matches ])
            matches.append(cur_points_matches)
            kps_inds.append(cur_kps)

    return (matches, kps_inds)


def get_kp_des(file_name, points_count = 150):
    all_gray_frames = []
    all_rgb_frames = []
    all_kp = []
    all_des = []
    cap = cv2.VideoCapture(file_name)
    orb = cv2.ORB_create(3500) 
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.transpose(gray)
        gray = cv2.resize(gray, dsize=(480, 640), interpolation=cv2.INTER_CUBIC)
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb = np.transpose(img_rgb, (1, 0, 2))
        img_rgb = cv2.resize(img_rgb, dsize=(480, 640), interpolation=cv2.INTER_CUBIC)

        kp = orb.detect(gray, None)
        kp, des = orb.compute(gray, kp)
        if len(kp) > points_count:
            all_gray_frames.append(gray)
            all_rgb_frames.append(img_rgb)
            all_kp.append(kp)
            all_des.append(des)
    
    return (all_rgb_frames, all_gray_frames, all_kp, all_des)

# TODO many matches (now only first)
def filter_with_prev_matches(all_matches, frame_index, point_index, 
                             distance = 10, isForward = True):
    cur_index = frame_index - 1
    match = all_matches[frame_index - 1] # match (frame_index-1, frame_index)
    
    cur_index = frame_index
    cur_point = match[point_index, 1, :]
    prev_point = match[point_index, 0, :]
#     prev
    isRun = False
    while (isRun):
        cur_index = cur_index - 1
        if (cur_index >= len(all_matches)):
            isRun = False
            break
        cur_match = all_matches[cur_index]
        points_prev_from = cur_match[:,0,:]
        points_prev_to = cur_match[:,1,:]
        cur_ind = np.where(np.all(points_prev_to == cur_point,axis=1))[0]
        if len(cur_ind) == 0:
            isRun = False
            break
        
        point_ind = cur_ind[0]
        cur_point = points_prev_from[point_ind]
        
    cur_dist = frame_index - cur_index - 1
    if (cur_dist >= distance):
        return True
 #     forward       
    isRun = True
    cur_index = frame_index - 1
    while ( (isRun) and (isForward) ):
        cur_index = cur_index + 1
        cur_match = all_matches[cur_index]
        points_forward_from = cur_match[:,0,:]
        points_forward_to = cur_match[:,1,:]       
        cur_ind = np.where(np.all(points_forward_from == cur_point,axis=1))[0]
        if len(cur_ind) == 0:
            isRun = False
            break
        
        point_ind = cur_ind[0]
        cur_point = points_forward_to[point_ind]
        
    cur_dist = cur_dist + ( cur_index - frame_index + 1 )
    if (cur_dist >= distance):
        return True
    
    return False
    

def get_frame_with_ind(ind, dist = 5):
    cur_frame = all_rgb_frames[ind].copy()
    cur_kps = all_kp[ind].copy()
    cur_points = np.int32([cur_kps[ind].pt for ind in range(len(cur_kps))])
    points = []
#     print(all_matches[ind+1])
#     print(cur_points)
    for i in range(len(all_matches[ind-1])):
        is_true = filter_with_prev_matches(all_matches, ind, i, dist)
        if is_true:
            cur_point = all_matches[ind+1][i][0]
            points.append(cur_point)

    for point in points:
        cur_frame = cv2.circle(cur_frame, (point[0], point[1]), 3, 255, 2)

    return cur_frame

def get_points_of_frame(desc, kps, frame_ind, matches, matches_kp_inds, dist = 5):
    
    cur_kps = kps[frame_ind].copy()
    cur_matches = matches[frame_ind]
    cur_descs = desc[frame_ind].copy()
    points = []
    points_kps = []
    points_desc = []
    for i in range(len(all_matches[frame_ind-1])):
        is_true = filter_with_prev_matches(matches, frame_ind, i, dist)
        if is_true:
#             print(len(cur_kps))
            
            cur_kp_ind = matches_kp_inds[frame_ind-1][i][1]
#             print(cur_kp_ind)
            cur_kp = cur_kps[cur_kp_ind]
#             print(cur_kp_ind)
            cur_desc = cur_descs[cur_kp_ind]
            cur_point = cur_kp.pt
            
            points.append(cur_point)
            points_kps.append(cur_kp)
            points_desc.append(cur_desc)
        
    return (points, points_kps, points_desc)

def draw_points_in_frame(points, frame):
    for point in points:
#         print(point)
        frame = cv2.circle(frame, (int(point[0]), int(point[1])), 3, 255, 2)
        
    return frame


def dest_between_points(point1, point2):
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    distance = np.sqrt(dx*dx + dy*dy)
    
    return distance

def find_good_points(H, points_from, points_to, max_dest = 5):
    dest_points = points_to.copy()
    ind_from = []
    ind_to = []
    for i in range(len(points_from)):
        cur_point_from = np.array([points_from[i][0], points_from[i][1], 1.0])
        calk_dest_cur_point = np.dot(H, cur_point_from)
        calk_dest_cur_point_2d = np.array([calk_dest_cur_point[0]/calk_dest_cur_point[2], 
                                          calk_dest_cur_point[1]/calk_dest_cur_point[2]])
        for j in range(len(dest_points)):
            cur_dest_point = dest_points[j]
            dist = dest_between_points(calk_dest_cur_point_2d, cur_dest_point)
            if (dist < max_dest) and ((j in ind_to) == False):
                ind_from.append(i)
                ind_to.append(j)
                break
            
    mask_from = np.zeros(len(points_from))
    mask_to = np.zeros(len(dest_points))
    
    for i in ind_from:
        mask_from[i]  = 1
        
    for i in ind_to:
        mask_to[i]  = 1
        
    return (mask_from, mask_to)


def find_H_mask_between_points(kp1, des1, kp2, des2):
    des1 = np.uint8(des1)
    des2 = np.uint8(des2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    if (len(matches) == 0):
        return []
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def get_points_to_match(masks, scan_points_to_mask, scan_kps_to_mask, scan_desc_to_mask, dist = 20):
    masked_scan_points = []
    masked_scan_kps = []
    masked_scan_desc = []
    for i in range(len(scan_points_to_mask)):
        cur_points = np.array( scan_points_to_mask[i].copy() )
        cur_kps = np.array( scan_kps_to_mask[i].copy() )
        cur_desc = np.array( scan_desc_to_mask[i].copy() )
        mask = masks[i].copy()
        mask = mask > dist

        cur_points = cur_points[(mask)==True]
        cur_kps = cur_kps[(mask)==True]
        cur_desc = cur_desc[(mask)==True]
        
        masked_scan_points.append(cur_points)
        masked_scan_kps.append(cur_kps)
        masked_scan_desc.append(cur_desc)
        
    return (masked_scan_points, masked_scan_kps, masked_scan_desc)


        
def homography_points_transform(H, points):
    points_to = []
    for i in range(len(points)):
#         print(points[i])
        cur_point_to = homography_point_transform(H, points[i])
        points_to.append(cur_point_to)
        
    return points_to
    
def homography_point_transform(H, point):
    point_from = np.array([point[0], point[1], 1])
    calk_dest_cur_point = np.dot(H, point_from)
    calk_dest_cur_point_2d = np.array([calk_dest_cur_point[0]/calk_dest_cur_point[2], 
                                          calk_dest_cur_point[1]/calk_dest_cur_point[2]])
    return calk_dest_cur_point_2d

def get_mask_of_points(frame_points, frame_points_kp, frame_points_des, points_to_match, kps_to_match, desc_to_match):
#     frame_points_kp, frame_points_des
    mask = np.zeros(len(frame_points_kp))
    for i in range(len(kps_to_match)):
        H_i = find_H_mask_between_points(frame_points_kp, frame_points_des, kps_to_match[i], desc_to_match[i])
        if ((H_i is None) == False):   
            if (len(H_i) > 0):
                (m_cur, mi) = find_good_points(H_ik, frame_points, points_to_match[i])
                mask = mask + m_cur
        
    return mask

def get_homo_chain(ind):
    chain = []
    cur_ind = ind
    chain.append(cur_ind)
    while cur_ind > 0:
        if cur_ind == 35:
            cur_ind = 14
        else:
            cur_ind = cur_ind - 1
        chain.append(cur_ind)
        
    return chain
        

# def find_matches_count(ind1, ind2, cur_scan_kps, cur_scan_desc):
#     print(len(cur_scan_desc))
#     des1 = np.uint8(cur_scan_desc[ind1].copy())
#     des2 = np.uint8(cur_scan_desc[ind2].copy())
#     kp1 = scan_kps[ind1].copy()
#     kp2 = scan_kps[ind2].copy()
#     matches = bf.match(des1,des2)

#     if (len(matches) == 0):
#         return []
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
#     H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

#     return (len(matches), np.count_nonzero(mask))


def get_H_scan(kps, desc):
    H = []
    for i in range(1, len(desc)):
        if (i==36):
            kp1 = kps[14]
            kp2 = kps[36]
            des1 = desc[14]
            des2 = desc[36]
            cur_H = find_H_mask_between_points(kp1, des1, kp2, des2)
            H.append(cur_H)

        else:
            kp1 = kps[i - 1]
            kp2 = kps[i]
            des1 = desc[i - 1]
            des2 = desc[i]
            cur_H = find_H_mask_between_points(kp1, des1, kp2, des2)
            H.append(cur_H)
            
    return H


# i=36 H[35] 14-36
# i=15 H[14] : 14-15

def get_homo_chain(ind):
    chain = []
    cur_ind = ind
    chain.append(cur_ind)
    while cur_ind > 0:
        if cur_ind == 35: #35
            cur_ind = 14 #14
        else:
            cur_ind = cur_ind - 1
        chain.append(cur_ind)
        
    chain.reverse()
    
    return chain

def get_homo_from_first(ind, Hs):
    chain = get_homo_chain(ind - 1)
    H = np.eye(3)
    for i in range(len(chain)):
        cur_H = Hs[chain[i]]
        H = np.dot(cur_H, H)
        
    return H


def find_H(kp1, des1, kp2, des2):
    des1 = np.uint8(des1)
    des2 = np.uint8(des2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    if (len(matches) == 0):
        return []
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
    matches = np.array(matches)

    return (H, matches[mask.reshape(-1) == 1])

def points_with_H(points, H):
    points = np.array(points)
    if points.shape[1] == 2:
        new_points = np.ones((points.shape[0], 3))
        new_points[:,:2] = points
        points = new_points
        
    points = np.dot(H, points.T).T
    points[:,0] = points[:,0] / points[:,2]
    points[:,1] = points[:,1] / points[:,2]
    return points[:,:2]

def find_good_points_with_opt(H, points_from, points_to, max_dest = 5):
    points_from = np.array(points_from)
    points_after_h = points_with_H(points_from, H)
    points_to = np.rint(np.array(points_to) / max_dest)
    points_after_h = np.rint(points_after_h / max_dest)
    inds_from = []
    inds_to = []
    a1 = points_after_h[:,0] * 1000 + points_after_h[:,1]
    b1 = points_to[:,0] * 1000 + points_to[:,1]
    
    mask_from = np.zeros(points_after_h.shape[0])
    mask_to = np.zeros(points_to.shape[0])
    
    for i in range(points_after_h.shape[0]):
        itemindex = np.array(np.where(b1==a1[i]))
        if (itemindex.shape[1] > 0):
            ind_to = itemindex[0,0]
            inds_from.append(i)
            inds_to.append(ind_to)
            mask_from[i] = 1
            mask_to[ind_to] = 1
            
    return (inds_from, inds_to, mask_from, mask_to)


def norm_vector(vector):
    np.sum([comp * comp for comp in vector])
    vector_len = np.sqrt(np.sum([comp*comp for comp in vector]))
    return vector / vector_len
    
    
def angle_between_vecs(vec1, vec2):
    vec1 = norm_vector(vec1)
    vec2 = norm_vector(vec2)
    cos = np.sum(np.dot(vec1, vec2))
    angle = np.arccos(cos)
    return angle

def get_H_distance(H, cx = 240, cy = 320):
    point1 = np.array([cx, cy])
    point2 = np.array([cx, cy - 50])

    points_n = homography_points_transform(H, [point1, point2])
    point1_n = points_n[0]
    point2_n = points_n[1]

    vec1 = point2 - point1
    vec2 = point2_n - point1_n
    angle = angle_between_vecs(vec1, vec2)
    dist = np.sqrt(np.sum([comp*comp for comp in (point1_n - point1)]))
    
    return (angle, dist)

def get_max_score(cur_kp, cur_des, cur_points, scan_inds):
    max_score = 0
    ind = -1
    for j in range(len(scan_inds)):
        kp = all_kp[scan_inds[j]]
        des = all_des[scan_inds[j]]
        points = [(int(k.pt[0]), int(k.pt[1])) for k in kp]
        
        cur_h = find_H_mask_between_points(kp, des, cur_kp, cur_des)
        (cur_ind_from, cur_ind_to, m_from, m_to) = find_good_points_with_opt(cur_h, points, cur_points)
        first_j = float(np.count_nonzero(m_from)) / len(m_from)
        secound_j = float(np.count_nonzero(m_to)) / len(m_to)
        score_j = first_j + secound_j
        if score_j > max_score:
            max_score = score_j
            ind = j
            
    return (max_score, j)
            
def get_min_distance(H_to_first, scan_inds, Hs):
    mind = 10000
    mina = 10000
    for j in range(len(scan_inds)):
        cur_H = Hs[j]
        Hj = np.dot(H_to_first, np.linalg.inv(cur_H))
        (a,d) = get_H_distance(Hj)
        if (mind > d):
            mind = d
            
        if mina > a:
            mina = a
            
    return (mind, mina)

