import math
import numpy as np
import os
import time
import json
import math
import scipy
from scipy.spatial.distance import cdist
from scipy import misc
import cv2
import tensorflow as tf
import copy
import FaceRecognition.detect_face
import FaceRecognition.facenet as facenet
from FaceRecognition.detect_recognize import *
from PeronReid.query import *

idToName_dict = {0: 'unknown', 1: 'feiguoyou', 2: 'jiyangyang', 3: 'lixin', 4: 'qiankun', 5: 'tianzhen',
                 6: 'wangtengfei', 7: 'yulei', 8: 'zhanglei', 9: 'internet2', 10: 'mengqingsen', 11: 'huoqishuai'}
nameToId_dict = {'unknown': 0, 'feiguoyou': 1, 'jiyangyang': 2, 'lixin': 3, 'qiankun': 4, 'tianzhen': 5,
                 'wangtengfei': 6, 'yulei': 7, 'zhanglei': 8, 'internet2': 9, 'mengqingsen': 10, 'huoqishuai': 11}
idToMethod = {1: 'face', 2: 'reid', 3: 'gait'}


def trans_coord(keypoints):
    if len(keypoints) == 0:
        return []
    num_keypoints = int(len(keypoints) / 3)
    coords_ori_1 = keypoints[3:5]
    coords_ori_8 = keypoints[24:26]
    ori_new = coords_ori_8
    scale = math.sqrt(sum([(coords_ori_1[i] - coords_ori_8[i]) ** 2 for i in range(len(coords_ori_8))]))
    for key_idx in range(num_keypoints):
        single_keypoint = keypoints[key_idx * 3:(key_idx + 1) * 3]
        if not single_keypoint[0] or not single_keypoint[1]:
            continue
        else:
            keypoints[key_idx * 3:(key_idx + 1) * 3 - 1] = [(single_keypoint[j] - ori_new[j]) / scale for j in range(2)]
    return keypoints


def match_feature_and_embeddings(feature, embeds_gt, label_gt, thresh_dist=0.8):
    result = []
    result_dist = []
    dist = scipy.spatial.distance.cdist(feature, embeds_gt)
    min_dist = np.min(dist, axis=1)
    label_idx = np.argmin(dist, axis=1)
    for d, idx in zip(min_dist, label_idx):
        result_dist.append(d)
        if d < thresh_dist:
            result.append(label_gt[idx])
        else:
            result.append('unknown')
    return result, min_dist


class FrameInfo:
    def __init__(self):
        # How to parse a record line:
        # 0frame_count, 1id, 2yolo position * 4, 6process necessity, 7keypoints * 75, 82face ability, 83reid ability,
        # 84gait ability, 85face position *4, 89body position * 4, 93process method, 94face result, 95face dist,
        # 96reid result, 97reid dist, 98gait result, 99gait dist, 100proceessed, 101final result, 102final method,
        # 103start time
        self.info_character_num = 104
        self.record_info = ['id', 'recv_times', 'process_method', 'embeds_dist', 'result', 'latest_process',
                            'latest_result', 'latest_method', 'latest_dist', 'keypoints', 'isInFront']
        self.frame_info_record = np.zeros((1, self.info_character_num)) - 1
        self.current_frame_info = np.zeros((1, self.info_character_num)) - 1
        self.id_record = []
        self.id_status = []  # In front: 1, Disappear: 0
        self.frame_count = 1
        self.current_frame_start_time = time.time()

    def process_necessity_judge(self, idx):
        if not len(self.frame_info_record):
            return 1
        else:
            this_idx = np.where(self.frame_info_record[:, 1] == idx)
            this_id_info_record = self.frame_info_record[this_idx]
            processed_idx = np.where(this_id_info_record[:, 100] > 0)[0]
            if len(processed_idx):
                last_process = processed_idx[-1]
                # If haven't process for a long time process anyway
                if self.frame_count - last_process > 75:
                    if np.any(0 < this_id_info_record[0, 95] < 0.4):
                        return 0
                    else:
                        return 1
                else:
                    # Processed recently but result is not confident
                    dist_idx = [95, 97, 99]
                    dist = this_id_info_record[last_process][dist_idx]
                    # if result is confident, don't need to process recently
                    if 0 < dist[0] < 0.5 or 0 < dist[1] < 0.7 or 0 < dist[2] < 8:
                        return 0
                    else:
                        return 1
            else:
                return 0

    def update_with_yolo_info(self, yolo_res):  # start of the process
        self.current_frame_start_time = time.time()
        self.current_frame_info = np.zeros((1, self.info_character_num)) - 1
        if len(yolo_res):
            for single_res in yolo_res:
                if single_res[1] in self.id_record:
                    new_frame_id_info = np.zeros((1, self.info_character_num)) - 1
                    new_frame_id_info[0][:6] = [self.frame_count, single_res[1], single_res[2][0], single_res[2][1],
                                                single_res[2][2], single_res[2][3]]
                    new_frame_id_info[0][6] = self.process_necessity_judge(single_res[1])
                    self.current_frame_info = np.vstack((self.current_frame_info, new_frame_id_info))
                else:
                    new_frame_id_info = np.zeros((1, self.info_character_num)) - 1
                    new_frame_id_info[0][:6] = [self.frame_count, single_res[1], single_res[2][0], single_res[2][1],
                                                single_res[2][2], single_res[2][3]]
                    new_frame_id_info[0][6] = 1
                    self.current_frame_info = np.vstack((self.current_frame_info, new_frame_id_info))
                    self.id_record.append(single_res[1])
            self.current_frame_info = self.current_frame_info[1:]

    def update_with_openpose_keypoints(self, keypoints):
        body_keypoints1 = [1, 2, 5, 8, 9, 10, 12, 13]
        for current_frame_id in self.current_frame_info:
            for kk in keypoints:
                main_body = kk[body_keypoints1, :][kk[body_keypoints1, 2] > 0]
                main_body_x = main_body[:, 0]
                main_body_y = main_body[:, 1]
                if not len(main_body_x) or not len(main_body_y):
                    continue
                main_body_left, main_body_right = min(main_body_x), max(main_body_x)
                main_body_top, main_body_bottom = min(main_body_y), max(main_body_y)
                hmin, hmax = current_frame_id[2] - 20, current_frame_id[4] + 20
                vmin, vmax = current_frame_id[3] - 20, current_frame_id[5] + 20
                # Are keypoints in the bbox of yolo
                if hmin < main_body_left < main_body_right < hmax and vmin < main_body_top < main_body_bottom < vmax:
                    kk = np.ravel(kk)
                    if sum(kk[3:6]) and sum(kk[24:27]):
                        kk = kk.copy().tolist()
                        # kk = trans_coord(kk)
                        current_frame_id[7:82] = kk
                    else:
                        current_frame_id[7:82] = [0] * len(current_frame_id[7:82])
                    break

    def face_reid_recognition_ability(self):
        # 9 is the score of the first keypoint
        face_keypoints1_score_idx = [9, 9 + 3 * 15, 9 + 3 * 16, 9 + 3 * 17]
        face_keypoints2_score_idx = [9, 9 + 3 * 15, 9 + 3 * 16, 9 + 3 * 18]
        body_keypoints1_score_idx = [9, 9 + 3 * 1, 9 + 3 * 8, 9 + 3 * 9, 9 + 3 * 10, 9 + 3 * 11]
        body_keypoints2_score_idx = [9, 9 + 3 * 1, 9 + 3 * 8, 9 + 3 * 12, 9 + 3 * 13, 9 + 3 * 14]
        body_keypoints3_score_idx = [9 + 3 * 1, 9 + 3 * 8, 9 + 3 * 12, 9 + 3 * 13, 9 + 3 * 14, 9 + 3 * 19, 9 + 3 * 18]
        for info in self.current_frame_info:
            if np.all(info[face_keypoints1_score_idx] > 0.5) or np.all(info[face_keypoints2_score_idx] > 0.5):
                # face ability = 1
                info[82] = 1
            if np.all(info[body_keypoints1_score_idx] > 0.1) or np.all(info[body_keypoints2_score_idx] > 0.1):
                # reid ability = 1
                info[83] = 1

    def face_recognition(self, raw_frame, pnet, rnet, onet, sess, images_placeholder,
                         phase_train_placeholder, embeddings, face_embeds_gt, face_labels_gt):
        self.face_reid_recognition_ability()
        face_keypoints_x_idx = [7, 7 + 3 * 15, 7 + 3 * 16]
        face_keypoints_y_idx = [8, 8 + 3 * 15, 8 + 3 * 16]

        # if any of id need to do face recognition
        if np.any(np.logical_and(self.current_frame_info[:, 6] > 0, self.current_frame_info[:, 82] > 0)):
            # Process face detection
            face, face_boxes = face_detect(raw_frame, pnet, rnet, onet)
            if len(face_boxes):
                face_embeds = face_recognize(face, sess, images_placeholder, phase_train_placeholder, embeddings)
                face_result, face_dist = match_feature_and_embeddings(face_embeds, face_embeds_gt, face_labels_gt, 0.7)

                # match facenet faces and openpose faces
                for info in self.current_frame_info:
                    if info[6] == 1 and info[82] == 1:
                        for which_face, box in enumerate(face_boxes):
                            face_x = info[face_keypoints_x_idx]
                            face_y = info[face_keypoints_y_idx]
                            if np.all(face_x > box[0]) and np.all(face_x < box[2]) and np.all(
                                    face_y > box[1]) and np.all(face_y < box[3]):
                                info[94] = nameToId_dict[face_result[which_face]]
                                info[95] = face_dist[which_face]
                                info[85:89] = box
                                info[100] = 1

    def reid_recognition(self, raw_frame, reid_model, reid_embeds_gt, reid_labels_gt):
        person_to_reid = []
        do_reid = False
        if np.any(np.logical_and(self.current_frame_info[:, 6] == 1, self.current_frame_info[:, 83] == 1)):
            for info in self.current_frame_info:
                # if haven't did face recognition or get unknown result, do reid if possible
                if info[94] <= 0 and info[83] == 1:
                    do_reid = True
                    person_img = raw_frame[int(max(info[3] - 20, 0)): int(info[5] + 10), int(info[2]): int(info[4])]
                    person_to_reid.append(person_img)
            if do_reid:
                person_embeds = query_imgs(person_to_reid, reid_model)
                reid_result, reid_dist = match_feature_and_embeddings(person_embeds, reid_embeds_gt, reid_labels_gt)
                reid_result_idx = 0
                for info in self.current_frame_info:
                    if info[94] <= 0 and info[83] == 1:
                        info[96] = nameToId_dict[reid_result[reid_result_idx]]
                        info[97] = reid_dist[reid_result_idx]
                        info[100] = 1
                        reid_result_idx += 1

    def gait_ability_check(self, idx, steps_thresh):
        idx_idx = np.where(self.frame_info_record[:, 2] == idx)
        keypoints = self.frame_info_record[idx_idx, 7:82]
        if len(keypoints[0]) < steps_thresh:
            return 0
        else:
            if np.all(keypoints[-steps_thresh:, 12] > 0):
                return 1
            else:
                return 0

    def gait_recognition(self, gt_embeddings, gt_labels, sess, embeds, steps_thresh=60):
        for id_record in self.current_frame_info:
            id_record[84] = self.gait_ability_check(id_record[2], steps_thresh)
            if id_record[84] and id_record[6] == 1:
                feed1 = {"Placeholder:0": [id_record['keypoints']], "Placeholder_2:0": [len(id_record['keypoints'])]}
                embeddings = sess.run(embeds, feed_dict=feed1)
                dist = [sum([xx ** 2 for xx in [gt_embeddings[i] - embeddings]][0][0]) for i in
                        range(len(gt_embeddings))]
                min_dist = np.sort(dist)[0]
                if min_dist < 10:
                    result = gt_labels[np.argsort(dist)[0]]
                else:
                    result = 'unknown'
                id_record[98] = idToName_dict[result]
                id_record[99] = min(dist)
                id_record[100] = 1

    def sortout_record(self):
        # Parse result
        # Normalize keypoints
        for info in self.current_frame_info:
            # Normalize keypoints
            if np.any(info[7:82] > 0):
                info[7:82] = trans_coord(info[7:82])

            # Parse result
            ii = np.where(info[[94, 96, 98]] > 0)[0]
            if len(ii) > 0:
                info[101] = info[[94, 96, 98]][ii[0]]
                info[102] = ii[0] + 1
            else:
                # if len(np.where(info[[94, 96, 98]] == 0)[0]):
                # If processed but get unknown, final result should be unknown
                if info[100] == 1:
                    info[101] = 0
                else:  # Not processed this frame
                    past_idx = np.where(self.frame_info_record[:, 1] == info[1])[0]
                    if len(past_idx):
                        info[101] = self.frame_info_record[past_idx, 101][-1]
                        info[102] = self.frame_info_record[past_idx, 102][-1]
                    else:
                        info[101] = 0

    def draw_result(self, raw_image):
        image = raw_image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.current_frame_info[0, 0] > 0:
            for info in self.current_frame_info:
                yolo_coord = [int(c) for c in info[2:6]]
                result = idToName_dict[info[101]]
                if info[101] > 0:
                    color = (0, 255, 0)  # Green
                    if info[102] > 0:
                        method = idToMethod[info[102]]
                        result = method + ': ' + result
                else:
                    color = (255, 0, 0)  # Blue
                image = cv2.rectangle(image, (yolo_coord[0], yolo_coord[1]), (yolo_coord[2], yolo_coord[3]), color, 2)
                image = cv2.putText(image, result, (yolo_coord[0], yolo_coord[1] + 20), font, 0.5, (0, 0, 0))
                if np.any(info[85:89] > 0):
                    face_box = [int(c) for c in info[85:89]]
                    image = cv2.rectangle(image, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 255, 0), 2)

                if info[100] == 1:
                    image = cv2.putText(image, 'Processed', (yolo_coord[0], yolo_coord[1] + 35), font, 0.5, (0, 255, 0))
                else:
                    image = cv2.putText(image, 'Unprocessed', (yolo_coord[0], yolo_coord[1] + 35), font, 0.5,
                                        (255, 0, 0))
        return image

    def at_end_of_frame(self):
        # Append current frame info to history info
        # Clear current info
        self.frame_info_record = np.vstack((self.frame_info_record, self.current_frame_info))
        if self.frame_count == 1:
            self.frame_info_record = self.frame_info_record[1:]
        self.frame_count += 1
        self.current_frame_info = []

    def save_reid_image(self, raw_frame, save_path='./gt_Img/reid'):
        for info in self.current_frame_info:
            if info[83] == 1:
                save_im = raw_frame[int(info[3]) - 20:int(info[5]) + 10, int(info[2]):int(info[4])]
                im_name = str(int(info[0])) + '_' + str(int(info[1])) + '.jpg'
                save_im_path = os.path.join(save_path, im_name)
                cv2.imwrite(save_im_path, save_im)
