from __future__ import print_function
import os
import time
import json
import numpy as np
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


def load_embeddings(embeddings_path):
    embeddings = []
    labels = []
    gt_embeddings_list = os.listdir(embeddings_path)
    for gt_embed in gt_embeddings_list:
        if 'gait' in embeddings_path:
            label = gt_embed.split('-')[0]
        else:
            label = gt_embed.split('_')[0]
        gt_embed_path = os.path.join(embeddings_path, gt_embed)
        j = json.load(open(gt_embed_path, 'r'))
        embeddings.append(j['embeddings'])
        labels.append(label)
    return np.asarray(embeddings), np.asarray(labels)


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


def parse_res(res, scale=1):
    out_res = []
    for rr in res:
        x0, y0 = int(rr['x'] * scale), int(rr['y'] * scale)
        x1, y1 = int(x0 + rr['w'] * scale), int(y0 + rr['h'] * scale)
        new_rr = ['person', rr['id'], (x0, y0, x1, y1), False, -1]  # (type, id, (position), process, idx_in_record)
        out_res.append(new_rr)
    return out_res


def match_res_and_record(res, record):
    res_process = []
    frame_process = True
    for rr in res:
        occured = -1
        for idx, his in enumerate(record):
            if his['id'] == rr[1]:
                occured = idx
                break
        # Haven't seen before, process anyway
        if occured == -1:
            rr[3] = True
            record.append({"id": rr[1], 'recv_times': 1, 'process_method': [], 'embeds_dist': [], 'result': [],
                           'latest_process': 1, 'latest_result': 'unknown', 'latest_method': '',
                           'latest_dist': 999, 'keypoints': []})
            rr[4] = len(record) - 1
        else:
            record[occured]['recv_times'] += 1
            rr[4] = occured
            # Processed but its a long time ago:
            if record[occured]['recv_times'] - record[occured]['latest_process'] > 20 or record[occured][
                'latest_result'] == 'unknown':
                rr[3] = True
                record[occured]['latest_process'] = record[occured]['recv_times']
            else:
                if record[occured]['latest_method'] == 'gait' and record[occured]['latest_dist'] > 0.7:
                    rr[3] = True
                    record[occured]['latest_process'] = record[occured]['recv_times']
                elif record[occured]['latest_method'] == 'reid' and record[occured]['latest_dist'] > 0.7:
                    rr[3] = True
                    record[occured]['latest_process'] = record[occured]['recv_times']
                elif record[occured]['latest_method'] == '':
                    rr[3] = True
                    record[occured]['latest_process'] = record[occured]['recv_times']
                else:
                    res_process.append(False)
                    record[occured]['process_method'].append('None')
    if len(res_process) and not np.any(res_process):
        frame_process = False
    return res, record, frame_process


def match_yolo_openpose(yolo_res, openpose_keypoints, record):
    body_keypoints1 = [1, 2, 5, 8, 9, 10, 12, 13]
    # body_keypoints1 = [1, 2, 5, 8, 9, 12]

    info = []
    for yolo in yolo_res:
        matched = False
        for kk in openpose_keypoints:
            main_body = kk[body_keypoints1, :][kk[body_keypoints1, 2] > 0]
            main_body_x = main_body[:, 0]
            main_body_y = main_body[:, 1]
            if not len(main_body_x) or not len(main_body_y):
                continue
            main_body_left, main_body_right = min(main_body[:, 0]), max(main_body[:, 0])
            main_body_top, main_body_bottom = min(main_body[:, 1]), max(main_body[:, 1])
            hmin, hmax = yolo[2][0] - 20, yolo[2][2] + 20
            vmin, vmax = yolo[2][1] - 20, yolo[2][3] + 20
            # Are keypoints in the bbox of yolo
            if hmin < main_body_left < main_body_right < hmax and vmin < main_body_top < main_body_bottom < vmax:
                id_info = {'id': yolo[1], 'yolo_pos': yolo[2], 'keypoints': kk, 'face_able': False, 'reid_able': False,
                           'face': [], 'body': [], 'face_name': '', 'face_dist': 999, 'reid_name': '', 'reid_dist': 999,
                           'record_idx': yolo[4]}
                if np.all(kk[body_keypoints1, 2] > 0):
                    kk = np.ravel(kk)
                    if sum(kk[3:6]) and sum(kk[24:27]):
                        kk = kk.copy().tolist()
                        kk = trans_coord(kk)
                        record[yolo[4]]['keypoints'].append(kk)
                else:
                    record[yolo[4]]['keypoints'] = []
                matched = True
                break
        if not matched:
            id_info = {'id': yolo[1], 'yolo_pos': yolo[2], 'keypoints': [], 'face_able': False, 'reid_able': False,
                       'face': [], 'body': [], 'face_name': '', 'face_dist': 999, 'reid_name': '', 'reid_dist': 999,
                       'record_idx': yolo[4]}
        info.append(id_info)

    return info, record


def face_detection_module(frame_info, raw_frame, pnet, rnet, onet, sess, images_placeholder, phase_train_placeholder,
                          embeddings, face_embeds_gt, face_labels_gt):
    do_face_detection = False
    face_keypoints = [0, 15, 16]
    face_keypoints1 = [0, 15, 16, 17]
    face_keypoints2 = [0, 15, 16, 18]
    body_keypoints1 = [0, 1, 8, 9, 10, 11]
    body_keypoints2 = [0, 1, 8, 12, 13, 14]
    body_keypoints3 = [1, 8, 12, 13, 14, 17, 18]
    for info in frame_info:
        kk = info['keypoints']
        # Face recognition able
        if len(kk):
            if np.all(kk[face_keypoints1, 2] > 0.5) or np.all(kk[face_keypoints2, 2] > 0.5):
                info['face_able'] = True
                do_face_detection = True
            if np.all(kk[body_keypoints1, 2] > 0.1) or np.all(kk[body_keypoints2, 2] > 0.1) or np.all(
                    kk[body_keypoints3, 2] > 0.1):
                info['reid_able'] = True

    if do_face_detection:
        # Process face detection
        face, face_boxes = face_detect(raw_frame, pnet, rnet, onet)
        if len(face_boxes):
            face_embeds = face_recognize(face, sess, images_placeholder, phase_train_placeholder, embeddings)
            face_result, face_dist = match_feature_and_embeddings(face_embeds, face_embeds_gt, face_labels_gt)

            for info in frame_info:
                if info['face_able']:
                    for which_face, box in enumerate(face_boxes):
                        face_x = info['keypoints'][face_keypoints, 0]
                        face_y = info['keypoints'][face_keypoints, 1]
                        if np.all(face_x > box[0]) and np.all(face_x < box[2]) and np.all(
                                face_y > box[1]) and np.all(face_y < box[3]):
                            info['face_name'] = face_result[which_face]
                            info['face_dist'] = face_dist[which_face]
                            info['face'] = box
    return frame_info


def reid_module(frame_info, raw_frame, reid_model, reid_embeds_gt, reid_labels_gt):
    person_to_reid = []
    do_reid = False
    for info in frame_info:
        if (not len(info['face_name']) or info['face_name'] == 'unknown') and info['reid_able']:
            do_reid = True
            person_img = raw_frame[max(info['yolo_pos'][1] - 20, 0):info['yolo_pos'][3] + 10,
                         info['yolo_pos'][0]: info['yolo_pos'][2]]
            person_to_reid.append(person_img)
    if do_reid:
        person_embeds = query_imgs(person_to_reid, reid_model)
        reid_result, reid_dist = match_feature_and_embeddings(person_embeds, reid_embeds_gt, reid_labels_gt)
        reid_result_idx = 0
        for info in frame_info:
            if (not len(info['face_name']) or info['face_name'] == 'unknown') and info['reid_able']:
                try:
                    info['reid_name'] = reid_result[reid_result_idx]
                except IndexError:
                    print('aaa')
                info['reid_dist'] = reid_dist[reid_result_idx]
                reid_result_idx += 1
    return frame_info


def gait_module(record, gt_embeddings, gt_labels, sess, embeds, steps_thresh=60):
    for id_record in record:
        if len(id_record['keypoints']) >= steps_thresh and np.all(np.asarray([len(kk) for kk in id_record['keypoints']]) > 0):
            feed1 = {"Placeholder:0": [id_record['keypoints']], "Placeholder_2:0": [len(id_record['keypoints'])]}
            embeddings = sess.run(embeds, feed_dict=feed1)
            dist = [sum([xx ** 2 for xx in [gt_embeddings[i] - embeddings]][0][0]) for i in
                    range(len(gt_embeddings))]
            min_dist = np.sort(dist)[0]
            if min_dist < 10:
                result = gt_labels[np.argsort(dist)[0]]
            else:
                result = 'unknown'
            id_record['latest_result'] = result
            id_record['latest_method'] = 'gait'
            id_record['latest_dist'] = min(dist)
            id_record['latest_process'] = id_record['recv_times']
            id_record['keypoints'] = []


def update_record(record, frame_info):
    '''
    :param record:
    :param frame_info:
    :return:
    '''
    for info in frame_info:
        if not info['face_name'] == 'unknown' and len(info['face_name']):
            record[info['record_idx']]['latest_process_method'] = 'face'
            record[info['record_idx']]['process_method'].append('face')
            record[info['record_idx']]['latest_result'] = info['face_name']
            record[info['record_idx']]['latest_dist'] = info['face_dist']
        elif not info['reid_name'] == 'unknown' and len(info['reid_name']):
            record[info['record_idx']]['latest_process_method'] = 'reid'
            record[info['record_idx']]['latest_result'] = info['reid_name']
            record[info['record_idx']]['process_method'].append('reid')
            record[info['record_idx']]['latest_dist'] = info['reid_dist']
        # elif not info['gait_name'] == 'unknown' and len(info['gait_name']):
        #     record[info['record_idx']]['latest_process_method'] = 'gait'
        #     record[info['record_idx']]['latest_result'] = info['gait_name']
        #     record[info['record_idx']]['process_method'].append('gait')
        #     record[info['record_idx']]['latest_dist'] = info['gait_dist']
        else:
            if len(info['face_name']):
                record[info['record_idx']]['process_method'].append('face')
                if record[info['record_idx']]['latest_result'] == 'unknown':
                    record[info['record_idx']]['latest_process_method'] = 'face'
                    record[info['record_idx']]['latest_dist'] = info['face_dist']

            elif len(info['reid_name']):
                record[info['record_idx']]['process_method'].append('reid')
                if record[info['record_idx']]['latest_result'] == 'unknown':
                    record[info['record_idx']]['latest_process_method'] = 'reid'
                    record[info['record_idx']]['latest_dist'] = info['reid_dist']
            # elif info['gait_able']:
            #     record[info['record_idx']]['latest_process_method'] = 'gait'
            #     record[info['record_idx']]['process_method'].append('gait')
            #     record[info['record_idx']]['latest_dist'] = info['gait_dist']
            else:
                record[info['record_idx']]['process_method'].append('')
    return record


def frame_info_to_json(info, record, start_time):
    json_out = []
    processing_time = round((time.time() - start_time) * 1000, 3)
    for id_info in info:
        id_json = {'id': id_info['id'], 'name': record[id_info['record_idx']]['latest_result'],
                   'processing_time': processing_time}
        # id_json['name'] = 'unknown'
        # if id_info['face_able'] and (not id_info['face_name'] == 'unknown'):
        #     id_json['name'] = id_info['face_name']
        # if id_info['reid_able'] and (not id_info['reid_name'] == 'unknown'):
        #     id_json['name'] = id_info['reid_name']
        # if (not id_info['face_able']) and (not id_info['reid_able']):
        #     id_json['name'] = ''
        json_out.append(id_json)
    return json_out


def draw_result(info, img):
    for person in info:
        # Draw yolo box:
        yolo = person['yolo_pos']
        if len(person['face_name']):
            box = person['face']
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (84, 168, 0), 1)
            out_text = person['face_name'] + ': ' + str(round(person['face_dist'], 3))
            img = cv2.putText(img, out_text, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        elif len(person['reid_name']):
            out_text = person['reid_name'] + ': ' + str(round(person['reid_dist'], 3))
            img = cv2.putText(img, out_text, (yolo[0], yolo[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        if len(person['face_name']) and not person['face_name'] == 'unknown':
            img = cv2.rectangle(img, (yolo[0], yolo[1]), (yolo[2], yolo[3]), (0, 255, 0), 2)
        elif len(person['reid_name']) and not person['reid_name'] == 'unknown':
            img = cv2.rectangle(img, (yolo[0], yolo[1]), (yolo[2], yolo[3]), (0, 255, 0), 2)
        elif person['face_able'] or person['reid_able']:
            img = cv2.rectangle(img, (yolo[0], yolo[1]), (yolo[2], yolo[3]), (255, 0, 0), 2)
        else:
            img = cv2.rectangle(img, (yolo[0], yolo[1]), (yolo[2], yolo[3]), (0, 0, 255), 2)


def iou(box1, box2):
    bbox1 = [float(x) for x in box1]
    bbox2 = [float(x) for x in box2]

    (left_1, top_1, right_1, bottom_1) = bbox1
    (left_2, top_2, right_2, bottom_2) = bbox2

    left = max(left_1, left_2)
    top = max(top_1, top_2)
    right = min(right_1, right_2)
    bottom = min(bottom_1, bottom_2)

    # check if there is an overlap
    if right - left <= 0 or bottom - top <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (right_1 - left_1) * (bottom_1 - top_1)
    size_2 = (right_2 - left_2) * (bottom_2 - top_2)
    size_intersection = (right - left) * (bottom - top)
    size_union = size_1 + size_2 - size_intersection
    return size_intersection / size_union


def match_yolo_openpose_iou(yolo_res, openpose_keypoints):
    body_keypoints1 = [1, 8, 9, 10]
    body_keypoints2 = [1, 8, 12, 13]
    info = []
    if not len(yolo_res) and not len(openpose_keypoints):
        return {}
    distance_mat = np.empty((len(yolo_res), len(openpose_keypoints)))
    for i, yolo in enumerate(yolo_res):
        for j, kk in enumerate(openpose_keypoints):
            x_axis = kk[:, 0][kk[:, 0] > 0]
            y_axis = kk[:, 1][kk[:, 1] > 0]
            op_pos = [min(x_axis), min(y_axis), max(x_axis), max(y_axis)]
            distance_mat[i, j] = iou(yolo[2], op_pos)
    for i, yolo in enumerate(yolo_res):
        min_idx = np.argmax(distance_mat)
        r = int(min_idx / distance_mat.shape[1])
        c = min_idx % distance_mat.shape[1]

        id_info = {'id': yolo[1], 'yolo_pos': yolo[2], 'keypoints': kk, 'face_able': False, 'reid_able': False,
                   'face': [], 'body': [], 'name': ''}

    return distance_mat


def detect_face_or_not(keypoints, face_boxes):
    face_keypoints = [0, 15, 16, 17, 18]
    for box in face_boxes:
        for person_keypoint in keypoints:
            face_x = person_keypoint[face_keypoints, 0]
            face_y = person_keypoint[face_keypoints, 1]
            face_p = person_keypoint[face_keypoints, 2]
            if np.all(face_x > box[0]) and np.all(face_x < box[2]) and np.all(face_y > box[1]) and np.all(
                    face_y < box[3]):
                if np.all(face_p > 0.3):
                    return True
    return False


def find_face_from_openpose(openpose_keypoints):
    face_boxes = []
    face_points_id = [0, 1, 15, 16, 17, 18]
    for person in openpose_keypoints:
        if np.all(person[face_points_id, 2] > 0):
            margin = np.max(
                np.sqrt(np.sum(np.square(person[face_points_id[2:4], :-1] - person[face_points_id[-2:], :-1]), axis=1)))
            # half = np.max(np.sqrt(np.sum(np.square(person[0][:-1] - person[face_points_id[1:], :-1]), axis=1)))
            bounding = np.zeros(4, dtype=np.int32)
            # bounding[0] = np.maximum((person[17][0] - half), 0)
            # bounding[1] = np.maximum((np.max(person[face_points_id, 1]) - half), 0)
            # bounding[2] = person[0][0] + half
            # bounding[3] = person[0][1] + half
            d01 = np.max(np.sqrt(np.sum(np.square(person[0, :-1] - person[1, :-1]))))
            bounding[0] = np.maximum((person[17][0] - margin), 0)
            bounding[1] = np.maximum((np.min(person[face_points_id, 1]) - d01), 0)
            bounding[2] = person[18][0] + margin
            bounding[3] = person[1][1]
            face_boxes.append(bounding)
        else:
            continue
    return face_boxes


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 100  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = FaceRecognition.detect_face.create_mtcnn(sess, None)

    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = FaceRecognition.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


def main(args):
    images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)

            nrof_images = len(args.image_files)

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, args.image_files[i]))
            print('')

            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                    print('  %1.4f  ' % dist, end='')
                print('')
