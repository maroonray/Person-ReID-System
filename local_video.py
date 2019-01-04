import time
import cv2
import zmq
import json
import numpy as np
from utils import *
from YOLOv3.yolo3 import load_net, load_meta, yolo3_detect_person
from PeronReid.query import *
from OpenPose.load_model import load_openpose_model
from FaceRecognition.detect_recognize import *
from GaitRecognition.gait_model import *


def process_latest_frame(video_path, openpose_path, mtcnn_path, facenet_path, reid_path, yolo_path, gait_dir, gait_graph):
    frame_count = 1

    # Load OpenPose Model
    openpose = load_openpose_model(openpose_path)
    print('---------OpenPose Loaded----------')
    # Load YOLOv3
    yolo3_meta = load_meta(yolo_path[0])
    yolo3_net = load_net(yolo_path[1], yolo_path[2], 0)
    print('----------YOLOv3 Loaded-----------')
    # Load MTCNN (face detection)
    pnet, rnet, onet = load_mtcnn(mtcnn_path)
    print('-----------MTCNN Loaded-----------')
    # Load Facenet
    sess, images_placeholder, embeddings, phase_train_placeholder = load_facenet(facenet_path)
    print('----------FaceNet Loaded----------')
    # Load Reid model
    reid_model = load_reid_model(reid_path)
    print('------------ReID Loaded-----------')
    # Load Gait model
    sess_gait, embeds_gait = load_gait_model(gait_graph, gait_dir)

    # Load GroundTruth Data
    face_embeds_gt, face_labels_gt = load_embeddings('/home/yul-j/Desktop/Demos/MixedModel/embeddings/faces')
    reid_embeds_gt, reid_labels_gt = load_embeddings('/home/yul-j/Desktop/Demos/MixedModel/embeddings/reid')
    gait_embeds_gt, gait_labels_gt = load_embeddings('/home/yul-j/Desktop/Demos/MixedModel/embeddings/gait')
    # frame_info = {'id': yolo[1], 'yolo_pos': yolo[2], 'keypoints': [], 'face_able': False, 'reid_able': False,
    #               'face': [], 'body': [], 'face_name': '', 'face_dist': [], 'reid_name': '', 'reid_dist': []}
    # id_json = {'id': id_info['id'], 'name': [], 'processing_time': processing_time}
    # res = ('person', id, (x0, y0, x1, y1), False, -1)
    # record = [{"id": -1, 'recv_times': 1, 'process_method': [], 'embeds_dist': [], 'result': [], 'latest_process': 1,
    #            'latest_result': 'unknown', 'latest_method': '', 'latest_dist': 999, 'keypoints': [[]]}]
    record = []
    out_record = []
    gait_recognize = True
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    while ret:
        start = time.time()
        frame = np.rot90(frame, 3)
        frame = cv2.resize(frame, (540, 960))
        raw_frame = frame
        _, res = yolo3_detect_person(yolo3_net, yolo3_meta, raw_frame)

        if len(res):
            res, record, process_frame = match_res_and_record(res, record)
        else:
            continue

        if process_frame:
            # TODO: Get json from openpose server
            # Process OpenPose
            keypoints = openpose.forward(raw_frame, False)
            openpose_frame = raw_frame
            keypoints = sorted(keypoints, key=lambda x: -sum(x[:, 2]))

            # Match yolo and openpose
            frame_info, record = match_yolo_openpose(res, keypoints, record)

            frame_info = face_detection_module(frame_info, raw_frame, pnet, rnet, onet, sess, images_placeholder,
                                               phase_train_placeholder, embeddings, face_embeds_gt, face_labels_gt)

            frame_info = reid_module(frame_info, raw_frame, reid_model, reid_embeds_gt, reid_labels_gt)

            record = update_record(record, frame_info)

            if gait_recognize:
                gait_module(record, gait_embeds_gt, gait_labels_gt, sess_gait, embeds_gait)

            draw_result(frame_info, openpose_frame)
            result = frame_info_to_json(frame_info, record, start)
            out_record.append(result)
            if len(out_record) > 10:
                out_record = out_record[1:]
        else:
            result = out_record[-1]
            openpose_frame = raw_frame
        print('-----------------------------------------------------')
        print(result)
        # print('Frame: {:>7}, {:.4}, person: {}'.format(frame_count, time.time() - start, len(keypoints)))

        cv2.imshow('a', openpose_frame)
        cv2.waitKey(1)
        frame_count += 1
        ret, frame = video.read()


if __name__ == '__main__':
    # rtsp_path = "rtsp://admin:admin123@10.1.83.133/cam/realmonitor?channel=10&subtype=0"
    rtsp_path = '/media/yul-j/ssd_disk/Datasets/Gait/private_data/backup/videos/fixed_angles/130-01-01-216.mp4'

    openpose_model = "/home/yul-j/Desktop/Demos/OpenPose/models/"
    mtcnn_model = '/home/yul-j/Desktop/Demos/FaceRecognition/facenet/src/align'
    facenet_model = '/home/yul-j/Desktop/Models/PlayGround/Face/facenet_model/20180402-114759.pb'
    reid_model = '/home/yul-j/Desktop/Demos/deep-person-reid/models/hacnn_market_xent/hacnn_market_xent.pth.tar'
    yolo_info = [b"YOLOv3/models/coco.data", b"YOLOv3/models/yolov3.cfg", b"YOLOv3/models/yolov3.weights"]
    gait_graph = '/home/yul-j/Desktop/Demos/Gait/main/logs/60k-steps/60000.meta'
    gait_model = '/home/yul-j/Desktop/Demos/Gait/main/logs/60k-steps/'
    process_latest_frame(rtsp_path, openpose_model, mtcnn_model, facenet_model, reid_model, yolo_info, gait_model, gait_graph)
