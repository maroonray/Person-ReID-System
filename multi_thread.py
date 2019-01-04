import sys
import time
import cv2
import zmq
import json
from utils import *
from frame_info import FrameInfo
from YOLOv3.yolo3 import load_net, load_meta, yolo3_detect_person
from PeronReid.query import *
from FaceRecognition.detect_recognize import *
from GaitRecognition.gait_model import *


frame_buffer = []
jsonStringToBehaviorBuffer = []
jsonStringFromIdentity = []


def parse_model_dir(model_path):
    mtcnn_path = os.path.join(model_path, 'facenet')
    facenet_path = os.path.join(model_path, 'facenet/20180402-114759.pb')
    reid_path = os.path.join(model_path, 'reid/hacnn_market_xent.pth.tar')
    yolo_path = [b"models/yolo3/coco.data", b"models/yolo3/yolov3.cfg", b"models/yolo3/yolov3.weights"]
    gait_dir = os.path.join(model_path, 'gait')
    return mtcnn_path, facenet_path, reid_path, yolo_path, gait_dir


def get_latest_frame(path, scale=1):
    global frame_buffer

    video = cv2.VideoCapture(path)
    size = (video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = video.read()
    while ret:
        if not scale == 1:
            frame = cv2.resize(frame, (int(size[0] * scale), int(size[1] * scale)))
        frame_buffer.append(frame)
        if len(frame_buffer) > 10:
            frame_buffer = frame_buffer[1:]
        ret, frame = video.read()


def process_latest_frame(model_dir, openpose_path, embeddings_path):
    global frame_buffer
    global jsonStringToBehaviorBuffer
    global jsonStringFromIdentity

    mtcnn_path, facenet_path, reid_path, yolo_path, gait_dir = parse_model_dir(model_dir)

    # Load OpenPose Model
    openpose_module = os.path.join(openpose_path, '../build/python/openpose/')
    sys.path.append(openpose_module)
    from OpenPose.load_model import load_openpose_model
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
    # Load Gait modek
    sess_gait, embeds_gait = load_gait_model(gait_dir)
    print('---------Gait Model Loaded---------')

    # Load GroundTruth Data
    face_embeds_gt, face_labels_gt = load_embeddings(os.path.join(embeddings_path, 'faces'))
    reid_embeds_gt, reid_labels_gt = load_embeddings(os.path.join(embeddings_path, 'reid'))
    gait_embeds_gt, gait_labels_gt = load_embeddings(os.path.join(embeddings_path, 'gait'))
    print('---------Embeddings Loaded---------')
    record = FrameInfo()
    while True:
        if len(frame_buffer):
            raw_frame = frame_buffer[-1]

            # if not len(jsonStringToBehaviorBuffer):
            #     continue
            # res = jsonStringToBehaviorBuffer[-1]
            # res = parse_res(res)
            _, res = yolo3_detect_person(yolo3_net, yolo3_meta, raw_frame)

            record.update_with_yolo_info(res)

            if len(res):
                # Process OpenPose
                keypoints = openpose.forward(raw_frame, False)
                keypoints = sorted(keypoints, key=lambda x: -sum(x[:, 2]))

                # Match yolo and openpose
                record.update_with_openpose_keypoints(keypoints)

                record.face_recognition(raw_frame, pnet, rnet, onet, sess, images_placeholder,
                                        phase_train_placeholder, embeddings, face_embeds_gt, face_labels_gt)

                record.reid_recognition(raw_frame, reid_model, reid_embeds_gt, reid_labels_gt)

                record.gait_recognition(gait_embeds_gt, gait_labels_gt, sess_gait, embeds_gait)

                record.sortout_record()

            draw_frame = record.draw_result(raw_frame)

            cv2.imshow('a', draw_frame)
            op = cv2.waitKey(1)
            if op == ord('s'):
                record.save_reid_image(raw_frame)
            elif op == ord('q'):
                sys.exit()
            else:
                continue
            record.at_end_of_frame()
            print(time.time() - record.current_frame_start_time, ' ', record.frame_count)


def subscribe_json(url):
    global jsonStringToBehaviorBuffer
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(url)
    socket.setsockopt_string(zmq.SUBSCRIBE, u'')
    while True:
        string = socket.recv()
        j = json.loads(string.decode(encoding='utf-8'))
        jsonStringToBehaviorBuffer.append(j)
        if len(jsonStringToBehaviorBuffer) > 100:
            jsonStringToBehaviorBuffer = jsonStringToBehaviorBuffer[1:]


def send_json():
    global jsonStringFromIdentity
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5000")
    while True:
        time.sleep(0.001)
        if len(jsonStringFromIdentity):
            msg = jsonStringFromIdentity[-1]
            msg = json.dumps(msg)
            socket.send_unicode(msg)


def json_frame_sync_test():
    global frame_buffer
    global jsonStringToBehaviorBuffer
    while True:
        if len(frame_buffer) and len(jsonStringToBehaviorBuffer):
            current_frame = frame_buffer[-1]
            current_json = jsonStringToBehaviorBuffer[-1]
            for person in current_json:
                current_frame = cv2.rectangle(current_frame, (person['x'], person['y']),
                                              (person['x'] + person['w'], person['y'] + person['h']), (84, 168, 0), 2)
            cv2.imshow('a', current_frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    openpose_model = "/home/yul-j/Desktop/Demos/OpenPose/models/"
    mtcnn_model = '/home/yul-j/Desktop/Demos/FaceRecognition/facenet/src/align'
    facenet_model = '/home/yul-j/Desktop/Models/PlayGround/Face/facenet_model/20180402-114759.pb'
    reid_model = '/home/yul-j/Desktop/Demos/deep-person-reid/models/hacnn_market_xent/hacnn_market_xent.pth.tar'
    yolo_info = [b"YOLOv3/models/coco.data", b"YOLOv3/models/yolov3.cfg", b"YOLOv3/models/yolov3.weights"]
    gait_graph = '/home/yul-j/Desktop/Demos/Gait/main/logs/60k-steps/60000.meta'
    gait_model = '/home/yul-j/Desktop/Demos/Gait/main/logs/60k-steps/'
    process_latest_frame(openpose_model, mtcnn_model, facenet_model, reid_model, yolo_info, gait_model, gait_graph)
