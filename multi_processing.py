import time
import cv2
import tensorflow as tf
import FaceRecognition.detect_face
import FaceRecognition.facenet as facenet
from PeronReid.query import *
from OpenPose.load_model import load_openpose_model


def get_latest_frame(path, queue=None):
    video = cv2.VideoCapture(path)
    size = (video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = video.read()
    while ret:
        queue.put(frame)
        ret, frame = video.read()


def process_latest_frame(openpose_path, mtcnn_path, facenet_path, reid_path, queue=None):
    frame_count = 1
    # Load OpenPose Model
    openpose = load_openpose_model(openpose_path)
    print('----------OpenPose Loaded----------')

    # Load MTCNN (face detection)
    minsize = 50  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709
    margin = 44
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess1 = tf.Session(config=config)
        with sess1.as_default():
            pnet, rnet, onet = FaceRecognition.detect_face.create_mtcnn(sess1, mtcnn_path)
    print('----------MTCNN Loaded----------')

    # Load Facenet
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    facenet.load_model(facenet_path)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    print('----------FaceNet Loaded----------')

    # Load Reid model
    reid_model = load_reid_model(reid_path)
    print('----------ReID Loaded----------')

    while True:
        if not queue.empty():
            frame_count += 1

            start = time.time()
            frame_to_process = queue.get()
            img_size = np.asarray(frame_to_process.shape)[0:2]
            keypoints, output_image = openpose.forward(frame_to_process, True)

            if frame_count % 200 == 0:
                print('--------------Do face recognition-------------')
                frame_to_face = frame_to_process.copy()
                frame_to_face = frame_to_face[..., ::-1]
                bounding_boxes, _ = FaceRecognition.detect_face.detect_face(frame_to_face, minsize, pnet, rnet, onet,
                                                                            threshold,
                                                                            factor)
                if len(bounding_boxes) > 0:
                    det = np.squeeze(bounding_boxes[0, 0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    # output_image = cv2.rectangle(output_image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 3)
                    # bounding_boxes = find_face_from_openpose(keypoints)

                    # for bb in bounding_boxes:
                    face_img = frame_to_face[bb[1]:bb[3], bb[0]:bb[2], :]
                    face_img = cv2.resize(face_img, (160, 160))
                    face_img = facenet.prewhiten(face_img)

                    feed_dict = {images_placeholder: [face_img],
                                 phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    output_image = cv2.rectangle(output_image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 3)
            print('Frame: {:>7}, {:4}'.format(frame_count, time.time() - start))
            cv2.imshow('a', output_image)
            cv2.waitKey(1)