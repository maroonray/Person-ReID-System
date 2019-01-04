import numpy as np
import cv2
import tensorflow as tf
import FaceRecognition.detect_face
import FaceRecognition.facenet as facenet


def load_mtcnn(mtcnn_path):
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess1 = tf.Session(config=config)
        with sess1.as_default():
            pnet, rnet, onet = FaceRecognition.detect_face.create_mtcnn(sess1, mtcnn_path)
    return pnet, rnet, onet


def load_facenet(facenet_path):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    facenet.load_model(facenet_path)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    return sess, images_placeholder, embeddings, phase_train_placeholder


def face_detect(input_img, pnet, rnet, onet,minsize=60, threshold=[0.6, 0.7, 0.7], factor=0.707):
    frame_to_face = input_img
    img_size = frame_to_face.shape
    frame_to_face = frame_to_face[..., ::-1]
    bounding_boxes, _ = FaceRecognition.detect_face.detect_face(frame_to_face, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) > 0:
        faces = []
        faces_boxes = []
        faces_confidence = []
        for i in range(len(bounding_boxes)):
            det = np.squeeze(bounding_boxes[i, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            margin_x = (det[2] - det[0]) / 4
            margin_y = (det[3] - det[1]) / 4
            assert margin_x > 0 and margin_y > 0
            bb[0] = np.maximum(det[0] - margin_x, 0)
            bb[1] = np.maximum(det[1] - margin_y, 0)
            bb[2] = np.minimum(det[2] + margin_x, img_size[1])
            bb[3] = np.minimum(det[3] + margin_y / 2, img_size[0])
            # output_image = cv2.rectangle(output_image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 3)
            # bounding_boxes = find_face_from_openpose(keypoints)

            # for bb in bounding_boxes:
            face_img = frame_to_face[bb[1]:bb[3], bb[0]:bb[2], :]
            face_img = cv2.resize(face_img, (160, 160))
            face_img = facenet.prewhiten(face_img)
            faces.append(face_img)
            faces_boxes.append(bb)
    else:
        faces = []
        faces_boxes = []
    return faces, faces_boxes


def face_recognize(input_faces, sess, images_placeholder, phase_train_placeholder, embeddings):
    feed_dict = {images_placeholder: input_faces,
                 phase_train_placeholder: False}
    emb = sess.run(embeddings, feed_dict=feed_dict)
    return emb


def face_detect_and_recognize(input_img, sess, pnet, rnet, onet, images_placeholder, phase_train_placeholder,
                              embeddings,
                              minsize, threshold, factor):
    frame_to_face = input_img
    img_size = frame_to_face.shape
    frame_to_face = frame_to_face[..., ::-1]
    bounding_boxes, _ = FaceRecognition.detect_face.detect_face(frame_to_face, minsize, pnet, rnet, onet,
                                                                threshold,
                                                                factor)
    if len(bounding_boxes) > 0:
        faces = []
        faces_boxes = []
        faces_confidence = []
        for i in range(len(bounding_boxes)):
            det = np.squeeze(bounding_boxes[i, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            margin_x = (det[2] - det[0]) / 4
            margin_y = (det[3] - det[1]) / 4
            assert margin_x > 0 and margin_y > 0
            bb[0] = np.maximum(det[0] - margin_x, 0)
            bb[1] = np.maximum(det[1] - margin_y, 0)
            bb[2] = np.minimum(det[2] + margin_x, img_size[1])
            bb[3] = np.minimum(det[3] + margin_y, img_size[0])
            # output_image = cv2.rectangle(output_image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 3)
            # bounding_boxes = find_face_from_openpose(keypoints)

            # for bb in bounding_boxes:
            face_img = frame_to_face[bb[1]:bb[3], bb[0]:bb[2], :]
            face_img = cv2.resize(face_img, (160, 160))
            face_img = facenet.prewhiten(face_img)
            faces.append(face_img)
            faces_boxes.append(bb)

        feed_dict = {images_placeholder: faces,
                     phase_train_placeholder: False}
        emb = sess.run(embeddings, feed_dict=feed_dict)
    else:
        faces_boxes = []
        emb = []
    return faces_boxes, emb


def update_face_img(img, boxes):
    ori_size = img.shape
    start_row = 0
    start_column = ori_size[1] - 1
    img = cv2.copyMakeBorder(img, 0, 0, 0, 400, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    for box in boxes:
        height = box[3] - box[1] - 1
        width = box[2] - box[0] - 1
        if start_row + height > ori_size[0]:
            start_row = 0
            start_column += width
            if start_column > ori_size[1]:
                print('too many faces')
                break
        img[start_row:start_row + height, start_column:start_column + width] = img[box[1] + 1:box[3], box[0] + 1:box[2]]
        start_row += height
    return img


def get_face_buffer(img, boxes):
    ori_size = img.shape
    start_row = 0
    start_column = 0
    max_width = 0
    buffer = np.zeros((ori_size[0], 300, 3), dtype=np.uint8)
    for box in boxes:
        face = img[box[1] + 1:box[3], box[0] + 1:box[2], :]
        face = cv2.resize(face, (100, 100))
        height = 100
        width = 100
        max_width = max(max_width, width)
        if start_row + height > ori_size[0]:
            start_row = 0
            start_column += max_width + 1
            if start_column > ori_size[1]:
                print('too many faces')
                break
        buffer[start_row:start_row + height, start_column:start_column + width, :] = face
        start_row += height
    return buffer
