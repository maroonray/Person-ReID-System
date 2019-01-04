import os
import sys
import time
import cv2
import zmq
import json
from PeronReid.query import *
from OpenPose.load_model import load_openpose_model
from FaceRecognition.detect_recognize import *


def create_face_embeddings(mtcnn_path, facenet_path, img_src, embeds_dst):
    faces = []
    embeds_name_list = []
    # Load MTCNN (face detection)
    pnet, rnet, onet = load_mtcnn(mtcnn_path)
    print('-----------MTCNN Loaded-----------')
    # Load Facenet
    sess, images_placeholder, embeddings, phase_train_placeholder = load_facenet(facenet_path)
    print('----------FaceNet Loaded----------')
    im_list = os.listdir(img_src)
    for im_file in im_list:
        im_name = im_file.split('.')[0]
        im_path = os.path.join(img_src, im_file)
        img = cv2.imread(im_path)
        face, face_boxes = face_detect(img, pnet, rnet, onet)
        for i, f in enumerate(face):
            faces.append(f)
            embeds_name = im_name + '_' + str(i) + '.json'
            embeds_name_list.append(embeds_name)
    face_embeds = face_recognize(faces, sess, images_placeholder, phase_train_placeholder, embeddings)
    for j, fe in enumerate(face_embeds):
        embeds_path = os.path.join(embeds_dst, embeds_name_list[j])
        out = {'embeddings': fe.tolist()}
        json.dump(out, open(embeds_path, 'w'))


def creat_reid_embeddings(reid_path, img_src, embeds_dst):
    embeds_list = []
    name_list = []
    # Load Reid model
    reid_model = load_reid_model(reid_path)
    if os.path.isdir(img_src):
        im_list = os.listdir(img_src)
        for im_file in im_list:
            im_name = im_file.split('.')[0]
            im_path = os.path.join(img_src, im_file)
            img = cv2.imread(im_path)
            person_embeds = query_imgs([img], reid_model)[0]
            embeds_list.append(person_embeds)
            name = im_name + '.json'
            name_list.append(name)
        for j, fe in enumerate(embeds_list):
            embeds_path = os.path.join(embeds_dst, name_list[j])
            out = {'embeddings': fe.tolist()}
            json.dump(out, open(embeds_path, 'w'))


if __name__ == '__main__':
    mtcnn_model = '/home/yul-j/Desktop/Demos/FaceRecognition/facenet/src/align'
    facenet_model = '/home/yul-j/Desktop/Models/PlayGround/Face/facenet_model/20180402-114759.pb'
    face_dir = '/home/yul-j/Desktop/Demos/MixedModel/gt_Img/faces'
    face_embeds_dir = '/home/yul-j/Desktop/Demos/MixedModel/embeddings/faces'
    reid_model = '/home/yul-j/Desktop/Demos/deep-person-reid/models/hacnn_market_xent/hacnn_market_xent.pth.tar'
    reid_dir = '/home/yul-j/Desktop/Demos/MixedModel/gt_Img/reid'
    reid_embeds_dir = '/home/yul-j/Desktop/Demos/MixedModel/embeddings/reid'

    creat_reid_embeddings(reid_model, reid_dir, reid_embeds_dir)
    create_face_embeddings(mtcnn_model, facenet_model, face_dir, face_embeds_dir)