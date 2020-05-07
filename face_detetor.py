#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/5/7
"""
import os

import cv2
import numpy as np
import dlib

from root_dir import IMGS_DIR, MODELS_DIR
from utils.mat_utils import get_box_size
from utils.video_utils import show_img_bgr, draw_box, draw_points


class FaceDetector(object):
    def __init__(self):
        model_path = os.path.join(MODELS_DIR, "3rdparty")
        self.mtcnn_pro = self.get_mtcnn_models()

        self.opencv_detector = self.get_opencv_model(model_path)
        self.dlib_detector = self.get_dlib_model(model_path)

        self.lms_detector = self.get_lms_model(model_path)

    def get_mtcnn_models(self):
        """
        获取MTCNN模型
        """
        from mtcnn.detector import PNet, RNet, ONet
        pnet, rnet, onet = PNet(), RNet(), ONet()
        pnet.eval()
        rnet.eval()
        onet.eval()
        return [pnet, rnet, onet]

    def get_faces_mtcnn(self, img_bgr):
        """
        基于MTCNN检测人脸
        """
        from mtcnn.detector import detect_faces
        bbox_mtcnn, landmarks_mtcnn = detect_faces(img_bgr, self.mtcnn_pro, min_face_size=50)

        bbox_list, lms_list, size_list = [], [], []

        for bbox, lms in zip(bbox_mtcnn, landmarks_mtcnn):
            box_prob = bbox[4]  # 置信度
            if box_prob < 0.9:  # 小于0.9直接pass
                continue

            box = bbox[0:4].astype(np.int32)
            bbox_list.append(box)

            lms_tmp = lms.astype(np.int32)
            lms_points = []
            for x, y in zip(lms_tmp[0:5], lms_tmp[5:10]):
                lms_points.append([x, y])
            lms_list.append(lms_points)

            # draw_box(img_bgr, box, is_new=False)
            # draw_points(img_bgr, lms_points, is_new=False)

        # from root_dir import IMGS_DIR
        # img_path = os.path.join(IMGS_DIR, "mtcnn_res.jpg")
        # cv2.imwrite(img_path, img_bgr)

        return bbox_list, lms_list

    def get_main_face_mtcnn(self, img_bgr):
        """
        基于MTCNN检测主要人脸
        """
        bbox_list, lms_list = self.get_faces_mtcnn(img_bgr=img_bgr)

        size_list = []
        for bbox in bbox_list:
            box_size = get_box_size(bbox)
            size_list.append(box_size)

        max_idx = size_list.index(max(size_list))  # 人脸最大
        main_box = bbox_list[max_idx]
        main_landmarks = lms_list[max_idx]

        return main_box, main_landmarks

    def get_opencv_model(self, model_path):
        """
        基于OpenCV的人脸检测模型
        """
        xml_path = os.path.join(model_path, 'lbpcascade_frontalface_improved.xml')
        opencv_detector = cv2.CascadeClassifier(xml_path)
        return opencv_detector

    def get_dlib_model(self, model_path):
        """
        基于Dlib的人脸检测模型
        """
        dat_path = os.path.join(model_path, 'mmod_human_face_detector.dat')
        dlib_detector = dlib.cnn_face_detection_model_v1(dat_path)
        return dlib_detector

    def get_lms_model(self, model_path):
        """
        基于dlib的人脸关键点模型，两个眼角和一个鼻子
        """
        dat_path = os.path.join(model_path, 'shape_predictor_5_face_landmarks.dat')
        landmarks_predictor = dlib.shape_predictor(dat_path)
        return landmarks_predictor

    def get_faces_opencv(self, img_gray):
        """
        基于OpenCV的人脸检测
        """
        detections = self.opencv_detector.detectMultiScale(img_gray)

        box_list = []  # 列表
        for d in detections:
            l, t, w, h = d
            r, b = l + w, t + h
            box = [l, t, r, b]
            box_list.append(box)

        #     draw_box(img_bgr, box, is_new=False)
        #
        # from root_dir import IMGS_DIR
        # img_path = os.path.join(IMGS_DIR, "opencv_res.jpg")
        # cv2.imwrite(img_path, img_bgr)

        return box_list

    def get_faces_dlib(self, img_gray):
        """
        基于Dlib的人脸检测
        """
        scale = 2

        detections = self.dlib_detector(cv2.resize(img_gray, (0, 0), fx=1 / scale, fy=1 / scale), 0)

        box_list = []
        for d in detections:
            box = [d.rect.left() * scale, d.rect.top() * scale, d.rect.right() * scale, d.rect.bottom() * scale]
            box_list.append(box)

        #     draw_box(img_bgr, box, is_new=False)
        #
        # from root_dir import IMGS_DIR
        # img_path = os.path.join(IMGS_DIR, "dlib_res.jpg")
        # cv2.imwrite(img_path, img_bgr)

        return box_list

    def detect_landmarks(self, img_gray, bbox):
        """Detect 5-point facial landmarks for faces in frame."""
        l, t, r, b = bbox

        rectangle = dlib.rectangle(left=int(l), top=int(t), right=int(r), bottom=int(b))
        landmarks_dlib = self.lms_detector(img_gray, rectangle)

        def tuple_from_dlib_shape(index):
            p = landmarks_dlib.part(index)
            return p.x, p.y

        num_landmarks = landmarks_dlib.num_parts
        landmarks = np.array([tuple_from_dlib_shape(i) for i in range(num_landmarks)])
        landmarks = list(landmarks)
        return landmarks

    def get_faces_dwo(self, img_bgr):
        """
        检测人脸关键点
        """
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 优先使用dlib，其次使用opencv
        box_list = self.get_faces_dlib(img_gray)
        if not box_list:
            box_list = self.get_faces_opencv(img_gray)

        lms_list = []
        for bbox in box_list:
            lms = self.detect_landmarks(img_gray, bbox)  # 检测脸部关键点
            lms_list.append(lms)

            # draw_box(img_bgr, bbox, is_new=False)
            # draw_points(img_bgr, lms, is_new=False)

        # from root_dir import IMGS_DIR
        # img_path = os.path.join(IMGS_DIR, "dwo_lms.jpg")
        # cv2.imwrite(img_path, img_bgr)

        box_list = list(box_list)
        return box_list, lms_list

    def get_main_faces_dwo(self, img_bgr):
        """
        基于dlib和OpenCV检测人脸关键点
        """
        bbox_list, lms_list = self.get_faces_dwo(img_bgr=img_bgr)

        size_list = []
        for bbox in bbox_list:
            box_size = get_box_size(bbox)
            size_list.append(box_size)

        max_idx = size_list.index(max(size_list))  # 人脸最大
        main_box = bbox_list[max_idx]
        main_landmarks = lms_list[max_idx]

        return main_box, main_landmarks


def face_detector_test():
    img_path = os.path.join(IMGS_DIR, 'gg_large.jpg')
    # img_path = os.path.join(IMGS_DIR, 'girl.jpg')
    img_bgr = cv2.imread(img_path)
    show_img_bgr(img_bgr)
    print('[Info] img_bgr: {}'.format(img_bgr.shape))

    fd = FaceDetector()
    # fd.get_faces_mtcnn(img_bgr)
    # main_box, main_landmarks = fd.get_main_face_mtcnn(img_bgr)
    # draw_box(img_bgr, main_box, is_new=False)
    # draw_points(img_bgr, main_landmarks, is_new=False)

    # fd.get_faces_opencv(img_bgr)
    # fd.get_faces_dlib(img_bgr)
    # fd.get_faces_dwo(img_bgr)
    fd.get_main_faces_dwo(img_bgr)


def main():
    face_detector_test()


if __name__ == '__main__':
    main()
