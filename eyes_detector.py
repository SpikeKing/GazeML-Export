#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/5/7
"""
import os
import cv2
import numpy as np

from face_detector import FaceDetector
from root_dir import IMGS_DIR
from utils.video_utils import show_img_bgr, show_img_gray


class EyesDetector(object):
    def __init__(self):
        self.fd = FaceDetector()

    def crop_eyes(self, img_gray, landmarks):
        """From found landmarks in previous steps, segment eye image."""
        eyes_info = []
        landmarks = np.array(landmarks)

        # Final output dimensions
        oh, ow = (108, 180)

        # Segment eyes
        # for corner1, corner2, is_left in [(36, 39, True), (42, 45, False)]:
        for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
            x1, y1 = landmarks[corner1, :]
            x2, y2 = landmarks[corner2, :]

            # 裁剪出1.5倍眼睛宽度
            eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
            if eye_width == 0.0:
                continue

            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

            # Centre image on middle of eye, 定位眼睛中心
            translate_mat = np.asmatrix(np.eye(3))
            translate_mat[:2, 2] = [[-cx], [-cy]]
            inv_translate_mat = np.asmatrix(np.eye(3))
            inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

            # w, h = img_gray.shape[0:2]
            # tmp_image = cv2.warpAffine(img_gray, translate_mat[:2, :], (w, h))
            # show_img_gray(tmp_image, save_name="translate.jpg")

            # Rotate to be upright, 旋转矩阵
            roll = 0.0 if x1 == x2 else np.arctan((y2 - y1) / (x2 - x1))
            rotate_mat = np.asmatrix(np.eye(3))
            cos = np.cos(-roll)
            sin = np.sin(-roll)
            rotate_mat[0, 0] = cos
            rotate_mat[0, 1] = -sin
            rotate_mat[1, 0] = sin
            rotate_mat[1, 1] = cos
            inv_rotate_mat = rotate_mat.T

            # w, h = img_gray.shape[0:2]
            # tmp_image = cv2.warpAffine(img_gray, rotate_mat[:2, :], (w, h))
            # show_img_gray(tmp_image, save_name="rotate.jpg")

            # Scale
            # 放大或缩小
            scale = ow / eye_width
            scale_mat = np.asmatrix(np.eye(3))
            scale_mat[0, 0] = scale_mat[1, 1] = scale
            inv_scale = 1.0 / scale
            inv_scale_mat = np.asmatrix(np.eye(3))
            inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

            # Centre image
            # 中心区域
            centre_mat = np.asmatrix(np.eye(3))
            centre_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
            inv_centre_mat = np.asmatrix(np.eye(3))
            inv_centre_mat[:2, 2] = -centre_mat[:2, 2]

            # Get rotated and scaled, and segmented image
            transform_mat = centre_mat * scale_mat * rotate_mat * translate_mat
            inv_transform_mat = inv_translate_mat * inv_rotate_mat * inv_scale_mat * inv_centre_mat

            eye_image = cv2.warpAffine(img_gray, transform_mat[:2, :], (ow, oh))
            if is_left:
                eye_image = np.fliplr(eye_image)

            eyes_info.append({
                'image': eye_image,
                'inv_landmarks_transform_mat': inv_transform_mat,
                'side': 'left' if is_left else 'right',
            })

            print('[Info] eye_image: {}'.format(eye_image.shape))
            print('[Info] inv_transform_mat: {}'.format(inv_transform_mat.shape))

            # 关键步骤3, 绘制眼睛
            show_img_gray(eye_image)

        return eyes_info

    def process(self, img_bgr):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        box, landmarks = self.fd.get_main_faces_dwo(img_bgr)
        print('[Info] box: {}, landmarks: {}'.format(box, landmarks))
        eyes_info = self.crop_eyes(img_gray, landmarks)


def eye_detector_test():
    # img_path = os.path.join(IMGS_DIR, 'gg_large.jpg')
    img_path = os.path.join(IMGS_DIR, 'girl.jpg')
    img_bgr = cv2.imread(img_path)
    show_img_bgr(img_bgr)
    print('[Info] img_bgr: {}'.format(img_bgr.shape))

    ed = EyesDetector()
    ed.process(img_bgr)


def main():
    eye_detector_test()


if __name__ == "__main__":
    main()
