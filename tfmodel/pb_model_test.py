#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/5/19
"""
import cv2
import tensorflow as tf
import numpy as np
import os

from face_detector import FaceDetector  # 人脸检测和人脸关键点检测
from root_dir import IMGS_DIR, MODELS_DIR


def get_img_and_lms5():
    """
    测试获取图像和人脸关键点
    """
    img_path = os.path.join(IMGS_DIR, 'eyes_up.jpg')
    img_bgr = cv2.imread(img_path)

    fd = FaceDetector()
    main_box, face_landmarks = fd.get_main_faces_dwo(img_bgr)

    corner_list = [[2, 3], [1, 0]]  # 关键点: 左眼、右眼

    return img_bgr, face_landmarks, corner_list


def get_model_sess(pb_path):
    sess = tf.Session()

    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    sess.run(tf.global_variables_initializer())

    return sess


def get_ops_m(sess):
    """
    获取sess的op
    """
    eye_op = sess.graph.get_tensor_by_name('Placeholder:0')
    landmarks_op = sess.graph.get_tensor_by_name('upscale/mul:0')
    radius_op = sess.graph.get_tensor_by_name('radius/out/fc/BiasAdd:0')

    return eye_op, landmarks_op, radius_op


def eye_norm(eye):
    """
    输入图像108x180
    """
    eye = eye / 255.0 * 2.0 - 1.0
    eye = np.reshape(eye, (36, 60, 1))
    return eye


def preprocess(img_bgr, landmarks, corner_list):
    """
    获取眼睛区域
    """
    input_height, input_width = 36, 60  # 输入图像尺寸, 眼睛尺寸108x180
    scale = input_height / input_width  # 比例，转换为Tensor

    img_eyes = []  # 两个眼睛的图像
    offsets = []

    for corner in corner_list:
        eye_from, eye_to = (landmarks[corner[0]], landmarks[corner[1]])  # 眼睛坐标

        width = abs(eye_to[0] - eye_from[0])  # 眼睛宽度
        height = abs(eye_to[1] - eye_from[1])  # 眼睛高度

        eye_width = int(1.5 * width)  # 宽度
        eye_height = int(eye_width * scale)  # 高度

        offset_scale = eye_width / input_width  # 尺寸

        center_x = int(eye_from[0] + width / 2)  # 眼睛中心x坐标
        center_y = int(eye_from[1] + height / 2)  # 眼睛中心y坐标

        start_x = int(center_x - eye_width / 2)  # 起始点
        start_y = int(center_y - eye_height / 2)  # 终止点

        if start_x < 0 or start_y < 0:  # 异常处理
            return False, None, None

        img_eye = img_bgr[start_y:(start_y + eye_height), start_x:(start_x + eye_width)]  # 矩阵剪裁

        img_eye = cv2.resize(img_eye, (input_width, input_height))  # resize图像
        img_eye_gray = cv2.cvtColor(img_eye, cv2.COLOR_BGR2GRAY)  # 灰度图

        img_eye_gray = eye_norm(img_eye_gray)

        img_eyes.append(img_eye_gray)
        offsets.append([start_x, start_y, offset_scale])

    img_eyes_batch = np.asarray(img_eyes)
    offsets_batch = np.asarray(offsets)

    return True, img_eyes_batch, offsets_batch


def proprocess(eye_landmarks, offsets_params):
    """
    后处理还原坐标系

    :param eye_landmarks: 眼睛坐标点
    :param offsets_params: 偏移参数
    :return: 图像中眼睛的坐标点
    """
    eye_points = []
    for i in range(2):
        landmarks = eye_landmarks[i]
        params = offsets_params[i]
        os_x, os_y, os_scale = params
        for landmark in landmarks:
            x = int(landmark[0] * os_scale + os_x)
            y = int(landmark[1] * os_scale + os_y)
            eye_point = [x, y]
            eye_points.append(eye_point)
    return eye_points


def main():
    # corner_list是两个眼睛的眼角坐标点的索引，例如[[2, 3], [1, 0]], 左眼、右眼
    # img_bgr是原始图像, face_landmarks是人脸关键点, corner_list是眼角坐标点
    img_bgr, face_landmarks, corner_list = get_img_and_lms5()  # 加载数据, 替换人脸关键点检测即可

    # eyes_batch是两个眼睛的图像2x108x180x1
    state, img_eyes_batch, offsets_batch = preprocess(img_bgr, face_landmarks, corner_list)  # 预处理, 抠出眼睛图像

    if not state:
        print('[Info] 眼睛区域过小, 无法检测')
        return

    # 模型部分，替换为MNN
    pb_path = os.path.join(MODELS_DIR, 'gaze_opt_b2_small_3p.m.pb')
    sess = get_model_sess(pb_path)
    eye_op, landmarks_op, radius_op = get_ops_m(sess)

    feed_dict = {eye_op: img_eyes_batch}
    eye_landmarks = sess.run(landmarks_op, feed_dict=feed_dict)
    print('[Info] eye_landmarks: {}'.format(eye_landmarks.shape))

    eye_points = proprocess(eye_landmarks, offsets_batch)  # 后处理, 还原坐标系

    from utils.video_utils import draw_points
    draw_points(img_bgr, eye_points, save_name="out.jpg")  # 绘制坐标点


if __name__ == '__main__':
    main()
