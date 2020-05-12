#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/4/29
"""

import os
import cv2
import numpy as np
import tensorflow as tf

from face_detector import FaceDetector
from utils.mat_utils import center_from_list
from utils.video_utils import show_img_bgr
from root_dir import MODELS_DIR, IMGS_DIR


class TFModelGenerater(object):
    def __init__(self):
        pass

    @staticmethod
    def get_img_and_lms5():
        """
        测试获取图像和人脸关键点
        """
        img_path = os.path.join(IMGS_DIR, 'girl.jpg')
        img_bgr = cv2.imread(img_path)

        fd = FaceDetector()
        main_box, main_landmarks = fd.get_main_faces_dwo(img_bgr)

        corner_list = [[2, 3], [1, 0]]  # 关键点, 左眼、右眼
        return img_bgr, main_landmarks, corner_list

    @staticmethod
    def get_img_and_lms51_fake():
        """
        测试获取图像和人脸关键点
        """
        img_path = os.path.join(IMGS_DIR, 'xxx.jpg')
        img_bgr = cv2.imread(img_path)

        # fd = FaceDetector()
        # main_box, main_landmarks = fd.get_main_faces_dwo(img_bgr)

        main_landmarks = [[0, 0]] * 51
        main_landmarks[6] = [202, 375]
        main_landmarks[7] = [283, 398]
        main_landmarks[14] = [413, 405]
        main_landmarks[15] = [497, 399]

        main_landmarks = np.array(main_landmarks, np.int32)

        corner_list = [[6, 7], [14, 15]]  # 关键点, 左眼、右眼

        return img_bgr, main_landmarks, corner_list

    @staticmethod
    def draw_img(img_bgr, eyes_landmarks, eyes_radius, offsets_list):
        """
        绘制图像
        """
        th = 1
        eye_upscale = 1
        eye_image_raw = cv2.resize(img_bgr, (0, 0), fx=eye_upscale, fy=eye_upscale)

        for eye_landmarks, eye_radius, offsets in zip(eyes_landmarks, eyes_radius, offsets_list):
            start_x, start_y, offset_scale = offsets
            eye_landmarks = eye_landmarks * offset_scale
            eye_landmarks = eye_landmarks + [start_x, start_y]

            cv2.polylines(
                eye_image_raw,
                [np.round(eye_upscale * eye_landmarks[0:8]).astype(np.int32)
                     .reshape(-1, 1, 2)],
                isClosed=True, color=(255, 255, 0),
                thickness=th, lineType=cv2.LINE_AA,
            )

            cv2.polylines(
                eye_image_raw,
                [np.round(eye_upscale * eye_landmarks[8:16]).astype(np.int32)
                     .reshape(-1, 1, 2)],
                isClosed=True, color=(0, 255, 255),
                thickness=th, lineType=cv2.LINE_AA,
            )

            iris_center = eye_landmarks[16]
            eye_center = eye_landmarks[17]
            eye_center_x = np.array(center_from_list(eye_landmarks[0:8]))

            # 虹膜中心
            cv2.drawMarker(
                eye_image_raw,
                tuple(np.round(eye_upscale * iris_center).astype(np.int32)),
                color=(255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
                thickness=th, line_type=cv2.LINE_AA,
            )

            # 眼睑中心
            cv2.drawMarker(
                eye_image_raw,
                tuple(np.round(eye_upscale * eye_center).astype(np.int32)),
                color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
                thickness=th, line_type=cv2.LINE_AA,
            )

            # 眼睑中心 - 计算
            cv2.drawMarker(
                eye_image_raw,
                tuple(np.round(eye_upscale * eye_center_x).astype(np.int32)),
                color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=4,
                thickness=th, line_type=cv2.LINE_AA,
            )
            show_img_bgr(eye_image_raw, save_name="xxx.jpg")  # 眼睑中心

    def equalize_histogram_with_tf(self, image):
        """
        cv2.equalizeHist()
        :param image: 灰度图HxWx1
        """
        from tfmodel.ops_customs import histogram_fixed_width, cumsum

        values_range = tf.constant([0., 255.], dtype=tf.float32)
        histogram = histogram_fixed_width(tf.to_float(image), values_range, 256)
        cdf = cumsum(histogram)

        xx = tf.greater(cdf, tf.convert_to_tensor(0, dtype=tf.int32))
        yy = tf.cast(tf.where(xx), dtype=tf.int32)
        zz = tf.reduce_min(yy)
        cdf_min = cdf[zz]

        img_shape = tf.shape(image)
        pix_cnt = img_shape[-3] * img_shape[-2]
        px_map = tf.floor(tf.to_float(cdf - cdf_min) * 255. / tf.to_float(pix_cnt - 1))
        px_map = tf.cast(px_map, tf.int32)

        eq_hist = tf.expand_dims(tf.gather(px_map, tf.cast(image, tf.int32)), 2)
        return eq_hist

    def eye_preprocess_with_tf(self, eye):
        """
        输入图像108x180
        :param eye: [108x180]
        :return: 1x108x180x1
        """
        eye = tf.reshape(eye, (108, 180, 1))
        eye = self.equalize_histogram_with_tf(eye)
        eye = tf.cast(eye, tf.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        eye = tf.reshape(eye, (1, 108, 180, 1))
        return eye

    def get_batch_with_tf(self, img_bgr, landmarks, corner_list):
        """
        获取眼睛区域

        :param img_bgr: 800x800
        :param landmarks: 5个关键点
        :param corner_list: 左眼、右眼，从左到右
        :return: 眼睛的灰度图108x180
        """

        from tfmodel.ops_customs import tf_abs

        input_height, input_width = 108, 180  # 输入图像尺寸, 眼睛尺寸108x180
        scale = tf.convert_to_tensor(input_height / input_width, tf.float32)  # 比例，转换为Tensor

        img_eyes = []  # 两个眼睛的图像
        offsets = []

        for corner in corner_list:
            eye_from, eye_to = (landmarks[corner[0]], landmarks[corner[1]])  # 眼睛坐标

            width = tf_abs(eye_to[0] - eye_from[0])  # 眼睛宽度
            height = tf_abs(eye_to[1] - eye_from[1])  # 眼睛高度

            eye_width = 1.5 * width  # 宽度
            eye_height = eye_width * scale  # 高度

            center_x = tf.cast(eye_from[0] + width / 2, tf.float32)  # 眼睛中心x坐标
            center_y = tf.cast(eye_from[1] + height / 2, tf.float32)  # 眼睛中心y坐标

            start_x = tf.cast(center_x - eye_width / 2, tf.int32, name="offset_x")  # 起始点
            start_y = tf.cast(center_y - eye_height / 2, tf.int32, name="offset_y")  # 终止点

            offset_scale = tf.cast(eye_width / input_width, tf.float32, name="offset_s")  # 尺寸

            img_eye = tf.slice(img_bgr, [0, start_y, start_x, 0], [1, eye_height, eye_width, 3])  # 矩阵剪裁
            img_eye = tf.image.resize(img_eye, (input_height, input_width))  # resize图像

            img_eye_gray = tf.image.rgb_to_grayscale(img_eye)  # 灰度图
            img_eye_gray = tf.cast(img_eye_gray, tf.uint8)  # unit8

            img_eye_pre = self.eye_preprocess_with_tf(img_eye_gray)  # 预处理灰度图像
            img_eyes.append(img_eye_pre)  # 图像列表

            offsets_tensor = tf.convert_to_tensor(
                [tf.cast(start_x, dtype=tf.float32), tf.cast(start_y, dtype=tf.float32), offset_scale])
            offsets.append(offsets_tensor)

        eyes_concat = tf.concat(img_eyes, axis=0, name='concat')  # 连接2个图像
        offsets_concat = tf.convert_to_tensor(offsets, name='axes_offsets')  # 连接2个偏移数据

        return eyes_concat, offsets_concat

    def get_model_lms(self, corner_list):
        """
        替换模型入口
        """
        img_bgr = tf.placeholder(tf.float32, [1, None, None, 3], name='img_bgr')
        # face_lm = tf.placeholder(tf.float32, [5, 2], name='landmarks')
        face_lm = tf.placeholder(tf.float32, [51, 2], name='landmarks')

        # corner_list = [[2, 3], [1, 0]]  # 关键点: 左眼 右眼
        # corner_list = [[6, 7], [14, 15]]  # 关键点, 左眼、右眼

        eyes_tensor, offsets_tensor = self.get_batch_with_tf(img_bgr, face_lm, corner_list)  # 获取眼睛图像2x108x180x1

        pb_path = os.path.join(MODELS_DIR, 'gaze_opt_b2.m.pb')  # 读取graph

        graph = tf.get_default_graph()
        graph_def = graph.as_graph_def()
        graph_def.ParseFromString(tf.gfile.FastGFile(pb_path, 'rb').read())
        tf.import_graph_def(graph_def, name='')

        eye_input = graph.get_tensor_by_name('Placeholder:0')  # 输入

        tf.contrib.graph_editor.reroute_ts(eyes_tensor, eye_input)  # 替换输出和placeholder

        landmarks_op = graph.get_tensor_by_name('upscale/mul:0')
        radius_op = graph.get_tensor_by_name('radius/out/fc/BiasAdd:0')
        offsets_op = graph.get_tensor_by_name('axes_offsets:0')

        return landmarks_op, radius_op, offsets_op

    def save_pb_model(self, sess, export_pb_name):
        """
        存储PB模型
        :param sess: session
        :param export_pb_name: pb名称
        :return: 写入文件
        """
        from tensorflow.python.framework import graph_util
        constant_graph = graph_util.convert_variables_to_constants(
            sess, sess.graph_def,
            [
                'upscale/mul',  # landmarks
                'radius/out/fc/BiasAdd',  # radius
                'axes_offsets',  # offsets
            ])
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(constant_graph.SerializeToString())
        tf.train.write_graph(graph_def, './', '{}.pb'.format(export_pb_name), as_text=False)
        tf.train.write_graph(graph_def, './', '{}.pbtxt'.format(export_pb_name), as_text=True)

    def generate_model(self):
        """
        生成模型
        """
        # img_bgr, face_landmarks, corner_list = self.get_img_and_lms5()
        img_bgr, face_landmarks, corner_list = self.get_img_and_lms51_fake()
        img_bgr_batch = np.expand_dims(img_bgr, axis=0)

        landmarks_op, radius_op, offsets_op = self.get_model_lms(corner_list)
        sess = tf.Session()

        eyes_landmarks, eyes_radius, offsets = \
            sess.run((landmarks_op, radius_op, offsets_op),
                     feed_dict={'img_bgr:0': img_bgr_batch, "landmarks:0": face_landmarks})

        print('[Info] eyes_landmarks: {}, eyes_radius: {}, offsets: {}'
              .format(eyes_landmarks.shape, eyes_radius.shape, offsets.shape))

        self.draw_img(img_bgr, eyes_landmarks, eyes_radius, offsets)  # 绘制图像
        print('[Info] 生成完成!')

        export_pb_name = "gaze-faceX-lms51"
        self.save_pb_model(sess, export_pb_name)
        print('[Info] 存储PB模型完成!')

    def test_model(self):
        """
        测试模型
        """
        # img_bgr, face_landmarks, corner_list = self.get_img_and_lms5()
        img_bgr, face_landmarks, corner_list = self.get_img_and_lms51_fake()
        img_bgr_batch = np.expand_dims(img_bgr, axis=0)

        pb_path = os.path.join('./gaze-faceX-lms51.pb')  # 读取graph

        graph = tf.get_default_graph()
        graph_def = graph.as_graph_def()
        graph_def.ParseFromString(tf.gfile.FastGFile(pb_path, 'rb').read())
        tf.import_graph_def(graph_def, name='')

        landmarks_op = graph.get_tensor_by_name('upscale/mul:0')
        radius_op = graph.get_tensor_by_name('radius/out/fc/BiasAdd:0')
        offsets_op = graph.get_tensor_by_name('axes_offsets:0')

        sess = tf.Session()
        eyes_landmarks, eyes_radius, offsets = \
            sess.run((landmarks_op, radius_op, offsets_op),
                     feed_dict={'img_bgr:0': img_bgr_batch, "landmarks:0": face_landmarks})

        self.draw_img(img_bgr, eyes_landmarks, eyes_radius, offsets)  # 绘制图像
        print('[Info] 测试模型完成!')


def tf_model_generater_test():
    tmg = TFModelGenerater()
    tmg.generate_model()  # 生成
    # tmg.test_model()  # 测试


def main():
    tf_model_generater_test()


if __name__ == '__main__':
    main()
