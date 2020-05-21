#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/5/7
"""
import copy
import os
import cv2
import tensorflow as tf
import numpy as np

from face_detector import FaceDetector
from root_dir import IMGS_DIR, MODELS_DIR
from utils.mat_utils import calculate_arrow, center_from_list
from utils.video_utils import show_img_bgr, show_img_gray


class EyesDetector(object):
    def __init__(self):
        self.fd = FaceDetector()

        # pb_path = os.path.join(MODELS_DIR, 'gaze_opt_b1.m.pb')
        # pb_path = os.path.join(MODELS_DIR, 'gaze_opt_b2.m.pb')  # 108, 180
        # pb_path = os.path.join(MODELS_DIR, 'gaze_opt_b2_small.pb')
        pb_path = os.path.join(MODELS_DIR, 'gaze_opt_b2_small.m.pb')  # 36, 60

        self.sess = self.get_model_sess(pb_path)

    def get_model_sess(self, pb_path):
        sess = tf.Session()

        with tf.gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        sess.run(tf.global_variables_initializer())

        return sess

    def get_ops(self, sess):
        frame_index_op = sess.graph.get_tensor_by_name('Video/fifo_queue_DequeueMany:0')
        # eye_op = sess.graph.get_tensor_by_name('Placeholder:0')
        eye_op = sess.graph.get_tensor_by_name('Video/fifo_queue_DequeueMany:1')
        eye_index_op = sess.graph.get_tensor_by_name('Video/fifo_queue_DequeueMany:2')
        heatmaps_op = sess.graph.get_tensor_by_name('hourglass/hg_2/after/hmap/conv/BiasAdd:0')
        landmarks_op = sess.graph.get_tensor_by_name('upscale/mul:0')
        radius_op = sess.graph.get_tensor_by_name('radius/out/fc/BiasAdd:0')

        return eye_op, landmarks_op, radius_op

    def get_ops_m(self, sess):
        eye_op = sess.graph.get_tensor_by_name('Placeholder:0')
        landmarks_op = sess.graph.get_tensor_by_name('upscale/mul:0')
        radius_op = sess.graph.get_tensor_by_name('radius/out/fc/BiasAdd:0')

        return eye_op, landmarks_op, radius_op

    def crop_eyes(self, img_gray, landmarks):
        """
        使用仿射变换，抠出眼睛区域
        """
        eyes_info = []
        landmarks = np.array(landmarks)

        # Final output dimensions
        # oh, ow = (108, 180)
        oh, ow = (36, 60)

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
            # show_img_gray(eye_image, 'eyes-{}.jpg'.format(str(is_left)))

        return eyes_info

    def eye_preprocess(self, eye_in):
        """
        眼睛预处理
        """
        _data_format = 'NHWC'

        # show_img_gray(eye_in, save_name='pre.jpg')
        eye = cv2.equalizeHist(eye_in)
        # show_img_gray(eye, save_name='eh.jpg')

        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        eye = np.expand_dims(eye, -1 if _data_format == 'NHWC' else 0)
        return eye

    def get_eyes_batch(self, eyes_info):
        """
        检测目光关键点
        """
        if len(eyes_info) != 2:
            return

        eye1 = self.eye_preprocess(eyes_info[0]['image'])
        eye2 = self.eye_preprocess(eyes_info[1]['image'])

        print('[Info] eye1: {}'.format(eye1.shape))
        print('[Info] eye2: {}'.format(eye1.shape))

        eyes = np.concatenate((eye1, eye2), axis=0)
        eyes_batch = eyes.reshape(2, 36, 60, 1)

        return eyes_batch

    def draw_img_with_info(self, info_dict):
        """
        绘制图像
        """
        info_dict = copy.deepcopy(info_dict)
        img_op = info_dict['img_bgr']

        th = 1
        for i in range(2):
            eye_landmarks = info_dict['eye_landmarks'][i]
            eye_image = info_dict['eyes_info'][i]['image']
            eye = info_dict['eyes_info'][i]
            eye_side = info_dict['eyes_info'][i]['side']

            if eye_side == 'left':
                eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]

            eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)), 'constant', constant_values=1.0))
            eye_landmarks = (eye_landmarks * eye['inv_landmarks_transform_mat'].T)[:, :2]
            eye_landmarks = np.asarray(eye_landmarks)

            # 眼睑范围[0:8]
            cv2.polylines(
                img_op,
                [np.round(eye_landmarks[0:8]).astype(np.int32).reshape(-1, 1, 2)],
                isClosed=True, color=(255, 255, 0),
                thickness=th, lineType=cv2.LINE_AA,
            )
            show_img_bgr(img_op)

            # 虹膜范围[8:16]
            cv2.polylines(  # 虹膜
                img_op,
                [np.round(eye_landmarks[8:16]).astype(np.int32).reshape(-1, 1, 2)],
                isClosed=True, color=(0, 255, 255),
                thickness=th, lineType=cv2.LINE_AA,
            )
            show_img_bgr(img_op)

            # 计算视线方向强度
            iris_center = eye_landmarks[16]
            eyeball_center = eye_landmarks[17]

            # 绘制瞳孔中心
            cv2.drawMarker(
                img_op,
                tuple(np.round(iris_center).astype(np.int32)),
                color=(255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
                thickness=th + 1, line_type=cv2.LINE_AA,
            )
            show_img_bgr(img_op)  # 瞳孔中心

            # 绘制眼睛中心
            cv2.drawMarker(
                img_op,
                tuple(np.round(eyeball_center).astype(np.int32)),
                color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=4,
                thickness=th + 1, line_type=cv2.LINE_AA,
            )
            # show_img_bgr(img_op, save_name='img_{}.jpg'.format(i))  # 眼睛中心

        return info_dict

    @staticmethod
    def draw_arrow(image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
        """Draw gaze angle on given image with a given eye positions."""
        image_out = image_in
        if len(image_out.shape) == 2 or image_out.shape[2] == 1:
            image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
        dx = -length * np.sin(pitchyaw[1])
        dy = -length * np.sin(pitchyaw[0])

        v_bias = np.linalg.norm(np.round(eye_pos) - np.round([eye_pos[0] + dx, eye_pos[1] + dy]))
        # print('[Info] 强度: {}'.format(v_bias))

        cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                        tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                        thickness, cv2.LINE_AA, tipLength=0.2)
        return image_out

    def draw_eye_with_info(self, info_dict):
        """
        绘制眼睛信息
        """
        info_dict = copy.deepcopy(info_dict)
        print('[Info] 绘制眼睛!')

        eye_upscale = 4  # 等比例放大
        th = 2 * eye_upscale
        for i in range(2):
            eye_landmarks = info_dict['eye_landmarks'][i]
            eye_image = info_dict['eyes_info'][i]['image']
            eye_radius = info_dict['eye_radius'][i][0]
            eye_side = info_dict['eyes_info'][i]['side']

            if eye_side == 'left':
                eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
                eye_image = np.fliplr(eye_image)

            eye_image = cv2.equalizeHist(eye_image)
            eye_image_raw = cv2.cvtColor(eye_image, cv2.COLOR_GRAY2BGR)
            eye_image_raw = cv2.resize(eye_image_raw, (0, 0), fx=eye_upscale, fy=eye_upscale)

            # 眼睑区域[0:8]
            eye_image_annotated = np.copy(eye_image_raw)
            cv2.polylines(
                eye_image_annotated,
                [np.round(eye_upscale * eye_landmarks[0:8]).astype(np.int32)
                     .reshape(-1, 1, 2)],
                isClosed=True, color=(255, 255, 0),
                thickness=th, lineType=cv2.LINE_AA,
            )
            # show_img_bgr(eye_image_annotated)

            # 虹膜区域[8:16]
            cv2.polylines(
                eye_image_annotated,
                [np.round(eye_upscale * eye_landmarks[8:16]).astype(np.int32)
                     .reshape(-1, 1, 2)],
                isClosed=True, color=(255, 128, 0),
                thickness=th, lineType=cv2.LINE_AA,
            )
            # show_img_bgr(eye_image_annotated)

            iris_center = eye_landmarks[16]
            eyeball_center = eye_landmarks[17]
            eyeball_center_x = np.array(center_from_list(eye_landmarks[0:8]))

            # 虹膜中心[16]
            cv2.drawMarker(
                eye_image_annotated,
                tuple(np.round(eye_upscale * iris_center).astype(np.int32)),
                color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=4,
                thickness=th + 1, line_type=cv2.LINE_AA,
            )
            # show_img_bgr(eye_image_annotated)

            # 眼睛中心[17]
            cv2.drawMarker(
                eye_image_annotated,
                tuple(np.round(eye_upscale * eyeball_center).astype(np.int32)),
                color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
                thickness=th + 1, line_type=cv2.LINE_AA,
            )
            # show_img_bgr(eye_image_annotated)

            # 眼睑中心 - 计算
            cv2.drawMarker(
                eye_image_annotated,
                tuple(np.round(eye_upscale * eyeball_center_x).astype(np.int32)),
                color=(0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
                thickness=th + 1, line_type=cv2.LINE_AA,
            )
            # show_img_bgr(eye_image_annotated)

            # 眼睛半径
            cv2.circle(
                eye_image_annotated,
                tuple(np.round(eye_upscale * eyeball_center)),
                int(eye_radius * eye_upscale),
                color=(128, 0, 255),
                thickness=th + 1
            )
            # show_img_bgr(eye_image_annotated)

            i_x0, i_y0 = iris_center
            e_x0, e_y0 = eyeball_center

            theta = -np.arcsin(np.clip((i_y0 - e_y0) / eye_radius, -1.0, 1.0))
            phi = np.arcsin(np.clip((i_x0 - e_x0) / (eye_radius * -np.cos(theta)), -1.0, 1.0))

            current_gaze = np.array([theta, phi])
            image_out = self.draw_arrow(
                eye_image_annotated,
                tuple(np.round(eye_upscale * eye_landmarks[16]).astype(np.int32)),
                current_gaze,
                length=50.0 * eye_upscale,
                thickness=th + 2
            )
            show_img_bgr(image_out, save_name='eye_{}.jpg'.format(eye_side))  # 瞳孔中心
            print()
            # import os
            # from root_dir import IMGS_DIR
            # img_path = os.path.join(IMGS_DIR, 'xxx.gaze.{}.jpg'.format(str(eye_side)))
            # cv2.imwrite(img_path, image_out)

    def merge_info(self, img_bgr, img_gray, box, landmarks, eyes_info, eye_landmarks):
        """
        合并算法信息
        """
        info_dict = dict()

        info_dict['img_bgr'] = img_bgr
        info_dict['img_gray'] = img_gray
        info_dict['face_bbox'] = box
        info_dict['face_landmarks'] = landmarks
        info_dict['eyes_info'] = eyes_info
        info_dict['eye_landmarks'] = eye_landmarks

        return info_dict

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
            ])
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(constant_graph.SerializeToString())
        tf.train.write_graph(graph_def, './', '{}.pb'.format(export_pb_name), as_text=False)
        tf.train.write_graph(graph_def, './', '{}.pbtxt'.format(export_pb_name), as_text=True)

    def process(self, img_bgr):
        """
        处理图像
        """
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        box, landmarks = self.fd.get_main_faces_dwo(img_bgr)
        print('[Info] box: {}, landmarks: {}'.format(box, landmarks))
        eyes_info = self.crop_eyes(img_gray, landmarks)
        eyes_batch = self.get_eyes_batch(eyes_info)

        eye_op, landmarks_op, radius_op = self.get_ops_m(self.sess)
        # Placeholder_1 = self.sess.graph.get_tensor_by_name('learning_params/Placeholder_1:0')
        # feed_dict = {eye: eyeI, Placeholder_1: False}
        feed_dict = {eye_op: eyes_batch}
        eye_landmarks, eye_radius = self.sess.run((landmarks_op, radius_op), feed_dict=feed_dict)
        print('[Info] eye_landmarks: {}'.format(eye_landmarks.shape))
        print('[Info] eye_landmarks: {}'.format(eye_landmarks))

        info_dict = self.merge_info(img_bgr, img_gray, box, landmarks, eyes_info, eye_landmarks)

        # self.draw_eye_with_info(info_dict)
        self.draw_img_with_info(info_dict)


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
