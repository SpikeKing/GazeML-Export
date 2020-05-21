#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/4/10
"""

import copy
import os

import coloredlogs
import cv2 as cv
import numpy as np
import tensorflow as tf

from datasources import Video
from models import ELG
from util.gaze import draw_gaze

from root_dir import DATA_DIR, VIDS_DIR
from utils.project_utils import mkdir_if_not_exist, traverse_dir_files
from utils.video_utils import *


def save_img(img_path, img):
    cv.imwrite(img_path, img)


def process_output(output, batch_size, data_source, out_dir):
    """
    处理结果
    :param output: 输出
    :param batch_size: batch_size
    :param data_source: 数据源
    :param out_dir: 输出文件夹
    :return: None
    """
    for j in range(batch_size):
        frame_index = output['frame_index'][j]
        print('[Info] frame_index: {}'.format(frame_index))
        if frame_index not in data_source._frames:
            continue
        frame = data_source._frames[frame_index]

        # Decide which landmarks are usable
        heatmaps_amax = np.amax(output['heatmaps'][j, :].reshape(-1, 18), axis=0)
        can_use_eye = np.all(heatmaps_amax > 0.7)
        print('[Info] 是否使用眼睛 can_use_eye: {}'.format(can_use_eye))
        can_use_eyelid = np.all(heatmaps_amax[0:8] > 0.75)
        print('[Info] 是否使用眼睑 can_use_eyelid: {}'.format(can_use_eyelid))  # 眼睑
        can_use_iris = np.all(heatmaps_amax[8:16] > 0.8)
        print('[Info] 是否使用虹膜 can_use_iris: {}'.format(can_use_iris))  # 虹膜

        eye_index = output['eye_index'][j]
        bgr = frame['bgr']
        show_img_bgr(bgr)  # 原始图像

        eye = frame['eyes'][eye_index]
        eye_image = eye['image']

        eye_side = eye['side']
        print('eye_side: {}'.format(eye_side))
        eye_landmarks = output['landmarks'][j, :]
        print('eye_landmarks: {}'.format(eye_landmarks.shape))
        eye_radius = output['radius'][j][0]
        print('eye_radius: {}'.format(eye_radius))

        if eye_side == 'left':
            eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
            eye_image = np.fliplr(eye_image)

        # Embed eye image and annotate for picture-in-picture
        eye_upscale = 2
        eye_image_raw = cv.cvtColor(cv.equalizeHist(eye_image), cv.COLOR_GRAY2BGR)
        eye_image_raw = cv.resize(eye_image_raw, (0, 0), fx=eye_upscale, fy=eye_upscale)

        # show_img(eye_image_raw)  # 眼睛图像

        eye_image_annotated = np.copy(eye_image_raw)
        if can_use_eyelid:
            print('[Info] 眼睑: {}'.format(eye_landmarks[0:8]))
            cv.polylines(
                eye_image_annotated,
                [np.round(eye_upscale * eye_landmarks[0:8]).astype(np.int32)
                     .reshape(-1, 1, 2)],
                isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv.LINE_AA,
            )
            show_img_bgr(eye_image_annotated)  # 眼睑图像

        if can_use_iris:
            print('[Info] 虹膜: {}'.format(eye_landmarks[8:16]))
            cv.polylines(  # 虹膜
                eye_image_annotated,
                [np.round(eye_upscale * eye_landmarks[8:16]).astype(np.int32)
                     .reshape(-1, 1, 2)],
                isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,
            )

            print('[Info] 眼部中心: {}'.format(eye_landmarks[16, :]))
            cv.drawMarker(
                eye_image_annotated,
                tuple(np.round(eye_upscale * eye_landmarks[16, :]).astype(np.int32)),
                color=(0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=4,
                thickness=1, line_type=cv.LINE_AA,
            )
            show_img_bgr(eye_image_annotated)  # 虹膜图像

        frame_landmarks = (frame['smoothed_landmarks']
                           if 'smoothed_landmarks' in frame
                           else frame['landmarks'])
        for f, face in enumerate(frame['faces']):
            for landmark in frame_landmarks[f][:-1]:
                cv.drawMarker(bgr, tuple(np.round(landmark).astype(np.int32)),
                              color=(0, 0, 255), markerType=cv.MARKER_STAR,
                              markerSize=2, thickness=10, line_type=cv.LINE_AA)

            show_img_bgr(bgr)  # 脸部图像
            cv.rectangle(
                bgr, tuple(np.round(face[:2]).astype(np.int32)),
                tuple(np.round(np.add(face[:2], face[2:])).astype(np.int32)),
                color=(0, 255, 255), thickness=10, lineType=cv.LINE_AA,
            )
            show_img_bgr(bgr)  # 脸部图像

        # Transform predictions
        eye_landmarks = np.concatenate([eye_landmarks,
                                        [[eye_landmarks[-1, 0] + eye_radius,
                                          eye_landmarks[-1, 1]]]])
        eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)), 'constant', constant_values=1.0))
        eye_landmarks = (eye_landmarks * eye['inv_landmarks_transform_mat'].T)[:, :2]
        eye_landmarks = np.asarray(eye_landmarks)
        eyelid_landmarks = eye_landmarks[0:8, :]
        iris_landmarks = eye_landmarks[8:16, :]
        iris_centre = eye_landmarks[16, :]
        print('[Info] iris_centre: {}'.format(iris_centre))
        eyeball_centre = eye_landmarks[17, :]
        print('[Info] eyeball_centre: {}'.format(eyeball_centre))
        eyeball_radius = np.linalg.norm(eye_landmarks[18, :] - eye_landmarks[17, :])
        print('[Info] eyeball_radius: {}'.format(eyeball_radius))

        all_gaze_histories = []

        # Smooth and visualize gaze direction
        num_total_eyes_in_frame = len(frame['eyes'])
        if len(all_gaze_histories) != num_total_eyes_in_frame:
            all_gaze_histories = [list() for _ in range(num_total_eyes_in_frame)]
        gaze_history = all_gaze_histories[eye_index]
        if can_use_eye:
            # Visualize landmarks
            cv.drawMarker(  # Eyeball centre
                bgr, tuple(np.round(eyeball_centre).astype(np.int32)),
                color=(0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=4,
                thickness=1, line_type=cv.LINE_AA,
            )
            # cv.circle(  # Eyeball outline
            #     bgr, tuple(np.round(eyeball_centre).astype(np.int32)),
            #     int(np.round(eyeball_radius)), color=(0, 255, 0),
            #     thickness=1, lineType=cv.LINE_AA,
            # )

            # Draw "gaze"
            # from models.elg import estimate_gaze_from_landmarks
            # current_gaze = estimate_gaze_from_landmarks(
            #     iris_landmarks, iris_centre, eyeball_centre, eyeball_radius)
            i_x0, i_y0 = iris_centre
            e_x0, e_y0 = eyeball_centre
            theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
            phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
                                    -1.0, 1.0))
            current_gaze = np.array([theta, phi])
            gaze_history.append(current_gaze)
            gaze_history_max_len = 10
            if len(gaze_history) > gaze_history_max_len:
                gaze_history = gaze_history[-gaze_history_max_len:]
            draw_gaze(bgr, iris_centre, np.mean(gaze_history, axis=0),
                      length=160.0, thickness=4)

            show_img_bgr(bgr)  # 眼部图像
        else:
            gaze_history.clear()

        if can_use_eyelid:
            cv.polylines(
                bgr, [np.round(eyelid_landmarks).astype(np.int32).reshape(-1, 1, 2)],
                isClosed=True, color=(255, 255, 0), thickness=5, lineType=cv.LINE_AA,
            )
            show_img_bgr(bgr)  # 眼部图像

        if can_use_iris:
            cv.polylines(
                bgr, [np.round(iris_landmarks).astype(np.int32).reshape(-1, 1, 2)],
                isClosed=True, color=(0, 255, 255), thickness=5, lineType=cv.LINE_AA,
            )
            show_img_bgr(bgr)  # 眼部图像
            cv.drawMarker(
                bgr, tuple(np.round(iris_centre).astype(np.int32)),
                color=(0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=4,
                thickness=5, line_type=cv.LINE_AA,
            )
            show_img_bgr(bgr)  # 眼部图像
        final_img = copy.deepcopy(bgr)

        img_path = os.path.join(out_dir, 'frame-{}.jpg'.format(str(frame_index).zfill(4)))
        save_img(img_path=img_path, img=final_img)
    # show_img(final_img)  # 眼部图像

    print('[Info] 绘制完成!')


def save_video():
    img_dir = os.path.join(DATA_DIR, 'frames')
    paths_list, names_list = traverse_dir_files(img_dir)
    frame_list = []
    for name, path in zip(names_list, paths_list):
        frame = cv.imread(path)
        frame_list.append(frame)

    from_video = "normal_video.mp4"
    cap, n_frame, fps, h, w = init_vid(from_video)

    video_path = "norm.out.mp4"
    write_video(video_path, frame_list, fps, h, w)


def main():
    frames_dir = os.path.join(DATA_DIR, 'frames')
    mkdir_if_not_exist(VIDS_DIR)
    mkdir_if_not_exist(frames_dir)

    # from_video = os.path.join(VIDS_DIR, "normal_video.mp4")
    from_video = os.path.join(VIDS_DIR, "vid_no_glasses.mp4")

    # record_video = os.path.join(DATA_DIR, "normal_video.out.mp4")

    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level="INFO",
    )

    # Check if GPU is available
    from tensorflow.python.client import device_lib

    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_available = False
    try:
        gpus = [d for d in device_lib.list_local_devices(config=session_config)
                if d.device_type == 'GPU']
        gpu_available = len(gpus) > 0
    except Exception as e:
        print('[Info] GPU异常，使用CPU!')

    print('[Info] 是否启用GPU: {}'.format(gpu_available))
    # -----------------------------------------------------------------------#

    tf.logging.set_verbosity(tf.logging.INFO)
    session = tf.Session(config=session_config)

    batch_size = 2  # 设置batch
    print('[Info] 输入视频路径: {}'.format(from_video))
    assert os.path.isfile(from_video)

    # 模型包括大模型和小模型
    data_source = Video(from_video,
                        tensorflow_session=session,
                        batch_size=batch_size,
                        data_format='NCHW' if gpu_available else 'NHWC',
                        # eye_image_shape=(108, 180)
                        eye_image_shape=(36, 60)
                        )

    # Define model
    model = ELG(
        session, train_data={'videostream': data_source},
        # first_layer_stride=3,
        first_layer_stride=1,
        # num_modules=3,
        num_modules=2,
        # num_feature_maps=64,
        num_feature_maps=32,
        learning_schedule=[
            {
                'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
            },
        ],
    )

    infer = model.inference_generator()

    count = 0

    while True:
        print('')
        print('-' * 50)
        output = next(infer)
        process_output(output, batch_size, data_source, frames_dir)  # 处理输出
        count += 1
        print('count: {}'.format(count))
        if count == 10:
            break


if __name__ == '__main__':
    main()
