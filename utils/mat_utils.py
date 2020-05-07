#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/4/17
"""

import math


def calculate_arrow(pnt1, pnt2):
    """
    计算角度 pnt2 -> pnt1
    :param pnt1: 终止点
    :param pnt2: 起始点
    :return: 角度
    """
    myradians = math.atan2(pnt1[1] - pnt2[1], pnt1[0] - pnt2[0])
    mydegrees = math.degrees(myradians)
    mydegrees = float(mydegrees + 360) % float(360.)
    return mydegrees


def box_from_list(p_list):
    """
    提取列表：最小点、最大点、中心点
    """
    import numpy as np
    x_list = [p[0] for p in p_list]
    y_list = [p[1] for p in p_list]

    x_arr = np.array(x_list)
    y_arr = np.array(y_list)

    x_min, x_max = int(np.min(x_arr)), int(np.max(x_arr))
    y_min, y_max = int(np.min(y_arr)), int(np.max(y_arr))

    # print(x_min, x_max)
    # print(y_min, y_max)

    x_c = int((x_max - x_min) // 2 + x_min)
    y_c = int((y_max - y_min) // 2 + y_min)

    return (x_min, y_min), (x_max, y_max), (x_c, y_c)


def wh_from_list(p_list):
    import numpy as np
    x_list = [p[0] for p in p_list]
    y_list = [p[1] for p in p_list]

    x_arr = np.array(x_list)
    y_arr = np.array(y_list)

    x_min, x_max = int(np.min(x_arr)), int(np.max(x_arr))
    y_min, y_max = int(np.min(y_arr)), int(np.max(y_arr))

    w = x_max - x_min
    h = y_max - y_min

    return w, h


def center_from_list(p_list):
    """
    提取列表中心点
    """
    _, _, (x_c, y_c) = box_from_list(p_list)
    return x_c, y_c
