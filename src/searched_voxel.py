#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import argparse
import pathlib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_search_area_filter(detection_range, voxel_width):
    '''
    :param detection_range: 検知できる距離
    :return:
    '''

    # 検知距離と、ボクセル中心からボクセル角までの距離を足してボクセルのサイズで割ったものを
    half_diagonal = np.sqrt(pow(voxel_width / 2, 2) * 3)
    max_range_voxels = int(np.ceil((detection_range + half_diagonal) / voxel_width))
    min_range = detection_range - half_diagonal
    matrix = np.zeros((max_range_voxels * 2 + 1, max_range_voxels * 2 + 1, max_range_voxels * 2 + 1))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                origin = max_range_voxels
                voxel_origin = np.array([(i - origin) * voxel_width,
                                         (j - origin) * voxel_width,
                                         (k - origin) * voxel_width])
                if np.linalg.norm(voxel_origin) <= min_range:
                    matrix[i, j, k] = 1
    return matrix


def apply_filter(f, map, position, detection_range, voxel_width):
    copied_f = np.array(f)
    unknown_area = np.where(f == 0)
    max_range_voxels = int((f.shape[0] - 1) / 2)
    f_origin = max_range_voxels

    voxel_position = [int(np.floor(p / voxel_width)) for p in position]
    voxel_origin = [p * voxel_width + voxel_width / 2 for p in voxel_position]

    for unknown_voxel in zip(unknown_area[0], unknown_area[1], unknown_area[2]):
        unkwon_voxel_origin = np.array([(i - f_origin) * voxel_width for i in unknown_voxel])
        if np.linalg.norm(unkwon_voxel_origin - (position - voxel_origin)) <= detection_range:
            copied_f[unknown_voxel] = 1

    x_min, x_max = voxel_position[0] - max_range_voxels, voxel_position[0] + max_range_voxels
    y_min, y_max = voxel_position[1] - max_range_voxels, voxel_position[1] + max_range_voxels
    z_min, z_max = voxel_position[2] - max_range_voxels, voxel_position[2] + max_range_voxels
    low_cut_x = x_min < 0
    low_cut_y = y_min < 0
    low_cut_z = z_min < 0
    x_size = min(x_max, map.shape[0]) - max(0, x_min)
    y_size = min(y_max, map.shape[1]) - max(0, y_min)
    z_size = min(z_max, map.shape[2]) - max(0, z_min)
    adapting_filter = copied_f
    adapting_filter = adapting_filter[-x_size:, :, :] if low_cut_x else adapting_filter[:x_size, :, :]
    adapting_filter = adapting_filter[:, -y_size:, :] if low_cut_y else adapting_filter[:, :y_size, :]
    adapting_filter = adapting_filter[:, :, -z_size:] if low_cut_z else adapting_filter[:, :, :z_size]

    map[max(0, x_min):min(x_max, map.shape[0]),
    max(0, y_min):min(y_max, map.shape[1]),
    max(0, z_min):min(z_max, map.shape[2])] += adapting_filter


if __name__ == '__main__':
    # 読み込み作業
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str,
                        help='csvファイルのパス')
    parser.add_argument('-d', '--detection_range', type=float, default=200,
                        help='人の検出できる距離 [cm]')

    args = parser.parse_args()
    detection_range = args.detection_range

    file = pathlib.Path(args.file_path)
    parameter_rows = 1  # パラメータの書かれてる行数
    df = pd.read_csv(file,
                     skiprows=parameter_rows,
                     names=['step', 'id', 'x', 'y', 'z', 'rx', 'ry', 'rz'])

    # 全体マップの生成
    height = 400  # [cm]
    width = 8000  # [cm]
    depth = 15000  # [cm]
    voxel_width = 10  # [cm]
    map = np.zeros((int(np.ceil(width / voxel_width)),
                    int(np.ceil(depth / voxel_width)),
                    int(np.ceil(height / voxel_width)),))

    search_area_filter = create_search_area_filter(detection_range, voxel_width)

    # マップの探索範囲を埋めていく
    for i, row in df.iterrows():
        position = row[['x', 'y', 'z']]
        apply_filter(search_area_filter, map, position, detection_range, voxel_width)

    tmp = np.where(map != 0)
    searched_area = np.array([[i, j, k] for i, j, k in zip(tmp[0], tmp[1], tmp[2])])
    np.savetxt('searched_area.csv', searched_area, delimiter=',', fmt='%d')


    # 球体になってるかとかのテストコード
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    tmp2 = np.zeros((10, 10, 10))
    for i in range(10):
        for j in range(10):
            for k in range(10):
                tmp2[i, j, k] = map[45 + i, 45 + j, 15 + k]
    ax.voxels(tmp2, edgecolor="k")

    plt.savefig('b.png')
    # plt.show()
