"""
This module contains functions to generate outputs in form of images or text
"""

import math

import numpy as np
import json
import cv2
from datetime import datetime

import util


def draw(input_image, subsets, candidates, resize_fac=1):
    """
    This function draws body parts and joints over an image

    :param input_image: cv2 image to draw on
    :param subsets: ndarray of subsets, one line for subset (i.e. person) as returned by processing.extract_parts()
    :param candidates: ndarray of body parts candidates, one line for part as returned by processing.extract_parts()
    :param resize_fac: resize factor of the output
    :return: cv2 image with parts and joints drawn on
    """
    canvas = input_image.copy()

    for part in candidates:
        i = part[-1].astype(int)
        a = part[0].astype(int) * resize_fac
        b = part[1].astype(int) * resize_fac
        cv2.circle(canvas, (a, b), 2, util.colors[i], thickness=-1)

    stickwidth = 4

    for i in range(17):
        for s in subsets:
            index = s[util.part_seq[i] - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            y = candidates[index.astype(int), 0]
            x = candidates[index.astype(int), 1]
            m_x = np.mean(x)
            m_y = np.mean(y)
            length = np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
            angle = math.degrees(math.atan2(x[0] - x[1], y[0] - y[1]))
            polygon = cv2.ellipse2Poly((int(m_y * resize_fac), int(m_x * resize_fac)),
                                       (int(length * resize_fac / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, util.colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas


def init_out_file(path):
    """
    Inits the output file with a header.
    The output file will contain the body parts positions for each subset (i.e. person), for each frame
    :param path: path of the file to write
    """

    with open(path, 'w') as f:
        f.write('[Header]\n')
        f.write('part_names: ' + str(util.get_parts_order()))
        f.write('\n')

    # header = {'part_names': util.get_parts_order()}
    # with open(path, 'w') as f:
    #     json.dump(header, f, indent=4)


def append_to_out_file(path, subsets, candidates):
    """
    Appends to the output file the body parts positions of each subset (i.e. person).
    The format is:

    `
    [timestamp in ISO 8601 format]
    one line for each subset, three columns for each body part in the order (x, y, score) and a final column with the
    overall score of the subset
    `
    Body parts are ordered as returned by util.get_parts_order()

    :param path: path of the file to write
    :param subsets: ndarray of subsets, one line for subset (i.e. person) as returned by processing.extract_parts()
    :param candidates: ndarray of body parts candidates, one line for part as returned by processing.extract_parts()
    """

    # Timestamp in ISO 8601 format
    timestamp = datetime.now().isoformat()

    with open(path, 'a') as f:
        f.write(timestamp + '\n\n')

        for i in range(subsets.shape[0]):
            line = []
            for j in range(subsets.shape[1] - 2):
                rowid = int(subsets[i, j])
                if rowid != -1:
                    # Add x, y and score of the joint
                    line += candidates[rowid, :2].astype(int).tolist() + [candidates[rowid, 2]]
                else:
                    line += [-1, -1, -1]
            line += [subsets[i, -2]]
            f.write(str(line) + '\n')

    # subsets_dict = {}
    # for i, subset in enumerate(subsets):
    #     # I need to write all elements except the second last one as int
    #     slist = subset[:-2].astype(int).tolist() + [float(subset[-2]), int(subset[-1])]
    #     subsets_dict[i+1] = slist  # start counting subsets from 1
    #
    # candidates_dict = {}
    # for i, candidate in enumerate(candidates):
    #     # I need to write all elements except the second last one as int
    #     clist = candidate[:-2].astype(int).tolist() + [float(candidate[-2]), int(candidate[-1])]
    #     candidates_dict[i] = clist
    #
    # data_dict = {'timestamp': timestamp, 'subsets': subsets_dict, 'parts': candidates_dict}
    #
    # with open(path, 'a') as f:
    #     json.dump(data_dict, f, indent=4)
