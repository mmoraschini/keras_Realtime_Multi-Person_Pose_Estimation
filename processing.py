"""
Module to process images and extract body parts and subsets, i.e. groups of parts, ideally persons
"""

import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter

import util


def extract_parts(input_image, params, model, model_params):
    """
    This function uses a Neural Network model to recognise persons and their body parts inside an image

    :param input_image: cv2 image to analyse
    :param params: parameters of the algorithm read from the config file
    :param model: keras weights
    :param model_params: parameters of the model read from the config file
    :return:
        an ndarray where each line is a subset (ideally a person), the first 18 columns are the ids of the parts which
            map to the corresponding line in the second retruned variable, the 19th is the confidence of the subset and
            the last one is the number of valid parts for that subset
        an ndarray where each line is a part, the first two columns the x and y of the part, the third column the
            confidence of that part and the fourth the part type
    """
    multiplier = [x * model_params['boxsize'] / input_image.shape[0] for x in params['scale_search']]

    # Body parts location heatmap, one per part (19)
    heatmap_avg = np.zeros((input_image.shape[0], input_image.shape[1], 19))
    # Part affinities, one per limb (38)
    paf_avg = np.zeros((input_image.shape[0], input_image.shape[1], 38))

    for scale in multiplier:
        image_to_test = cv2.resize(input_image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        image_to_test_padded, pad = util.pad_right_down_corner(image_to_test, model_params['stride'],
                                                               model_params['padValue'])

        # required shape (1, width, height, channels)
        input_img = np.transpose(np.float32(image_to_test_padded[:, :, :, np.newaxis]), (3, 0, 1, 2))

        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:image_to_test_padded.shape[0] - pad[2], :image_to_test_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:image_to_test_padded.shape[0] - pad[2], :image_to_test_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    # all_peaks is an array where each element contains an array of tuples. Each element of the tuple is a peak for
    # a given body part
    all_peaks = []
    peak_counter = 0

    for i in range(18):
        hmap_ori = heatmap_avg[:, :, i]
        hmap = gaussian_filter(hmap_ori, sigma=3)

        # Find the pixel that has maximum value compared to those around it
        hmap_left = np.zeros(hmap.shape)
        hmap_left[1:, :] = hmap[:-1, :]
        hmap_right = np.zeros(hmap.shape)
        hmap_right[:-1, :] = hmap[1:, :]
        hmap_up = np.zeros(hmap.shape)
        hmap_up[:, 1:] = hmap[:, :-1]
        hmap_down = np.zeros(hmap.shape)
        hmap_down[:, :-1] = hmap[:, 1:]

        # reduce needed because there are > 2 arguments
        peaks_binary = np.logical_and.reduce(
            (hmap >= hmap_left, hmap >= hmap_right, hmap >= hmap_up, hmap >= hmap_down, hmap > params['thre1']))

        # peaks will contain a tuple for each peak in the image
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (hmap_ori[x[1], x[0]],) for x in peaks]  # add a third element to tuples (the score)
        idx = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (idx[i],) for i in range(len(idx))]  # add id to tuples

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(util.limb_paf_idx)):
        # PAF for the given limbs (i.e. connection between parts)
        score_mid = paf_avg[:, :, [x - 19 for x in util.limb_paf_idx[k]]]
        # get all peaks for the given parts
        cand_a = all_peaks[util.part_seq[k][0] - 1]  # each element is [x, y, score, unique_id]
        cand_b = all_peaks[util.part_seq[k][1] - 1]  # each element is [x, y, score, unique_id]
        n_a = len(cand_a)
        n_b = len(cand_b)
        # index_a, index_b = util.part_seq[k]
        if n_a != 0 and n_b != 0:
            connection_candidate = []
            for i in range(n_a):
                for j in range(n_b):
                    vec = np.subtract(cand_b[j][:2], cand_a[i][:2])
                    norm = np.linalg.norm(vec)
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(cand_a[i][0], cand_b[j][0], num=mid_num),
                                        np.linspace(cand_a[i][1], cand_b[j][1], num=mid_num)))
                    startend = np.array(startend).round().astype(int)

                    vec_x = np.array([score_mid[se_i[1], se_i[0], 0] for se_i in startend])
                    vec_y = np.array([score_mid[se_i[1], se_i[0], 1] for se_i in startend])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * input_image.shape[0] / norm - 1, 0)  # mean of the points + penalty if segment is too big
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)  # If more than 80% of midpoints have an high score
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            # Will contain [seq_id_a, seq_id_b, score, ith_of_a, jth_of_b]
            connections = np.zeros((0, 5))
            for cc in connection_candidate:
                i, j, s = cc
                if i not in connections[:, 3] and j not in connections[:, 4]:
                    connections = np.vstack([connections, [cand_a[i][3], cand_b[j][3], s, i, j]])
                    if len(connections) >= min(n_a, n_b):
                        break

            connection_all.append(connections)
        else:
            special_k.append(k)
            connection_all.append([])

    # Each subset is a person (or at least should be)
    # The last element of each row is the total number of parts belonging to that person
    # The second last element of each row is the score of the overall configuration
    # Here you basically map each part to the person they belong to
    subsets = np.empty((0, 20))
    # Put together all rows of all_peaks and add the part type
    candidates = np.array([item[:-1] + (i,) for i, sublist in enumerate(all_peaks) for item in sublist])

    del all_peaks

    for k in range(len(util.limb_paf_idx)):
        if k in special_k:
            continue

        connections = connection_all[k]

        part_as = connections[:, 0]
        part_bs = connections[:, 1]
        index_a, index_b = util.part_seq[k] - 1

        # For each connection of the limb
        for i in range(connections.shape[0]):
            found = 0
            subset_idx = [-1, -1]
            for j in range(subsets.shape[0]):
                if subsets[j][index_a] == part_as[i] or subsets[j][index_b] == part_bs[i]:
                    subset_idx[found] = j
                    found += 1

            if found == 1:
                j = subset_idx[0]
                if subsets[j][index_b] != part_bs[i]:
                    subsets[j][index_b] = part_bs[i]
                    subsets[j][-1] += 1
                    subsets[j][-2] += candidates[part_bs[i].astype(int), 2] + connections[i][2]
            elif found == 2:  # if found 2 and disjoint, merge them
                j1, j2 = subset_idx
                membership = ((subsets[j1] >= 0).astype(int) + (subsets[j2] >= 0).astype(int))[:-2]
                if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                    subsets[j1][:-2] += (subsets[j2][:-2] + 1)
                    subsets[j1][-2:] += subsets[j2][-2:]
                    subsets[j1][-2] += connections[i][2]
                    subsets = np.delete(subsets, j2, 0)
                else:  # as like found == 1
                    subsets[j1][index_b] = part_bs[i]
                    subsets[j1][-1] += 1
                    subsets[j1][-2] += candidates[part_bs[i].astype(int), 2] + connections[i][2]

            # if find no partA in the subset, create a new subset
            elif not found and k < 17:
                row = np.full(20, fill_value=-1)
                row[index_a] = part_as[i]  # sequential unique id of part a across persons
                row[index_b] = part_bs[i]  # sequential unique id of part b across persons
                row[-1] = 2
                row[-2] = sum(candidates[connections[i, :2].astype(int), 2]) + connections[i][2]
                subsets = np.vstack([subsets, row])

    # delete some rows of subset which has few parts occur
    delete_idx = []
    for i in range(len(subsets)):
        # Too few parts or low average score
        if subsets[i][-1] < 4 or subsets[i][-2] / subsets[i][-1] < 0.4:
            delete_idx.append(i)
    subsets = np.delete(subsets, delete_idx, axis=0)

    return subsets, candidates
