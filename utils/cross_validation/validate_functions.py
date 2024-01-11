import os
import json
import warnings
import math
import itertools
import multiprocessing as mul
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from pynverse import inversefunc
from lib.cv_utils import bbox_overlap, contour_overlap, global2part_contour, enlarge_bbox


PIXEL_RATIO = 0.44


def anno_dict2pd(anno_dict):
    return {
        'info': anno_dict['info'],
        'categories': pd.DataFrame(anno_dict['categories']).ffill(),
        'images': pd.DataFrame(anno_dict['images']).ffill(),
        'annotations': pd.DataFrame(anno_dict['annotations']).ffill(),
    }


def regularize_annotation(annotation):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
        for idx in annotation.index:
            _ = annotation['segmentation'].loc[idx]
            _ = np.frompyfunc(round, 1, 1)(_).reshape(-1, 2).astype(int)
            annotation['segmentation'].loc[idx] = _
            _ = annotation['bbox'].loc[idx]
            _[0] = math.floor(_[0])
            _[1] = math.floor(_[1])
            _[2] = math.ceil(_[2])
            _[3] = math.ceil(_[3])
            annotation['bbox'].loc[idx] = _

    return annotation


def calculate_contour_properties(contour, area, idx):
    area = area * (PIXEL_RATIO ** 2)
    perimeter = cv2.arcLength(contour, True) * PIXEL_RATIO
    sphericity = math.sqrt(4 * math.pi * area) / perimeter
    rect = cv2.minAreaRect(contour)
    length = max(rect[1][0], rect[1][1]) * PIXEL_RATIO
    slenderness = min(rect[1][0], rect[1][1]) * PIXEL_RATIO / length
    ellip = cv2.fitEllipse(contour) if len(contour) >= 5 else cv2.minAreaRect(contour)
    volume = math.pi * ellip[1][1] * ellip[1][0] ** 2 * PIXEL_RATIO ** 3 / 6
    return idx, area, length, perimeter, sphericity, slenderness, volume


def extend_image_annotation(image_annotation):
    contour_infos = []
    for idx in image_annotation.index:
        contour_infos.append((image_annotation['segmentation'].loc[idx], image_annotation['area'].loc[idx], idx))
    pool = mul.Pool(processes=os.cpu_count())
    result_async = pool.starmap_async(calculate_contour_properties, contour_infos)
    pool.close()
    pool.join()
    contour_properties = result_async.get()
    image_annotation = pd.concat((image_annotation, pd.DataFrame(columns=[
        'length', 'perimeter', 'sphericity', 'slenderness', 'volume'
    ])), axis=1)
    for c_prop in contour_properties:
        image_annotation['area'].loc[c_prop[0]] = c_prop[1]
        image_annotation['length'].loc[c_prop[0]] = c_prop[2]
        image_annotation['perimeter'].loc[c_prop[0]] = c_prop[3]
        image_annotation['sphericity'].loc[c_prop[0]] = c_prop[4]
        image_annotation['slenderness'].loc[c_prop[0]] = c_prop[5]
        image_annotation['volume'].loc[c_prop[0]] = c_prop[6]

    return image_annotation


def calculate_grain_distribution(image_annotation):
    def calculate_grain_dist_parameters(cumulative_percentage, length_list):
        _cumulative_percentage = []
        for i in range(len(cumulative_percentage)):
            if cumulative_percentage[i] < (cumulative_percentage[i - 1] if i > 0 else 0.0):
                raise ValueError('The cumulative percentage must be monotonically increasing')
            elif cumulative_percentage[i] == (cumulative_percentage[i - 1] if i > 0 else 0.1):
                _cumulative_percentage.append(cumulative_percentage[i] + 1e-6)
            else:
                _cumulative_percentage.append(cumulative_percentage[i])

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            grain_dist_func = lambda d: np.interp(d, length_list, _cumulative_percentage)
            grain_dist_inv = inversefunc(grain_dist_func)
            d10, d30, d50, d60 = grain_dist_inv([0.1, 0.3, 0.5, 0.6])
            cu = d60 / d10
            cc = d30 ** 2 / (d10 * d60)
            ci = sum([1 - i for i in list(grain_dist_dict['cum_area_percentage'].values)])
        return {'Cu': cu, 'Cc': cc, 'CI': ci, 'D10': d10, 'D30': d30, 'D50': d50, 'D60': d60}

    max_length = image_annotation['length'].max()
    length_list = np.linspace(0, max_length, 50)
    grain_dist_dict = pd.DataFrame(columns=['area_sum', 'volume_sum'], index=length_list)
    for i in range(len(grain_dist_dict)):
        if i == 0:
            grain_dist_dict['area_sum'] = 0
            grain_dist_dict['volume_sum'] = 0
        else:
            size_lb = grain_dist_dict.index[i - 1]
            size_ub = grain_dist_dict.index[i]
            grain_dist_dict['area_sum'].iloc[i] = image_annotation['area'].loc[
                image_annotation['length'] > size_lb
                ].loc[image_annotation['length'] <= size_ub].sum()
            grain_dist_dict['volume_sum'].iloc[i] = image_annotation['volume'].loc[
                image_annotation['length'] > size_lb
                ].loc[image_annotation['length'] <= size_ub].sum()

    grain_dist_dict['cum_area'] = list(itertools.accumulate(grain_dist_dict['area_sum'].values))
    grain_dist_dict['cum_volume'] = list(itertools.accumulate(grain_dist_dict['volume_sum'].values))
    total_area = grain_dist_dict['cum_area'].iloc[-1]
    total_volume = grain_dist_dict['cum_volume'].iloc[-1]
    grain_dist_dict['cum_area_percentage'] = grain_dist_dict['cum_area'].apply(lambda x: x / total_area)
    grain_dist_dict['cum_volume_percentage'] = grain_dist_dict['cum_volume'].apply(lambda x: x / total_volume)

    grain_dist_parameters = {
        'area': calculate_grain_dist_parameters(list(grain_dist_dict['cum_area_percentage'].values), length_list),
        'volume': calculate_grain_dist_parameters(list(grain_dist_dict['cum_volume_percentage'].values), length_list)
    }

    return grain_dist_dict, grain_dist_parameters


def calculate_grain_distribution_wrapper(image_name, source_annotation, target_annotation):
    source_grain_dist_data, source_grain_dist_param = calculate_grain_distribution(source_annotation)
    target_grain_dist_data, target_grain_dist_param = calculate_grain_distribution(target_annotation)
    return image_name, source_grain_dist_data, source_grain_dist_param, target_grain_dist_data, target_grain_dist_param


def match_image_annotation(image_name, source_annotation, target_annotation):
    num_source_anno = len(source_annotation)
    num_target_anno = len(target_annotation)
    source_orphan_annotation = pd.DataFrame(columns=source_annotation.columns)
    target_orphan_annotation = pd.DataFrame(columns=target_annotation.columns)

    source_unmatched = []
    target_unmatched = []
    bbox_cross_matr = np.zeros((num_source_anno, num_target_anno))
    for i in range(num_source_anno):
        source_bbox = source_annotation['bbox'].iloc[i]
        for j in range(num_target_anno):
            target_bbox = target_annotation['bbox'].iloc[j]
            bbox_cross_matr[i, j] = bbox_overlap(source_bbox, target_bbox)
    for i in range(num_source_anno):
        if np.max(bbox_cross_matr[i, :]) == 0:
            source_unmatched.append(i)
    for j in range(num_target_anno):
        if np.max(bbox_cross_matr[:, j]) == 0:
            target_unmatched.append(j)
    for idx in source_unmatched:
        source_orphan_annotation.loc[len(source_orphan_annotation)] = source_annotation.loc[idx]
    source_annotation.drop(index=source_unmatched, inplace=True)
    source_annotation.reset_index(drop=True, inplace=True)
    bbox_cross_matr = np.delete(bbox_cross_matr, source_unmatched, axis=0)
    for idx in target_unmatched:
        target_orphan_annotation.loc[len(target_orphan_annotation)] = target_annotation.loc[idx]
    target_annotation.drop(index=target_unmatched, inplace=True)
    target_annotation.reset_index(drop=True, inplace=True)
    bbox_cross_matr = np.delete(bbox_cross_matr, target_unmatched, axis=1)

    num_source_anno = len(source_annotation)
    num_target_anno = len(target_annotation)
    source_unmatched = []
    target_unmatched = []
    contour_cross_matr = np.zeros((num_source_anno, num_target_anno))
    for i in range(num_source_anno):
        source_bbox = source_annotation['bbox'].iloc[i]
        contour1 = source_annotation['segmentation'].iloc[i]
        for j in range(num_target_anno):
            if bbox_cross_matr[i, j] > 0:
                target_bbox = target_annotation['bbox'].iloc[j]
                contour2 = target_annotation['segmentation'].iloc[j]
                contour_cross_matr[i, j] = contour_overlap(contour1, source_bbox, contour2, target_bbox)
    for i in range(num_source_anno):
        if np.max(bbox_cross_matr[i, :]) == 0:
            source_unmatched.append(i)
    for j in range(num_target_anno):
        if np.max(bbox_cross_matr[:, j]) == 0:
            target_unmatched.append(j)
    for idx in source_unmatched:
        source_orphan_annotation.loc[len(source_orphan_annotation)] = source_annotation.loc[idx]
    source_annotation.drop(index=source_unmatched, inplace=True)
    source_annotation.reset_index(drop=True, inplace=True)
    contour_cross_matr = np.delete(contour_cross_matr, source_unmatched, axis=0)
    for idx in target_unmatched:
        target_orphan_annotation.loc[len(target_orphan_annotation)] = target_annotation.loc[idx]
    target_annotation.drop(index=target_unmatched, inplace=True)
    target_annotation.reset_index(drop=True, inplace=True)
    contour_cross_matr = np.delete(contour_cross_matr, target_unmatched, axis=1)

    source_matched = []
    target_matched = []
    source_unmatched = []
    target_unmatched = []
    match_source2target = np.argmax(contour_cross_matr, axis=1)
    match_target2source = np.argmax(contour_cross_matr, axis=0)
    for i in range(len(match_source2target)):
        if match_target2source[match_source2target[i]] == i:
            source_matched.append(i)
            target_matched.append(match_source2target[i])
        else:
            source_unmatched.append(i)
    for j in range(len(match_target2source)):
        if j not in target_matched:
            target_unmatched.append(j)
    matched = list(zip(source_matched, target_matched))
    for idx in source_unmatched:
        source_orphan_annotation.loc[len(source_orphan_annotation)] = source_annotation.loc[idx]
    source_annotation.drop(index=source_unmatched, inplace=True)
    for idx in target_unmatched:
        target_orphan_annotation.loc[len(target_orphan_annotation)] = target_annotation.loc[idx]
    target_annotation.drop(index=target_unmatched, inplace=True)

    merged_annotation = pd.DataFrame(columns=[
        'source_segmentation', 'source_bbox', 'source_area', 'source_length',
        'source_perimeter', 'source_sphericity', 'source_slenderness', 'source_volume',
        'target_segmentation', 'target_bbox', 'target_area', 'target_length',
        'target_perimeter', 'target_sphericity', 'target_slenderness', 'target_volume',
    ])
    for source_idx, target_idx in matched:
        merged_annotation.loc[len(merged_annotation)] = [
            source_annotation['segmentation'].loc[source_idx],
            source_annotation['bbox'].loc[source_idx],
            source_annotation['area'].loc[source_idx],
            source_annotation['length'].loc[source_idx],
            source_annotation['perimeter'].loc[source_idx],
            source_annotation['sphericity'].loc[source_idx],
            source_annotation['slenderness'].loc[source_idx],
            source_annotation['volume'].loc[source_idx],
            target_annotation['segmentation'].loc[target_idx],
            target_annotation['bbox'].loc[target_idx],
            target_annotation['area'].loc[target_idx],
            target_annotation['length'].loc[target_idx],
            target_annotation['perimeter'].loc[target_idx],
            target_annotation['sphericity'].loc[target_idx],
            target_annotation['slenderness'].loc[target_idx],
            target_annotation['volume'].loc[target_idx],
        ]

    return image_name, merged_annotation, source_orphan_annotation, target_orphan_annotation


def calculate_Hausdorff_distance(merged_annotation):
    extractor = cv2.createHausdorffDistanceExtractor(cv2.NORM_L2)
    merged_annotation['60-HD'] = merged_annotation[['source_segmentation', 'target_segmentation']].apply(
        lambda row: extractor.computeDistance(row['source_segmentation'].reshape(-1, 1, 2), row['target_segmentation'].reshape(-1, 1, 2)) * PIXEL_RATIO,
        axis=1
    )
    extractor.setRankProportion(0.95)
    merged_annotation['95-HD'] = merged_annotation[['source_segmentation', 'target_segmentation']].apply(
        lambda row: extractor.computeDistance(row['source_segmentation'].reshape(-1, 1, 2), row['target_segmentation'].reshape(-1, 1, 2)) * PIXEL_RATIO,
        axis=1
    )
    return merged_annotation


def calculate_orphan_penalty(orphan_annotation):
    def calculate_penalty(segmentation, bbox):
        enlarged_bbox = enlarge_bbox(bbox, 1)
        part_mask = np.zeros((enlarged_bbox[3], enlarged_bbox[2]), dtype=np.uint8)
        part_contour = global2part_contour([segmentation.reshape(-1, 1, 2)], enlarged_bbox)
        part_mask = cv2.fillPoly(part_mask, part_contour, 1)
        part_mask = cv2.distanceTransform(part_mask, cv2.DIST_L2, 5)
        return np.max(part_mask) * PIXEL_RATIO

    if len(orphan_annotation) > 0:
        orphan_annotation['penalty'] = orphan_annotation[['segmentation', 'bbox']].apply(
            lambda row: calculate_penalty(row['segmentation'], row['bbox']),
            axis=1
        )
    else:
        orphan_annotation = pd.concat([orphan_annotation, pd.DataFrame(columns=['penalty'])], axis=1)
    return orphan_annotation


def calculate_orphan_penalty_wrapper(source_orphan_annotation, target_orphan_annotation):
    return calculate_orphan_penalty(source_orphan_annotation), calculate_orphan_penalty(target_orphan_annotation)


def match_image_annotation_wrapper(image_name, source_annotation, target_annotation):
    image_name, merged_annotation, source_orphan_annotation, target_orphan_annotation = match_image_annotation(
        image_name, source_annotation, target_annotation)
    merged_annotation = calculate_Hausdorff_distance(merged_annotation)
    source_orphan_annotation, target_orphan_annotation = calculate_orphan_penalty_wrapper(
        source_orphan_annotation, target_orphan_annotation
    )
    return image_name, merged_annotation, source_orphan_annotation, target_orphan_annotation


def cross_validate(source_annotation_path, target_annotation_path):
    with open(source_annotation_path, 'r') as f:
        source_anno_json = json.load(f)
    with open(target_annotation_path, 'r') as f:
        target_anno_json = json.load(f)
    source_anno_pd = anno_dict2pd(source_anno_json)
    target_anno_pd = anno_dict2pd(target_anno_json)
    source_anno_pd['annotations'] = regularize_annotation(source_anno_pd['annotations'])
    target_anno_pd['annotations'] = regularize_annotation(target_anno_pd['annotations'])

    img_dict = pd.concat((source_anno_pd['images'][['id', 'file_name']], pd.DataFrame(columns=[
        'matched_id'
    ])), axis=1)
    drop_idx = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
        for idx in img_dict.index:
            record = target_anno_pd['images']['id'].loc[
                target_anno_pd['images']['file_name'] == img_dict['file_name'].loc[idx]
            ]
            if len(record) == 1:
                img_dict['matched_id'].loc[idx] = record.item()
            else:
                drop_idx.append(idx)
        for idx in drop_idx:
            img_dict.drop(index=idx, inplace=True)

        img_infos = []
        for idx in img_dict.index:
            source_img_annotation = source_anno_pd['annotations'][['segmentation', 'bbox', 'area']].loc[
                source_anno_pd['annotations']['image_id'] == img_dict['id'].loc[idx]
            ]
            target_img_annotation = target_anno_pd['annotations'][['segmentation', 'bbox', 'area']].loc[
                target_anno_pd['annotations']['image_id'] == img_dict['matched_id'].loc[idx]
                ]
            source_img_annotation.reset_index(drop=True, inplace=True)
            target_img_annotation.reset_index(drop=True, inplace=True)
            source_img_annotation = extend_image_annotation(source_img_annotation)
            target_img_annotation = extend_image_annotation(target_img_annotation)
            img_infos.append((Path(img_dict['file_name'].loc[idx]).stem, source_img_annotation, target_img_annotation))

        pool = mul.Pool(processes=os.cpu_count())
        result_async = pool.starmap_async(calculate_grain_distribution_wrapper, img_infos)
        pool.close()
        pool.join()
        grain_dist_result = result_async.get()

        grain_dist_dict = {}
        for img_name, source_gd_data, source_gd_param, target_gd_data, target_gd_param in grain_dist_result:
            grain_dist_dict[img_name] = {
                'source_grain_dist_data': source_gd_data,
                'source_grain_dist_param': source_gd_param,
                'target_grain_dist_data': target_gd_data,
                'target_grain_dist_param': target_gd_param,
            }

        merged_annotations = []
        for img_info in img_infos:
            merged_annotations.append(match_image_annotation_wrapper(*img_info))

        validation_dict = {}
        for img_name, m_anno, source_o_anno, target_o_anno in merged_annotations:
            validation_dict[img_name] = {
                'merged_annotation': m_anno,
                'source_orphan_annotation': source_o_anno,
                'target_orphan_annotation': target_o_anno,
            }

    return grain_dist_dict, validation_dict

# EOF