import os
import warnings
import itertools
from copy import copy
import multiprocessing as mul
import numpy as np
import pandas as pd
from pynverse import inversefunc
import cv2
from .post_processing_functions import (
    dilate_contour_and_calculate_characteristics,
)
from .post_processing_enums import *


class StatisticalAnalyzer:
    def __init__(self, pixel_ratio=0.44, grain_size_group=[5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300],
                 redilate_iter=2, kernel_shape='ellipse', kernel_size=3,
                 multiprocessing=False):
        self.pixel_ratio = pixel_ratio
        self.grain_size_group = grain_size_group
        self.redilate_iter = redilate_iter
        kernel_shape = getattr(KernelShape, kernel_shape.capitalize(), KernelShape.Ellipse)
        self.kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
        self.multiprocessing = multiprocessing

    def __call__(self, contour_dict, image_size):
        contour_dict = self.calculate_segmentation_characteristics(contour_dict)
        statistic_dict = self.calculate_statistics(contour_dict, image_size)
        grain_dist_dict, grain_dist_parameters = self.calculate_grain_distribution(
            contour_dict['length'], contour_dict['area'], contour_dict['volume']
        )
        return contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters

    def calculate_segmentation_characteristics(self, contour_dict):
        contour_infos = []
        for contour_index in contour_dict.index:
            points = contour_dict['points'].loc[contour_index]
            center = contour_dict['center'].loc[contour_index]
            contour_infos.append((points, center, contour_index, self.redilate_iter, self.pixel_ratio, self.kernel))

        if self.multiprocessing:
            pool = mul.Pool(processes=os.cpu_count())
            result_async = pool.starmap_async(
                dilate_contour_and_calculate_characteristics,
                contour_infos)
            result_async.wait()
            characteristics = result_async.get()
            pool.close()
            pool.join()
        else:
            characteristics = []
            for contour_info in contour_infos:
                characteristics.append(dilate_contour_and_calculate_characteristics(*contour_info))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)

            contour_dict = pd.concat([contour_dict, pd.DataFrame(columns=[
                'area', 'length', 'perimeter', 'sphericity', 'slenderness', 'volume'
            ])], axis=1)
            for chara in characteristics:
                contour_dict['points'].loc[chara[0]] = chara[1]
                contour_dict['bbox'].loc[chara[0]] = chara[2]
                contour_dict['center'].loc[chara[0]] = chara[3]
                contour_dict['area'].loc[chara[0]] = chara[4]
                contour_dict['length'].loc[chara[0]] = chara[5]
                contour_dict['perimeter'].loc[chara[0]] = chara[6]
                contour_dict['sphericity'].loc[chara[0]] = chara[7]
                contour_dict['slenderness'].loc[chara[0]] = chara[8]
                contour_dict['volume'].loc[chara[0]] = chara[9]

        return contour_dict

    def calculate_statistics(self, contour_dict, image_size):
        def quantity_statistic(quantity_series):
            return {
                'max': quantity_series.max(),
                'min': quantity_series.min(),
                'avg': quantity_series.sum() / quantity_series.count(),
                'median': quantity_series.median(),
                'up_quartile': quantity_series.quantile(0.75),
                'lo_quartile': quantity_series.quantile(0.25),
                'total': quantity_series.sum(),
            }

        statistic_dict = {}
        statistic_dict['image_width'] = image_size[1]
        statistic_dict['image_height'] = image_size[0]
        statistic_dict['muck_count'] = len(contour_dict)
        statistic_dict['area'] = quantity_statistic(contour_dict['area'])
        statistic_dict['length'] = quantity_statistic(contour_dict['length'])
        statistic_dict['perimeter'] = quantity_statistic(contour_dict['perimeter'])
        statistic_dict['sphericity'] = quantity_statistic(contour_dict['sphericity'])
        statistic_dict['slenderness'] = quantity_statistic(contour_dict['slenderness'])
        statistic_dict['volume'] = quantity_statistic(contour_dict['volume'])

        return statistic_dict

    def calculate_grain_distribution(self, length_series, area_series, volume_series):
        def calculate_grain_dist_parameters(cumulative_percentage):
            assert len(cumulative_percentage) == len(self.grain_size_group), 'The length of input cumulative percentage must be equal to the length of grain size group'
            _grain_size_group = [0, ]
            _cumulative_percentage = [0.0, ]
            for i in range(len(self.grain_size_group)):
                if cumulative_percentage[i] < (cumulative_percentage[i - 1] if i > 0 else 0.0):
                    raise ValueError('The cumulative percentage must be monotonically increasing')
                elif cumulative_percentage[i] == (cumulative_percentage[i - 1] if i > 0 else 0.0):
                    _cumulative_percentage.append(cumulative_percentage[i] + 1e-6)
                    _grain_size_group.append(self.grain_size_group[i])
                else:
                    _cumulative_percentage.append(cumulative_percentage[i])
                    _grain_size_group.append(self.grain_size_group[i])

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                grain_dist_func = lambda d: np.interp(d, _grain_size_group, _cumulative_percentage)
                grain_dist_inv = inversefunc(grain_dist_func)
                d10, d30, d50, d60 = grain_dist_inv([0.1, 0.3, 0.5, 0.6])
                cu = d60 / d10
                cc = d30 ** 2 / (d10 * d60)
                ci = sum([1 - i for i in list(grain_dist_dict['cum_area_percentage'].values)])

            return {'Cu': cu, 'Cc': cc, 'CI': ci, 'D10': d10, 'D30': d30, 'D50': d50, 'D60': d60}

        _index = copy(self.grain_size_group)
        _index.append(10000)
        grain_dist_dict = pd.DataFrame(columns=['area_sum', 'volume_sum'], index=_index)
        for i in range(len(grain_dist_dict)):
            if i == 0:
                size_lb = 0
            else:
                size_lb = grain_dist_dict.index[i - 1]
            size_ub = grain_dist_dict.index[i]
            grain_dist_dict['area_sum'].iloc[i] = area_series.loc[
                length_series >= size_lb].loc[length_series < size_ub].sum()
            grain_dist_dict['volume_sum'].iloc[i] = volume_series.loc[
                length_series >= size_lb].loc[length_series < size_ub].sum()

        grain_dist_dict['cum_area'] = list(itertools.accumulate(grain_dist_dict['area_sum'].values))
        grain_dist_dict['cum_volume'] = list(itertools.accumulate(grain_dist_dict['volume_sum'].values))
        total_area = grain_dist_dict['cum_area'].iloc[-1]
        total_volume = grain_dist_dict['cum_volume'].iloc[-1]
        grain_dist_dict['cum_area_percentage'] = grain_dist_dict['cum_area'].apply(lambda x: x / total_area)
        grain_dist_dict['cum_volume_percentage'] = grain_dist_dict['cum_volume'].apply(lambda x: x / total_volume)

        grain_dist_parameters = {
            'area': calculate_grain_dist_parameters(list(grain_dist_dict['cum_area_percentage'].values)[:-1]),
            'volume': calculate_grain_dist_parameters(list(grain_dist_dict['cum_volume_percentage'].values)[:-1])
        }

        return grain_dist_dict, grain_dist_parameters

    def get_placeholders(self, contour_dict, image_size):
        contour_dict = pd.concat([contour_dict, pd.DataFrame(columns=[
            'area', 'length', 'perimeter', 'sphericity', 'slenderness', 'volume'
        ])], axis=1)
        null_quantity_statistic = {
            'max': 0,
            'min': 0,
            'avg': 0,
            'median': 0,
            'up_quartile': 0,
            'lo_quartile': 0,
            'total': 0,
        }
        statistic_dict = {
            'image_width': image_size[1],
            'image_height': image_size[0],
            'muck_count': 0,
            'area': null_quantity_statistic,
            'length': null_quantity_statistic,
            'perimeter': null_quantity_statistic,
            'sphericity': null_quantity_statistic,
            'slenderness': null_quantity_statistic,
            'volume':null_quantity_statistic,
        }
        _index = copy(self.grain_size_group)
        _index.append(10000)
        grain_dist_dict = pd.DataFrame(
            columns=['area_sum', 'volume_sum', 'cum_area', 'cum_volume', 'cum_area_percentage', 'cum_volume_percentage'],
            index=_index
        )
        null_grain_dist_parameter = {'Cu': 0.0, 'Cc': 0.0, 'CI': 0.0, 'D10': 0.0, 'D30': 0.0, 'D50': 0.0, 'D60': 0.0}
        grain_dist_parameters = {
            'area': null_grain_dist_parameter,
            'volume': null_grain_dist_parameter
        }

        return contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters

# EOF