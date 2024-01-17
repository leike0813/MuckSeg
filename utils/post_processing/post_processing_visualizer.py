import os
import math
import multiprocessing as mul
import matplotlib as mpl
import numpy as np
import cv2
import torch
from .post_processing_functions import (
    draw_single_mask,
    draw_single_contour,
    draw_legend,
    draw_boxplot,
    draw_vhistogram,
    draw_grain_dist_curve,
    MPL_RC,
)
from .post_processing_enums import *
from lib.cv_utils import (
    torch2cv,
    cv2torch,
    make_background_transparent,
    global2part_contour,
    id2hsv_ufunc,
    merge_image,
)


class Visualizer:
    AVAILABLE_COLORMAPS = [
        'turbo',
        'jet',
        'viridis',
        'hsv',
        'plasma',
    ]
    colormap_cv = cv2.COLORMAP_TURBO
    colormap_mat = mpl.cm.turbo

    AVAILABLE_VISUALIZATION_MODES = {
        'logarea': 'LogArea',
        'log_area': 'LogArea',
        'area': 'Area',
        'logvolume': 'LogVolume',
        'log_volume': 'LogVolume',
        'volume': 'Volume',
        'loglength': 'LogLength',
        'log_length': 'LogLength',
        'length': 'Length',
        'id': 'ID',
    }

    AVAILABLE_RESULT_FORMAT = [
        'png',
        'jpg',
        'gif',
        'bmp',
    ]

    DEFAULT_LEGEND_AND_GRAPHS_LAYOUT = {
        'GLOBAL_ORIGIN': (0, 0),
        'LEGEND_SCALE': 3,
        'LEGEND_WIDTH': 250,
        'LEGEND_ANCHOR': (30, 30),
        'GRAPH_SIZE': (800, 320),
        'GRAPH_SPACING': 100,
        'GRAIN_DIST_GRAPH_WIDTH': 350,
    }


    def __init__(self, visualization_mode='log_area', visualization_alpha=0.5, colormap='turbo',
                 multiprocessing=False, draw_contours=False, draw_legend=True, draw_graindist=True,
                 draw_statistics=True, independent_graphs=False, landscape_mode=False, additional_mpl_rc={},
                 legend_and_graphs=None, result_format='png'):
        self.visualization_mode = getattr(
            VisualizationMode,
            self.AVAILABLE_VISUALIZATION_MODES.get(visualization_mode.lower(), 'LogArea'),
        )
        self.visualization_alpha = visualization_alpha
        self.multiprocessing = multiprocessing
        self.draw_contours = draw_contours
        self.draw_legend = draw_legend
        self.draw_graindist = draw_graindist
        self.draw_statistics = draw_statistics
        self.independent_graphs = independent_graphs
        self.landscape_mode = landscape_mode
        if colormap is not None and colormap in self.AVAILABLE_COLORMAPS:
            self.colormap_cv = getattr(cv2, 'COLORMAP_' + colormap.upper(), self.colormap_cv)
            self.colormap_mat = getattr(mpl.cm, colormap, self.colormap_mat)
        self.mpl_rc = MPL_RC
        self.mpl_rc.update(additional_mpl_rc)
        mpl.rcParams.update(self.mpl_rc)
        if legend_and_graphs is None:
            self.legend_and_graphs = self.DEFAULT_LEGEND_AND_GRAPHS_LAYOUT
        else:
            self.legend_and_graphs = legend_and_graphs
        self.result_format = result_format.lower() if result_format.lower() in self.AVAILABLE_RESULT_FORMAT else 'png'

    def __call__(self, orig_image, contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters):
        result_img = self.draw_result(orig_image, contour_dict)
        result_img_dict = self.draw_legend_and_graphs(result_img, contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters)
        for k, v in result_img_dict.items():
            result_img_dict[k] = cv2torch(v).unsqueeze(0)
        return result_img_dict


    def draw_result(self, orig_image, contour_dict):
        if isinstance(orig_image, torch.Tensor):
            orig_image = torch2cv(orig_image)

        contour_infos = []
        if self.visualization_mode == VisualizationMode.ID:
            for contour_index in contour_dict.index:
                bbox = contour_dict['bbox'].loc[contour_index]
                part_contour = global2part_contour([contour_dict['points'].loc[contour_index]], bbox)[0]
                contour_infos.append((
                    part_contour,
                    bbox,
                    contour_index + 1,
                ))
        elif self.visualization_mode in [VisualizationMode.Area, VisualizationMode.LogArea,
                                         VisualizationMode.Volume, VisualizationMode.LogVolume,
                                         VisualizationMode.Length, VisualizationMode.LogLength]:
            quantity = VisualizationMode.get_quantity_name(self.visualization_mode)
            max_quantity = contour_dict[quantity].max()
            min_quantity = contour_dict[quantity].min()
            for contour_index in contour_dict.index:
                bbox = contour_dict['bbox'].loc[contour_index]
                part_contour = global2part_contour([contour_dict['points'].loc[contour_index]], bbox)[0]
                if self.visualization_mode in [VisualizationMode.Area, VisualizationMode.Volume, VisualizationMode.Length]:
                    value = int(255 * (contour_dict[quantity].loc[contour_index] - min_quantity)
                                / (max_quantity - min_quantity))
                elif self.visualization_mode in [VisualizationMode.LogArea, VisualizationMode.LogVolume, VisualizationMode.LogLength]:
                    # value = int(255 * math.log10(
                    #     contour_dict[quantity].loc[contour_index] - min_quantity + 1
                    # ) / math.log10(max_quantity - min_quantity + 1))
                    value = int(255 * (math.log10(contour_dict[quantity].loc[contour_index]) - math.log10(min_quantity))
                                / (math.log10(max_quantity) - math.log10(min_quantity)))

                contour_infos.append((
                    part_contour,
                    bbox,
                    value,
                ))

        if self.multiprocessing:
            pool = mul.Pool(processes=os.cpu_count())
            result_async = pool.starmap_async(
                draw_single_mask,
                contour_infos)
            result_async.wait()
            temp_masks = result_async.get()
            pool.close()
            pool.join()
            if self.draw_contours:
                pool = mul.Pool(processes=os.cpu_count())
                result_async = pool.starmap_async(
                    draw_single_contour,
                    [contour_info for contour_info in contour_infos])
                result_async.wait()
                temp_contours = result_async.get()
                pool.close()
                pool.join()
        else:
            temp_masks = []
            for contour_info in contour_infos:
                temp_masks.append(draw_single_mask(*contour_info))
            if self.draw_contours:
                temp_contours = []
                for contour_info in contour_infos:
                    temp_contours.append(draw_single_contour(*contour_info))

        result_mask = np.zeros((orig_image.shape[0], orig_image.shape[1]), dtype=int)
        for mask in temp_masks:
            merge_image(result_mask, mask[0], mask[1])
        if self.draw_contours:
            result_contour = np.zeros((orig_image.shape[0], orig_image.shape[1]), dtype=int)
            for contour in temp_contours:
                merge_image(result_contour, contour[0], contour[1])

        if self.visualization_mode == VisualizationMode.ID:
            result_mask_image = np.stack(id2hsv_ufunc(result_mask), axis=2).astype(np.uint8)
            if self.draw_contours:
                result_contour_image = np.stack(id2hsv_ufunc(result_contour), axis=2).astype(np.uint8)
        elif self.visualization_mode in [VisualizationMode.Area, VisualizationMode.LogArea,
                                         VisualizationMode.Volume, VisualizationMode.LogVolume,
                                         VisualizationMode.Length, VisualizationMode.LogLength]:
            background_mask = np.stack([(result_mask == 0).astype(int) * 255 for i in range(3)], axis=2)
            result_mask_image = np.clip(
                cv2.applyColorMap(result_mask.astype(np.uint8), self.colormap_cv) - background_mask, 0, 255).astype(
                np.uint8)
            if self.draw_contours:
                background_contour = np.stack([(result_contour == 0).astype(int) * 255 for i in range(3)], axis=2)
                result_contour_image = np.clip(
                    cv2.applyColorMap(result_contour.astype(np.uint8), self.colormap_cv) - background_contour, 0, 255).astype(
                    np.uint8)

        result_image = cv2.addWeighted(
            cv2.cvtColor(orig_image, cv2.COLOR_GRAY2BGR),
            1,
            cv2.cvtColor(
                result_mask_image,
                cv2.COLOR_HSV2BGR
            ) if self.visualization_mode == VisualizationMode.ID else result_mask_image,
            self.visualization_alpha,
            0
        )
        if self.draw_contours:
            _, mask_array = cv2.threshold(result_contour_image, 1, 255, cv2.THRESH_BINARY)
            mask_array = mask_array == 255
            result_image[mask_array] = result_contour_image[mask_array]

        return result_image


    def draw_legend_and_graphs(self, image, contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters):
        if self.visualization_mode == VisualizationMode.ID:
            return image
        # -----------parameters------------------
        global_origin = self.legend_and_graphs['GLOBAL_ORIGIN']
        legend_scale = self.legend_and_graphs['LEGEND_SCALE']
        legend_width = self.legend_and_graphs['LEGEND_WIDTH']
        legend_size = (256 * legend_scale + 150, legend_width)
        legend_anchor = self.legend_and_graphs['LEGEND_ANCHOR']

        graph_size = self.legend_and_graphs['GRAPH_SIZE']
        graph_anchor_left = (legend_anchor[0] + legend_size[0] + 150, legend_anchor[1])
        graph_anchor_right = (
            legend_anchor[0] + legend_size[0] + 150,
            image.shape[1] - legend_anchor[1] - graph_size[1]
        )
        graph_spacing = self.legend_and_graphs['GRAPH_SPACING']
        grain_dist_graph_width = self.legend_and_graphs['GRAIN_DIST_GRAPH_WIDTH']
        grain_dist_graph_size = (legend_size[0], grain_dist_graph_width)
        grain_dist_graph_anchor = (legend_anchor[0], image.shape[1] - legend_anchor[1] - grain_dist_graph_width)
        labelparam = {'only_base': True if self.landscape_mode else False,
                      'minor_thresh': (3, 1) if self.landscape_mode else (5, 3)}
        # -----------parameters------------------
        quantity = VisualizationMode.get_quantity_name(self.visualization_mode)
        mode = VisualizationMode.get_mode(self.visualization_mode)

        background = np.zeros_like(image)
        if self.independent_graphs:
            legend_background = np.zeros((legend_size[0], legend_size[1], 3), dtype=np.uint8)
            area_background = np.zeros((graph_size[0], graph_size[1], 3), dtype=np.uint8)
            volume_background = np.zeros((graph_size[0], graph_size[1], 3), dtype=np.uint8)
            length_background = np.zeros((graph_size[0], graph_size[1], 3), dtype=np.uint8)
            perimeter_background = np.zeros((graph_size[0], graph_size[1], 3), dtype=np.uint8)
            sphericity_background = np.zeros((graph_size[0], graph_size[1], 3), dtype=np.uint8)
            slenderness_background = np.zeros((graph_size[0], graph_size[1], 3), dtype=np.uint8)
            grain_dist_graph_background = np.zeros((grain_dist_graph_size[0], grain_dist_graph_size[1], 3), dtype=np.uint8)
            legend_anchor = area_anchor = volume_anchor = length_anchor = perimeter_anchor \
                = sphericity_anchor = slenderness_anchor = grain_dist_graph_anchor = (0, 0)
        else:
            legend_background = area_background = volume_background = length_background = perimeter_background \
                = sphericity_background = slenderness_background = grain_dist_graph_background = background
            legend_anchor = (global_origin[0] + legend_anchor[0], global_origin[1] + legend_anchor[1])
            area_anchor = (global_origin[0] + graph_anchor_left[0], global_origin[1] + graph_anchor_left[1])
            volume_anchor = (global_origin[0] + graph_anchor_left[0] + graph_size[0] + graph_spacing,
                             global_origin[1] + graph_anchor_left[1])
            length_anchor = (global_origin[0] + graph_anchor_left[0] + 2 * (graph_size[0] + graph_spacing),
                             global_origin[1] + graph_anchor_left[1])
            perimeter_anchor = (global_origin[0] + graph_anchor_right[0], global_origin[1] + graph_anchor_right[1])
            sphericity_anchor = (global_origin[0] + graph_anchor_right[0] + graph_size[0] + graph_spacing,
                                 global_origin[1] + graph_anchor_right[1])
            slenderness_anchor = (global_origin[0] + graph_anchor_right[0] + 2 * (graph_size[0] + graph_spacing),
                                  global_origin[1] + graph_anchor_right[1])

        if self.draw_legend:
            draw_legend(legend_background,
                        legend_anchor,
                        legend_size,
                        statistic_dict, quantity, mode, self.colormap_mat, labelparam,
                        self.landscape_mode, self.mpl_rc)
        if self.draw_statistics:
            draw_boxplot(area_background,
                         area_anchor,
                         graph_size,
                         contour_dict['area'], 'Area(mm$^{2}$)',
                         yaxis_pos='right' if not self.landscape_mode else 'left',
                         log_yaxis=True, landscape_mode=self.landscape_mode, rc=self.mpl_rc)
            draw_boxplot(volume_background,
                         volume_anchor,
                         graph_size,
                         contour_dict['volume'] / 1000, 'Vol.(cm$^{3}$)',
                         yaxis_pos='right' if not self.landscape_mode else 'left',
                         log_yaxis=True, landscape_mode=self.landscape_mode, rc=self.mpl_rc)
            draw_boxplot(length_background,
                         length_anchor,
                         graph_size,
                         contour_dict['length'], 'Length(mm)',
                         yaxis_pos='right' if not self.landscape_mode else 'left',
                         landscape_mode=self.landscape_mode, rc=self.mpl_rc)
            draw_boxplot(perimeter_background,
                         perimeter_anchor,
                         graph_size,
                         contour_dict['perimeter'], 'Perim.(mm)',
                         landscape_mode=self.landscape_mode, rc=self.mpl_rc)
            draw_vhistogram(sphericity_background,
                            sphericity_anchor,
                            graph_size,
                            contour_dict['sphericity'], 'Sphericity',
                            landscape_mode=self.landscape_mode, rc=self.mpl_rc)
            draw_vhistogram(slenderness_background,
                            slenderness_anchor,
                            graph_size,
                            contour_dict['slenderness'], 'Slenderness',
                            landscape_mode=self.landscape_mode, rc=self.mpl_rc)
        if self.draw_graindist:
            draw_grain_dist_curve(grain_dist_graph_background,
                                  grain_dist_graph_anchor,
                                  grain_dist_graph_size,
                                  grain_dist_dict, grain_dist_parameters,
                                  landscape_mode=self.landscape_mode, rc=self.mpl_rc)

        if not self.independent_graphs:
            _, mask_array = cv2.threshold(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
            mask_array = mask_array == 255
            image[:, :, :][mask_array] = background[:, :, :][mask_array]

        ret = {'result.{}'.format(self.result_format): image}
        if self.independent_graphs:
            if self.draw_legend:
                ret.update({'legend.png': make_background_transparent(legend_background)})
            if self.draw_statistics:
                ret.update({
                    'area_boxplot.png': make_background_transparent(area_background),
                    'volume_boxplot.png': make_background_transparent(volume_background),
                    'length_boxplot.png': make_background_transparent(length_background),
                    'perimeter_boxplot.png': make_background_transparent(perimeter_background),
                    'sphericity_histogram.png': make_background_transparent(sphericity_background),
                    'slenderness_histogram.png': make_background_transparent(slenderness_background),
                })
            if self.draw_graindist:
                ret.update({'grain_dist_curve.png': make_background_transparent(grain_dist_graph_background)})

        return ret

# EOF