import math
import os
from io import BytesIO
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import FixedLocator
from matplotlib.ticker import MultipleLocator, LogLocator, LogFormatter
from matplotlib.gridspec import GridSpec
import cv2
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
import pickle
from pathlib import Path
import json
from lib.cv_utils import (
    merge_image,
    global2part_contour,
    bbox_union, torch2cv,
    cv2torch,
    replace_background
)


mpl.rcParams.update({
    'backend': 'SVG',
    'figure.constrained_layout.use': True,
    'figure.dpi': 300,
    'savefig.bbox': 'standard',
})
spec_context = {
    'legend.frameon': True,
    'legend.facecolor': 'white',
    'savefig.bbox': 'standard',
    'font.size': 10,
    'font.family': ['serif'],
    'font.serif': ['Times New Roman'],
}


INKSCAPE_PATH = '/usr/bin/inkscape'


def save_figures(output, prefix, fld):
    if not Path(fld).exists():
        os.makedirs(fld)
    for k, v in output.items():
        with open(Path(fld) / '{}_{}.svg'.format(prefix, k), 'w', encoding='utf-8') as f:
            v['figure'].savefig(f, format='svg')


def save_images(output, prefix, fld):
    if not Path(fld).exists():
        os.makedirs(fld)
    for k, v in output.items():
        TF.to_pil_image(cv2torch(v['image'])).save(Path(fld) / '{}_{}.png'.format(prefix, k))


def visualize_errormap(result_fld, gt_fld): # This function behaves eccentricly from other functions
    palette = {
        'errormap-FN': (187, 85, 102),
        'errormap-FP': (85, 170, 68),
        'errormap-TP': (255, 255, 255),
    }

    def convert_and_binarize_bool(img_path):
        img = PILImage.open(img_path).convert('L')
        img = TF.to_tensor(img).squeeze(0)
        img = torch2cv(img)
        img = img > 127
        return img

    def plot_errormap(pred_path, gt_path):
        pred_mask = convert_and_binarize_bool(pred_path)
        GT_mask = convert_and_binarize_bool(gt_path)

        fn_mask = GT_mask & ~pred_mask
        fp_mask = ~GT_mask & pred_mask
        tp_mask = GT_mask & pred_mask

        fn_mask = np.stack([
            fn_mask.astype(np.uint8) * palette['errormap-FN'][2],
            fn_mask.astype(np.uint8) * palette['errormap-FN'][1],
            fn_mask.astype(np.uint8) * palette['errormap-FN'][0],
        ], axis=2)
        fp_mask = np.stack([
            fp_mask.astype(np.uint8) * palette['errormap-FP'][2],
            fp_mask.astype(np.uint8) * palette['errormap-FP'][1],
            fp_mask.astype(np.uint8) * palette['errormap-FP'][0],
        ], axis=2)
        tp_mask = np.stack([
            tp_mask.astype(np.uint8) * palette['errormap-TP'][2],
            tp_mask.astype(np.uint8) * palette['errormap-TP'][1],
            tp_mask.astype(np.uint8) * palette['errormap-TP'][0],
        ], axis=2)

        emap = cv2.addWeighted(tp_mask, 1, fn_mask, 1, 0)
        emap = cv2.addWeighted(emap, 1, fp_mask, 1, 0)
        return emap


    output_boundary = {}
    output_region = {}
    for img_path in Path(gt_fld).glob('*.jpg'):
        result_boundary_path = Path(result_fld) / (img_path.stem + '_boundary.png')
        result_region_path = Path(result_fld) / (img_path.stem + '_region.png')
        gt_boundary_path = Path(gt_fld) / 'GT_Boundary' / (img_path.stem + '_Muck_boundary.png')
        gt_region_path = Path(gt_fld) / 'GT_Region' / (img_path.stem + '_Muck_region.png')
        if result_boundary_path.is_file() and result_region_path.is_file() and gt_boundary_path.is_file() and gt_region_path.is_file():
            emap_boundary = plot_errormap(result_boundary_path, gt_boundary_path)
            emap_region = plot_errormap(result_region_path, gt_region_path)
            output_boundary[img_path.stem] = {'image': emap_boundary}
            output_region[img_path.stem] = {'image': emap_region}

    return output_boundary, output_region


def plot_grain_dist_curve(grain_dist_dict):
    plt.style.use(['science', 'high-contrast'])
    with mpl.rc_context(spec_context):
        output = {}
        for k, v in grain_dist_dict.items():
            s_data = v['source_grain_dist_data']
            t_data = v['target_grain_dist_data']
            x1 = s_data.index.to_numpy()
            y1 = s_data['cum_area_percentage'].to_numpy()
            x2 = t_data.index.to_numpy()
            y2 = t_data['cum_area_percentage'].to_numpy()

            fig, axes = plt.subplots(1, 1, figsize=(2.756, 2.756))
            axes.plot(x1, y1, label='Prediction')
            axes.plot(x2, y2, label='Ground truth')
            axes.set_xlabel('Size of muck chips (mm)')
            axes.set_ylabel('Cumulative area proportion')
            axes.xaxis.set_ticks_position('bottom')
            axes.yaxis.set_ticks_position('left')
            axes.set_ymargin(0.0)
            axes.grid(visible=True, which='major', ls='-')
            axes.grid(visible=True, which='minor', ls=(0, (5, 10)))
            axes.legend()

            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(-0)
            img = cv2.imdecode(np.asarray(bytearray(buf.read())), cv2.IMREAD_COLOR)

            output[k] = {'figure': fig, 'image': img}
            plt.close()

    return output


def plot_image_cross_validation_result(target_area, source_area, target_orphan_area, source_orphan_area):
    gt = target_area
    pred = source_area
    orphan_gt = target_orphan_area
    orphan_pd = source_orphan_area

    area2size = np.frompyfunc(lambda t: math.sqrt(4 * t / math.pi), 1, 1)
    gt = area2size(gt).astype(float)
    pred = area2size(pred).astype(float)
    orphan_gt = area2size(orphan_gt).astype(float)
    orphan_pd = area2size(orphan_pd).astype(float)

    dif = gt - pred
    dif_min = np.min(dif)
    dif_max = np.max(dif)
    reg_dif_min = int(((dif_min // -10) + (1 if dif_min % -10 != 0 else 0)) * -10)
    reg_dif_max = int(((dif_max // 10) + (1 if dif_min % 10 != 0 else 0)) * 10)
    # make axis range symmetrical to provide correct positioning
    reg_dif_max = max(abs(reg_dif_max), abs(reg_dif_min))
    reg_dif_min = -reg_dif_max
    # ----------------------------------------------------------
    dif_bins, bin_edges = np.histogram(
        dif, bins=reg_dif_max - reg_dif_min,
        range=(reg_dif_min, reg_dif_max)
    )
    bin_centers = [0.5 * (bin_edges[i + 1] + bin_edges[i]) for i in range(len(dif_bins))]

    plt.style.use(['science', 'scatter'])
    with mpl.rc_context(spec_context):
        fig = plt.figure(figsize=(5.91, 5.91))
        gs = GridSpec(3, 3, figure=fig)
        axes_main = fig.add_subplot(gs[1:, :-1])
        axes_main.set_aspect(1.)
        axes_main.invert_xaxis()
        axes_main.invert_yaxis()
        axes_main.xaxis.set_major_locator(MultipleLocator(25))
        axes_main.yaxis.set_major_locator(MultipleLocator(25))
        axes_main.xaxis.set_minor_locator(MultipleLocator(5))
        axes_main.yaxis.set_minor_locator(MultipleLocator(5))
        axes_main.xaxis.set_tick_params(labelbottom=False, labeltop=True)
        axes_main.yaxis.set_tick_params(labelleft=False, labelright=True)
        axes_gt = fig.add_subplot(gs[0, :-1], sharex=axes_main)
        axes_pd = fig.add_subplot(gs[1:, -1], sharey=axes_main)
        axes_gt.xaxis.set_tick_params(labelbottom=False)
        axes_pd.yaxis.set_tick_params(labelleft=False)

        axes_main.axline(xy1=(0, 0), slope=1, color='black', ls='--')
        axes_main.axline(xy1=(dif_max, 0), slope=1, color='red', ls='-', lw=0.25)
        axes_main.axline(xy1=(0, -dif_min), slope=1, color='red', ls='-', lw=0.25)
        axes_main.scatter(gt, pred, s=2, label='Matched chips')
        max_lim = max(axes_main.get_xlim()[0], axes_main.get_ylim()[0])
        axe_lim = int((max_lim // 5 + 1)) * 5
        axes_main.set_xlim(axe_lim, 0)
        axes_main.set_ylim(axe_lim, 0)
        axes_main.set_xlabel('Size of muck chips in ground truth (mm)')
        axes_main.set_ylabel('Size of muck chips in prediction (mm)')
        axes_main.set_xmargin(0.0)
        axes_main.set_ymargin(0.0)

        num_side_bins = axe_lim // 5
        # bar_width = 3.6 * 40 / num_side_bins
        bar_width = 4
        # main_locs = np.hstack([axes_main.xaxis.get_ticklocs(), axes_main.xaxis.get_minorticklocs()])
        # main_locs.sort()
        # num_side_bins = len(main_locs) - 1
        gt_bins, gt_bin_edges = np.histogram(
            gt, bins=num_side_bins,
            range=(0, axe_lim)
        )
        gt_bin_centers = [0.5 * (gt_bin_edges[i + 1] + gt_bin_edges[i]) for i in range(len(gt_bins))]
        axes_gt.bar(gt_bin_centers, gt_bins, width=bar_width, color='#7f7f7f', edgecolor='black', label='Matched in GT')
        pd_bins, pd_bin_edges = np.histogram(
            pred, bins=num_side_bins,
            range=(0, axe_lim)
        )
        pd_bin_centers = [0.5 * (pd_bin_edges[i + 1] + pd_bin_edges[i]) for i in range(len(pd_bins))]
        axes_pd.barh(pd_bin_centers, pd_bins, height=bar_width, color='#7fa1c3', edgecolor='#004488', label='Matched in Pred.')
        ogt_bins, ogt_bin_edges = np.histogram(
            orphan_gt, bins=num_side_bins,
            range=(0, axe_lim)
        )
        opd_bins, opd_bin_edges = np.histogram(
            orphan_pd, bins=num_side_bins,
            range=(0, axe_lim)
        )
        ogt_bin_centers = [0.5 * (ogt_bin_edges[i + 1] + ogt_bin_edges[i]) for i in range(len(ogt_bins))]
        opd_bin_centers = [0.5 * (opd_bin_edges[i + 1] + opd_bin_edges[i]) for i in range(len(opd_bins))]
        axes_gt_twin = axes_gt.twinx()
        axes_pd_twin = axes_pd.twiny()
        axes_gt_twin.invert_yaxis()
        axes_pd_twin.invert_xaxis()
        axes_gt_twin.bar(ogt_bin_centers, ogt_bins, width=bar_width, color='#ddaab2', edgecolor='#BB5566', label='Orphan in GT')
        axes_pd_twin.barh(opd_bin_centers, opd_bins, height=bar_width, color='#eed499', edgecolor='#DDAA33',
                          label='Orphan in Pred.')
        axes_gt.set_ylim(0, math.ceil((np.max(gt_bins) + np.max(ogt_bins)) * 1.1))
        axes_gt_twin.set_ylim((math.ceil(np.max(gt_bins) + np.max(ogt_bins)) * 1.1), 0)
        axes_pd.set_xlim(0, math.ceil((np.max(pd_bins) + np.max(opd_bins)) * 1.1))
        axes_pd_twin.set_xlim(math.ceil((np.max(pd_bins) + np.max(opd_bins)) * 1.1), 0)
        axes_pd_twin.set_xlabel('Amount of muck chips')

        tr = Affine2D().scale(1 / math.sqrt(2), 1 / math.sqrt(2)).rotate_deg(135)
        dif_vec = tr.transform(np.array([math.sqrt(2), 0.]))
        center_value = np.array([axe_lim / 2, axe_lim / 2])
        x_c, y_c = fig.transFigure.inverted().transform(axes_main.transData.transform(center_value))
        x_u, y_u = fig.transFigure.inverted().transform(axes_main.transData.transform(center_value + reg_dif_max * dif_vec))
        x_l, y_l = fig.transFigure.inverted().transform(axes_main.transData.transform(center_value + reg_dif_min * dif_vec))
        h_u = w_u = math.sqrt((x_u - x_c) ** 2 + (y_u - y_c) ** 2)
        h_l = w_l = math.sqrt((x_l - x_c) ** 2 + (y_l - y_c) ** 2)
        R = tr.transform([
            (x_c - w_l, y_c - h_l),
            (x_c + w_u, y_c - h_l),
            (x_c - w_l, y_c + h_u),
            (x_c + w_u, y_c + h_u),
        ])
        w_tr = R[:, 0].max() - R[:, 0].min()
        h_tr = R[:, 1].max() - R[:, 1].min()
        w_tr_l = w_tr * w_l / (w_l + w_u)
        h_tr_l = h_tr * h_l / (h_l + h_u)
        w_tr_u = w_tr - w_tr_l
        h_tr_u = h_tr - h_tr_l

        with mpl.rc_context({
            'axes.facecolor': 'none',
            'xtick.direction': 'out',
            'xtick.labelsize': 6,
            'xtick.major.size': 2,
            'xtick.major.width': 0.5,
            'ytick.direction': 'out',
            'ytick.labelsize': 6,
            'ytick.major.size': 2,
            'ytick.major.width': 0.5,
        }):
            xmin, xmax = reg_dif_min, reg_dif_max
            ymin, ymax = 0, xmax - xmin
            grid_helper = floating_axes.GridHelperCurveLinear(
                tr, (xmin, xmax, ymin, ymax),
                grid_locator1=FixedLocator(np.linspace(reg_dif_min, reg_dif_max, (reg_dif_max - reg_dif_min) // 10 + 1))
            )
            fl_axes = floating_axes.FloatingSubplot(
                fig, 111, grid_helper=grid_helper, zorder=0
            )
            fl_axes.set_position((x_c - w_tr_u, y_c - h_tr_u, w_tr, h_tr))
            fl_axes.axis['left'].set_visible(False)
            fl_axes.axis['right'].set_visible(False)
            fl_axes.axis['top'].set_visible(False)
            fl_axes.axis['bottom'].major_ticks.set_tick_out(True)

            fig.add_subplot(fl_axes)
            aux_axes = fl_axes.get_aux_axes(tr)
            aux_axes.bar(bin_centers, dif_bins * (xmax - xmin) / max(dif_bins), color='#DDAA33', alpha=0.7)

        fig.legend(loc='center')

        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(-0)
        img = cv2.imdecode(np.asarray(bytearray(buf.read())), cv2.IMREAD_COLOR)

    return {'figure': fig, 'image': img}


def plot_cross_validation_results(validation_dict):
    merged_gt = np.array([])
    merged_pd = np.array([])
    merged_orphan_gt = np.array([])
    merged_orphan_pd = np.array([])
    output = {}
    image_infos = []
    for k, v in validation_dict.items():
        target_area = v['merged_annotation']['target_area'].to_numpy()
        source_area = v['merged_annotation']['source_area'].to_numpy()
        target_orphan_area = v['target_orphan_annotation']['area'].to_numpy()
        source_orphan_area = v['source_orphan_annotation']['area'].to_numpy()
        output[k] = plot_image_cross_validation_result(target_area, source_area, target_orphan_area, source_orphan_area)
        plt.close()
        merged_gt = np.hstack([merged_gt, target_area])
        merged_pd = np.hstack([merged_pd, source_area])
        merged_orphan_gt = np.hstack([merged_orphan_gt, target_orphan_area])
        merged_orphan_pd = np.hstack([merged_orphan_pd, source_orphan_area])

    output['Overall'] = plot_image_cross_validation_result(merged_gt, merged_pd, merged_orphan_gt, merged_orphan_pd)
    return output


def plot_image_geom_parameter_scatter(source_quantity, target_quantity, label):
    gt = target_quantity
    pred = source_quantity
    dif = gt - pred
    dif_min = np.min(dif)
    dif_max = np.max(dif)
    # make axis range symmetrical to provide correct positioning
    reg_dif_max = math.ceil(max(abs(dif_max), abs(dif_min)) * 10) / 10
    reg_dif_min = -reg_dif_max
    # ----------------------------------------------------------
    dif_bins, bin_edges = np.histogram(
        dif, bins=10 * int(reg_dif_max * 10),
        range=(reg_dif_min, reg_dif_max)
    )
    bin_centers = [0.5 * (bin_edges[i + 1] + bin_edges[i]) for i in range(len(dif_bins))]
    bin_spacing = bin_edges[1] - bin_edges[0]

    plt.style.use(['science', 'scatter'])
    with mpl.rc_context(spec_context):
        fig = plt.figure(figsize=(2.756, 2.756))
        axes = fig.gca()
        axes.set_aspect(1.)

        axes.axline(xy1=(0, 0), slope=1, color='black', ls='--')
        axes.axline(xy1=(dif_max, 0), slope=1, color='red', ls='-', lw=0.25)
        axes.axline(xy1=(0, -dif_min), slope=1, color='red', ls='-', lw=0.25)
        axes.scatter(gt, pred, s=1, label='Matched chips')
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.set_xlabel('{} of chips in GT (mm)'.format(label))
        axes.set_ylabel('{} of chips in pred. (mm)'.format(label))
        axes.set_xmargin(0.0)
        axes.set_ymargin(0.0)

        tr = Affine2D().scale(1 / math.sqrt(2), 1 / math.sqrt(2)).rotate_deg(-45)
        dif_vec = tr.transform(np.array([math.sqrt(2), 0.]))
        center_value = np.array([0.5, 0.5])
        x_c, y_c = fig.transFigure.inverted().transform(axes.transData.transform(center_value))
        x_u, y_u = fig.transFigure.inverted().transform(axes.transData.transform(center_value + reg_dif_max * dif_vec))
        x_l, y_l = fig.transFigure.inverted().transform(axes.transData.transform(center_value + reg_dif_min * dif_vec))
        h_u = w_u = math.sqrt((x_u - x_c) ** 2 + (y_u - y_c) ** 2)
        h_l = w_l = math.sqrt((x_l - x_c) ** 2 + (y_l - y_c) ** 2)
        R = tr.transform([
            (x_c - w_l, y_c - h_l),
            (x_c + w_u, y_c - h_l),
            (x_c - w_l, y_c + h_u),
            (x_c + w_u, y_c + h_u),
        ])
        w_tr = R[:, 0].max() - R[:, 0].min()
        h_tr = R[:, 1].max() - R[:, 1].min()
        w_tr_l = w_tr * w_l / (w_l + w_u)
        h_tr_l = h_tr * h_l / (h_l + h_u)
        w_tr_u = w_tr - w_tr_l
        h_tr_u = h_tr - h_tr_l

        with mpl.rc_context({
            'axes.facecolor': 'none',
            'xtick.direction': 'out',
            'xtick.labelsize': 6,
            'xtick.major.size': 2,
            'xtick.major.width': 0.5,
            'ytick.direction': 'out',
            'ytick.labelsize': 6,
            'ytick.major.size': 2,
            'ytick.major.width': 0.5,
        }):
            xmin, xmax = reg_dif_min, reg_dif_max
            ymin, ymax = 0, xmax - xmin
            grid_helper = floating_axes.GridHelperCurveLinear(
                tr, (xmin, xmax, ymin, ymax),
                grid_locator1=FixedLocator([-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                                            0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            )
            fl_axes = floating_axes.FloatingSubplot(
                fig, 111, grid_helper=grid_helper, zorder=0
            )
            fl_axes.set_position((x_c - w_tr_l, y_c - h_tr_l, w_tr, h_tr))
            fl_axes.axis['left'].set_visible(False)
            fl_axes.axis['right'].set_visible(False)
            fl_axes.axis['top'].set_visible(False)
            fl_axes.axis['bottom'].major_ticks.set_tick_out(True)

            fig.add_subplot(fl_axes)
            aux_axes = fl_axes.get_aux_axes(tr)
            aux_axes.bar(bin_centers, dif_bins * 0.5 * (xmax - xmin) / max(dif_bins), width=0.016, color='#DDAA33', alpha=0.7)

        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(-0)
        img = cv2.imdecode(np.asarray(bytearray(buf.read())), cv2.IMREAD_COLOR)

    return {'figure': fig, 'image': img}


def plot_geom_parameter_scatter(validation_dict, size_threshold=20):
    area_threshold = size_threshold ** 2 * math.pi / 4
    merged_source_slenderness = np.array([])
    merged_source_sphericity = np.array([])
    merged_target_slenderness = np.array([])
    merged_target_sphericity = np.array([])
    output_slenderness = {}
    output_sphericity = {}
    for k, v in validation_dict.items():
        mask = v['merged_annotation']['target_area'] >= area_threshold
        source_slenderness = v['merged_annotation']['source_slenderness'].loc[mask].to_numpy()
        source_sphericity = v['merged_annotation']['source_sphericity'].loc[mask].to_numpy()
        target_slenderness = v['merged_annotation']['target_slenderness'].loc[mask].to_numpy()
        target_sphericity = v['merged_annotation']['target_sphericity'].loc[mask].to_numpy()
        merged_source_slenderness = np.hstack([merged_source_slenderness, source_slenderness])
        merged_source_sphericity = np.hstack([merged_source_sphericity, source_sphericity])
        merged_target_slenderness = np.hstack([merged_target_slenderness, target_slenderness])
        merged_target_sphericity = np.hstack([merged_target_sphericity, target_sphericity])

    output_slenderness['Overall'] = plot_image_geom_parameter_scatter(merged_source_slenderness, merged_target_slenderness, 'Slenderness')
    output_sphericity['Overall'] = plot_image_geom_parameter_scatter(merged_source_sphericity, merged_target_sphericity, 'Sphericity')
    return output_slenderness, output_sphericity


def plot_Hausdorff_distance_histogram(validation_dict):
    def plot_hist(hd60_array, hd95_array, source_p_array, target_p_array, figsize):
        max_hd = max(np.max(hd60_array), np.max(hd95_array))
        if max_hd <= 10:
            hd_base = 0.5
            major_spacing = 2
        elif max_hd <= 20:
            hd_base = 1.
            major_spacing = 5
        elif max_hd <= 30:
            hd_base = 1.5
            major_spacing = 5
        else:
            hd_base = 2.
            major_spacing = 10
        reg_max_hd = min(40, int((max_hd // hd_base + (1 if max_hd % hd_base != 0 else 0))) * hd_base)
        num_bins = int(reg_max_hd / hd_base)
        bar_width = 0.8 * hd_base

        hd60_bins, hd60_edges = np.histogram(hd60_array, bins=num_bins, range=(0, reg_max_hd))
        hd95_bins, hd95_edges = np.histogram(hd95_array, bins=num_bins, range=(0, reg_max_hd))
        source_p_bins, source_p_edges = np.histogram(source_p_array, bins=num_bins, range=(0, reg_max_hd))
        target_p_bins, target_p_edges = np.histogram(target_p_array, bins=num_bins, range=(0, reg_max_hd))

        main_centers = [0.5 * (source_p_edges[i + 1] + source_p_edges[i]) for i in range(len(source_p_bins))]
        hd60_centers = [i - 0.2 * hd_base for i in main_centers]
        hd95_centers = [i + 0.2 * hd_base for i in main_centers]
        total_instances = np.sum(hd60_bins) + np.sum(source_p_bins) + np.sum(target_p_bins)
        hd60_bins = hd60_bins * 100 / total_instances
        hd95_bins = hd95_bins * 100 / total_instances
        source_p_bins = source_p_bins * 100 / total_instances
        target_p_bins = target_p_bins * 100 / total_instances
        bottom = source_p_bins + target_p_bins

        plt.style.use(['science', 'high-contrast'])
        with mpl.rc_context(spec_context):
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            # axes.hist(hd_array, bins=num_bins, range=(0, reg_max_hd), label=['60-HD', '95-HD'], density=True)
            axes.bar(main_centers, target_p_bins, width=bar_width, color='#BB5566', label='GT penalty')
            axes.bar(main_centers, source_p_bins, width=bar_width, bottom=target_p_bins, color='#55AA44', label='Pred. penalty')
            axes.bar(hd60_centers, hd60_bins, width=0.5 * bar_width, bottom=bottom, color='#004488', label='60-HD')
            axes.bar(hd95_centers, hd95_bins, width=0.5 * bar_width, bottom=bottom, color='#DDAA33', label='95-HD')
            axes.xaxis.set_major_locator(MultipleLocator(major_spacing))
            axes.xaxis.set_minor_locator(MultipleLocator(hd_base))
            axes.set_xlabel('Hausdorff dist. w/ penalty (mm)')
            axes.set_ylabel('Percentage ${(\%)}$')
            axes.set_ymargin(0.0)
            axes.set_ylim(0, axes.get_ylim()[1] * 1.05)
            axes.legend()

            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(-0)
            img = cv2.imdecode(np.asarray(bytearray(buf.read())), cv2.IMREAD_COLOR)

        return fig, img

    output = {}
    merged_hd60 = np.array([])
    merged_hd95 = np.array([])
    merged_s_penalty = np.array([])
    merged_t_penalty = np.array([])
    for k, v in validation_dict.items():
        hd60 = v['merged_annotation']['60-HD'].to_numpy()
        hd95 = v['merged_annotation']['95-HD'].to_numpy()
        s_penalty = v['source_orphan_annotation']['penalty'].to_numpy()
        t_penalty = v['target_orphan_annotation']['penalty'].to_numpy()
        merged_hd60 = np.hstack([merged_hd60, hd60])
        merged_hd95 = np.hstack([merged_hd95, hd95])
        merged_s_penalty = np.hstack([merged_s_penalty, s_penalty])
        merged_t_penalty = np.hstack([merged_t_penalty, t_penalty])
        fig, img = plot_hist(hd60, hd95, s_penalty, t_penalty, figsize=(1.87, 2.756))
        output[k] = {'figure': fig, 'image': img}
        plt.close()

    fig, img = plot_hist(merged_hd60, merged_hd95, merged_s_penalty, merged_t_penalty, figsize=(2.756, 2.756))
    output['Overall'] = {'figure': fig, 'image': img}

    average_hd60 = np.sum(merged_hd60) / len(merged_hd60)
    average_hd95 = np.sum(merged_hd95) / len(merged_hd95)
    average_hd60_penalty = (np.sum(merged_hd60) + np.sum(merged_s_penalty) + np.sum(merged_t_penalty)) \
                           / (len(merged_hd60) + len(merged_s_penalty) + len(merged_t_penalty))
    average_hd95_penalty = (np.sum(merged_hd95) + np.sum(merged_s_penalty) + np.sum(merged_t_penalty)) \
                           / (len(merged_hd95) + len(merged_s_penalty) + len(merged_t_penalty))

    return output, {
        'average-60-HD': average_hd60, 'average-95-HD': average_hd95,
        'average-60-HD-with-penalty': average_hd60_penalty,
        'average-95-HD-with-penalty': average_hd95_penalty
    }


def visualize_Hausdorff_distance(validation_dict, orig_img_fld):
    @mpl.rc_context({'figure.constrained_layout.use': True})
    def draw_legend(min, max, label, colormap, log_scale=False):
        plt.style.use('science')
        with mpl.rc_context(spec_context):
            fig, axes = plt.subplots(1, 1, figsize=(2, 0.393))
            if log_scale:
                norm = mpl.colors.LogNorm(vmin=min, vmax=max)
                axes.yaxis.set_major_locator(LogLocator(base=10, subs=[0.2, 0.5, 1.0]))
                axes.yaxis.set_major_formatter(LogFormatter(base=10, minor_thresholds=(3, 1)))
            else:
                norm = mpl.colors.Normalize(vmin=min, vmax=max)
            im = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
            cbar = fig.colorbar(im, cax=axes, orientation='horizontal')
            axes.xaxis.set_ticks_position('top')
            axes.xaxis.set_tick_params(which='major', direction='out', length=4, width=0.5)
            axes.xaxis.set_tick_params(which='minor', direction='out', length=2, width=0.25)
            axes.set_ylabel(label, rotation=0, loc='top')

            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(-0)
            img = cv2.imdecode(np.asarray(bytearray(buf.read())), cv2.IMREAD_COLOR)

        return {'figure': fig, 'image': img}

    colormap = 'autumn'
    inverse_colormap = True
    colormap_cv = getattr(cv2, 'COLORMAP_' + colormap.upper())
    colormap_mat = getattr(mpl.cm, (colormap + '_r') if inverse_colormap else colormap)

    output_60 = {}
    output_95 = {}
    output_60_legend = {}
    output_95_legend = {}
    for k, v in validation_dict.items():
        orig_img = PILImage.open(orig_img_fld / (k + '.jpg'))
        orig_img = torch2cv(TF.to_tensor(orig_img))
        gt_cts = v['merged_annotation']['target_segmentation']
        pd_cts = v['merged_annotation']['source_segmentation']
        gt_bboxes = v['merged_annotation']['target_bbox']
        pd_bboxes = v['merged_annotation']['source_bbox']
        hd60s = v['merged_annotation']['60-HD']
        hd95s = v['merged_annotation']['95-HD']
        orphan_gt_cts = v['target_orphan_annotation']['segmentation']
        orphan_pd_cts = v['source_orphan_annotation']['segmentation']

        min_hd60 = hd60s.min()
        max_hd60 = hd60s.max()
        min_hd95 = hd95s.min()
        max_hd95 = hd95s.max()

        output_60_legend[k] = draw_legend(min_hd60, max_hd60, '60-HD', colormap_mat)
        output_95_legend[k] = draw_legend(min_hd95, max_hd95, '95-HD', colormap_mat)

        hd60_mask = np.zeros((4096, 2048, 3), dtype=np.uint8)
        hd95_mask = np.zeros((4096, 2048, 3), dtype=np.uint8)
        for idx in gt_cts.index:
            gt_ct = gt_cts.loc[idx].reshape(-1, 1, 2)
            pd_ct = pd_cts.loc[idx].reshape(-1, 1, 2)
            gt_bbox = gt_bboxes.loc[idx]
            pd_bbox = pd_bboxes.loc[idx]
            hd60_value = int(255 * ((hd60s.loc[idx] - min_hd60) / (max_hd60 - min_hd60)) + 0.5)
            hd95_value = int(255 * ((hd95s.loc[idx] - min_hd95) / (max_hd95 - min_hd95)) + 0.5)
            if inverse_colormap:
                hd60_value = 255 - hd60_value
                hd95_value = 255 - hd95_value

            bbox = bbox_union(gt_bbox, pd_bbox)
            temp_mask = np.zeros((bbox[3], bbox[2]), dtype=np.uint8)
            part_gt_contour = global2part_contour([gt_ct], bbox)
            cv2.fillPoly(temp_mask, part_gt_contour, 1)
            temp_mask_60 = cv2.applyColorMap(temp_mask * hd60_value, colormap_cv) * \
                           np.stack([temp_mask, temp_mask, temp_mask], axis=2) * 0.6
            temp_mask_95 = cv2.applyColorMap(temp_mask * hd95_value, colormap_cv) * \
                           np.stack([temp_mask, temp_mask, temp_mask], axis=2) * 0.6
            temp_mask_60 = cv2.polylines(
                temp_mask_60, part_gt_contour, True,
                color=tuple([int(i) for i in cv2.applyColorMap(np.array([[hd60_value]], dtype=np.uint8), colormap_cv)[0, 0]]),
                thickness=1
            )
            temp_mask_95 = cv2.polylines(
                temp_mask_95, part_gt_contour, True,
                color=tuple([int(i) for i in cv2.applyColorMap(np.array([[hd95_value]], dtype=np.uint8), colormap_cv)[0, 0]]),
                thickness=1
            )
            merge_image(hd60_mask, temp_mask_60.astype(np.uint8), bbox)
            merge_image(hd95_mask, temp_mask_95.astype(np.uint8), bbox)

        hd60_img = hd60_mask
        hd95_img = hd95_mask
        hd60_img = cv2.polylines(hd60_img, list(orphan_gt_cts), True, color=(0, 0, 255), thickness=2)
        hd60_img = cv2.polylines(hd60_img, list(orphan_pd_cts), True, color=(0, 255, 0), thickness=2)
        hd60_img = cv2.polylines(hd60_img, list(pd_cts), True, color=(255, 0, 0), thickness=2)
        hd60_img = replace_background(hd60_img, (60, 60, 60))
        hd95_img = cv2.polylines(hd95_img, list(orphan_gt_cts), True, color=(0, 0, 255), thickness=2)
        hd95_img = cv2.polylines(hd95_img, list(orphan_pd_cts), True, color=(0, 255, 0), thickness=2)
        hd95_img = cv2.polylines(hd95_img, list(pd_cts), True, color=(255, 0, 0), thickness=2)
        hd95_img = replace_background(hd95_img, (60, 60, 60))

        output_60[k] = {'image': hd60_img}
        output_95[k] = {'image': hd95_img}

    return output_60, output_95, output_60_legend, output_95_legend

# EOF