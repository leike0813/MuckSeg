from copy import copy
import math
from io import BytesIO
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
import cv2
from lib.cv_utils import (
    enlarge_bbox,
    global2part_image,
    global2part_contour,
    part2global_contour,
    part2global_center,
    part2global_bbox,
    filter_max_area_contour,
    contour_in_contours_test,
)


def calculate_valid_contours(part_contour, part_region_subtract, part_region_mask, bbox, part_orig_image,
                             center_prob_thresh, center_open_iter, max_erosion_iter, erosion_mode,
                             kernel, center_rel_dist_thresh):
    part_center_mask = (part_region_subtract > center_prob_thresh).astype(np.uint8)
    part_center_mask = cv2.morphologyEx(part_center_mask, cv2.MORPH_OPEN, kernel, iterations=center_open_iter)
    num_center, center_instance_mask, stats, center_points = cv2.connectedComponentsWithStats(
        part_center_mask, None, 4)
    centers = [tuple(center_points[i]) for i in range(1, len(center_points))]

    if len(centers) == 0:
        return []
    elif len(centers) == 1:
        return [(part2global_contour([part_contour], bbox)[0], part2global_center(centers[0], bbox), bbox)]
    else:
        contours = [{'contour': part_contour, 'centers': copy(centers)}]
        output = []
        for i in range(max_erosion_iter if erosion_mode == 1 else 1):
            cur_num_con = len(contours)
            erosion_iter = i + 1 if erosion_mode == 1 else max_erosion_iter
            part_region_mask = cv2.erode(part_region_mask, kernel, iterations=1 if erosion_mode == 1 else max_erosion_iter)
            eroded_contours, _ = cv2.findContours(part_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_con = len(eroded_contours)
            if num_con > cur_num_con:
                contours = []
                _center_remove_list = []
                for contour in eroded_contours:
                    if contour.shape[0] > 2:
                        _c_dict = {'contour': contour, 'centers': []}
                        for center in centers:
                            ret = cv2.pointPolygonTest(contour, center, measureDist=False)
                            if ret > 0:
                                _c_dict['centers'].append(center)
                        if len(_c_dict['centers']) == 1:
                            reverse_mask = np.zeros((bbox[3], bbox[2]), dtype=np.uint8)
                            cv2.fillPoly(reverse_mask, [contour], 1)
                            part_region_mask -= reverse_mask
                            reverse_mask = cv2.dilate(reverse_mask, kernel, iterations=erosion_iter)
                            reverse_contours, _ = cv2.findContours(reverse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            output.append((
                                part2global_contour(reverse_contours, bbox)[0],
                                part2global_center(_c_dict['centers'][0], bbox), bbox
                            ))
                            _center_remove_list.append(_c_dict['centers'][0])
                        elif len(_c_dict['centers']) > 1:
                            reverse_mask = np.zeros((bbox[3], bbox[2]), dtype=np.uint8)
                            cv2.fillPoly(reverse_mask, [_c_dict['contour']], 1)
                            reverse_mask = cv2.dilate(reverse_mask, kernel, iterations=erosion_iter)
                            reverse_contours, _ = cv2.findContours(reverse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            _c_dict['contour'] = reverse_contours[0]
                            contours.append(_c_dict)
                for center in _center_remove_list:
                    centers.remove(center)
            if len(contours) == 0:
                return output

        for _c_dict in contours:
            _c_mask = np.zeros((bbox[3], bbox[2]), dtype=np.uint8)
            cv2.fillPoly(_c_mask, [_c_dict['contour']], 1)

            num_centers = len(_c_dict['centers'])
            rel_dist_matr = np.zeros((num_centers, num_centers))
            dist_transform = cv2.distanceTransform(_c_mask, cv2.DIST_L2, 5)
            center_dists = [dist_transform[int(_center[1]), int(_center[0])] for _center in _c_dict['centers']]
            for i in range(num_centers):
                _center_i = _c_dict['centers'][i]
                _dist_i = center_dists[i]
                for j in range(num_centers):
                    _center_j = _c_dict['centers'][j]
                    _dist_j = dist_transform[int(_center_j[1]), int(_center_j[0])]
                    if j == i:
                        _rel_dist = 0
                    else:
                        _rel_dist = math.sqrt(
                            (_center_i[0] - _center_j[0]) ** 2 + (_center_i[1] - _center_j[1]) ** 2) / _dist_j
                    rel_dist_matr[i, j] = _rel_dist

            _drop_list = set()
            for i in range(num_centers):
                for j in range(num_centers):
                    if j > i and max(rel_dist_matr[i, j], rel_dist_matr[j, i]) < center_rel_dist_thresh:
                        _drop_list.add((_c_dict['centers'][j], center_dists[j]))

            for _drop in _drop_list:
                _c_dict['centers'].remove(_drop[0])
                center_dists.remove(_drop[1])

            if len(_c_dict['centers']) == 1:
                output.append((
                    part2global_contour([_c_dict['contour']], bbox)[0],
                    part2global_center(_c_dict['centers'][0], bbox), bbox
                ))
            else:
                if part_orig_image is not None:
                    # -----------apply stage 2-----------------------
                    num_centers = len(_c_dict['centers'])
                    # orig_image = torch2cv(orig_image)
                    # marker = reverse_mask.astype(np.int32) - 1
                    marker = np.zeros((_c_mask.shape[0], _c_mask.shape[1]), dtype=np.int32)
                    for i in range(num_centers):
                        marker[int(_c_dict['centers'][i][1]), int(_c_dict['centers'][i][0])] = i + 1
                    marker = cv2.watershed(part_orig_image, marker)
                    marker[_c_mask == 0] = 0
                    temp_output = []
                    for idx in range(num_centers):
                        seg_mask = (marker == idx + 1).astype(np.uint8)
                        seg_contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if len(seg_contours) > 0:
                            if len(seg_contours) > 1:
                                seg_contours = filter_max_area_contour(seg_contours)
                            temp_output.append((
                                part2global_contour(seg_contours, bbox)[0],
                                part2global_center(_c_dict['centers'][idx], bbox), bbox
                            ))
                    # filter out full-contained contours
                    _drop_list = set()
                    for i in range(len(temp_output)):
                        contour_to_test = temp_output[i][0]
                        contours_for_test = []
                        for j in range(len(temp_output)):
                            if i != j:
                                contours_for_test.append(temp_output[j][0])
                        test_result = contour_in_contours_test(contour_to_test, contours_for_test)
                        if np.any(test_result):
                            _drop_list.add(i)
                    for idx in reversed(sorted(list(_drop_list))):
                        temp_output.pop(idx)
                    output.extend(temp_output)
                    # -----------apply stage 2-----------------------
                else:
                    # -----------retain center with maximum dist-----------------------
                    output.append((
                        part2global_contour([_c_dict['contour']], bbox)[0],
                        part2global_center(_c_dict['centers'][center_dists.index(max(center_dists))], bbox), bbox
                    ))
                    # -----------retain center with maximum dist-----------------------
        return output


def dilate_contour(contours, kernel, anchor=(-1, -1), iterations=1, ret_bbox=True, calc_area=False):
    bbox = cv2.boundingRect(np.vstack(contours))
    bbox = enlarge_bbox(bbox, iterations)
    _mask = np.zeros((bbox[3], bbox[2]), dtype=np.uint8)
    part_contours = global2part_contour(contours, bbox)
    cv2.fillPoly(_mask, part_contours, 1)
    _mask = cv2.dilate(_mask, kernel, anchor=anchor, iterations=iterations)
    part_contours, _ = cv2.findContours(_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = part2global_contour(part_contours, bbox)
    ret = [contours]
    if ret_bbox:
        ret.append(bbox)
    if calc_area:
        ret.append(np.sum(_mask))
    return tuple(ret)


def dilate_contour_and_calculate_characteristics(points, center, contour_index, redilate_iter, pixel_ratio, kernel):
    points, bbox, area = dilate_contour([points], kernel, iterations=redilate_iter, calc_area=True)
    points = points[0]
    area = area * (pixel_ratio ** 2)
    perimeter = cv2.arcLength(points, True) * pixel_ratio
    sphericity = math.sqrt(4 * math.pi * area) / perimeter
    rect = cv2.minAreaRect(points)
    length = max(rect[1][0], rect[1][1]) * pixel_ratio
    slenderness = min(rect[1][0], rect[1][1]) * pixel_ratio / length
    ellip = cv2.fitEllipse(points) if len(points) >= 5 else cv2.minAreaRect(points)
    volume = math.pi * ellip[1][1] * ellip[1][0] ** 2 * pixel_ratio ** 3 / 6

    return contour_index, points, bbox, center, area, length, perimeter, sphericity, slenderness, volume


def seed_filling(markers, max_iterations, connectivity=4):
    """
    :param markers: 2-D array of int32, -1 means boundary, 0 means unknown area, >0 means seed area of corresponding seed index
    :param max_iterations:
    :param connectivity: 4 or 8
    :return:
    """
    def candidates(point, image_shape, connectivity):
        _can = []
        if point[0] > 0:
            _can.append((point[0] - 1, point[1]))
        if point[0] < image_shape[0] - 1:
            _can.append((point[0] + 1, point[1]))
        if point[1] > 0:
            _can.append((point[0], point[1] - 1))
        if point[1] < image_shape[1] - 1:
            _can.append((point[0], point[1] + 1))
        if connectivity == 8:
            if point[0] > 0 and point[1] > 0:
                _can.append((point[0] - 1, point[1] - 1))
            if point[0] > 0 and point[1] < image_shape[1] - 1:
                _can.append((point[0] - 1, point[1] + 1))
            if point[0] < image_shape[0] - 1 and point[1] > 0:
                _can.append((point[0] + 1, point[1] - 1))
            if point[0] < image_shape[0] - 1 and point[1] < image_shape[1] - 1:
                _can.append((point[0] + 1, point[1] + 1))
        return _can

    def expand_iter(front_points, seed_id, image_shape, connectivity):
        new_front_points = set()
        for point in front_points:
            point_candiates = candidates(point, image_shape, connectivity)
            for can in point_candiates:
                if markers[can[1], can[0]] == 0:
                    markers[can[1], can[0]] = seed_id
                    new_front_points.add(can)
        return new_front_points

    image_shape = (markers.shape[1], markers.shape[0])
    num_seeds = np.max(markers)
    seed_contour_mask = np.zeros(markers.shape, dtype=int)
    for i in range(num_seeds):
        seed_id = i + 1
        seed_mask = (markers == seed_id).astype(np.uint8)
        seed_contours, _ = cv2.findContours(seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        seed_mask.fill(0)
        seed_mask = cv2.polylines(seed_mask, seed_contours, True, 1)
        seed_contour_mask += seed_mask * seed_id

    assert connectivity in [4, 8], 'Invalid connectivity, must be 4 or 8'
    seed_front_points = [set() for i in range(num_seeds)]
    for x in range(image_shape[0]):
        for y in range(image_shape[1]):
            for i in range(num_seeds):
                seed_id = i + 1
                if seed_contour_mask[y, x] == seed_id:
                    seed_front_points[i].add((x, y))

    for it in range(max_iterations):
        new_seed_front_points = []
        for i in range(num_seeds):
            seed_id = i + 1
            new_seed_front_points.append(expand_iter(seed_front_points[i], seed_id, image_shape, connectivity))
        seed_front_points = new_seed_front_points

    return markers


def extract_bbox_and_mask(subtracted_mask):
    num_center, center_instance_mask, stats, center_points = cv2.connectedComponentsWithStats(
        subtracted_mask, None, 4)
    part_masks = []
    for i in range(1, num_center):
        instance_mask = (center_instance_mask == i).astype(np.uint8)
        instance_contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        instance_bbox = cv2.boundingRect(instance_contours[0])
        part_instance_mask = global2part_image(instance_mask, instance_bbox)
        part_masks.append((part_instance_mask, instance_contours[0], instance_bbox))

    return part_masks


def erode_and_seedfill_part_mask(part_instance_mask, instance_contour, instance_bbox, kernel, erosion_lambda=0.15):
    dist_transform = cv2.distanceTransform(part_instance_mask, cv2.DIST_L2, 5)
    erosion_iter = int(erosion_lambda * np.max(dist_transform))
    eroded_mask = cv2.erode(part_instance_mask, kernel, iterations=erosion_iter)
    num_center, eroded_instance_mask, stats, center_points = cv2.connectedComponentsWithStats(
        eroded_mask, None, 4)
    if num_center == 2:
        return [(part_instance_mask, instance_contour, instance_bbox)]
    else:
        part_instance_contours = global2part_contour([instance_contour], instance_bbox)
        marker = np.zeros(part_instance_mask.shape, dtype=np.uint8)
        marker = cv2.polylines(marker, part_instance_contours, True, 1)
        marker = eroded_instance_mask + marker * -1
        marker = seed_filling(marker, max_iterations=int(erosion_iter * 2), connectivity=4)
        output = []
        for i in range(1, num_center):
            sep_local_instance_mask = (marker == i).astype(np.uint8)
            if np.max(sep_local_instance_mask) == 1:
                sep_local_instance_contours, _ = cv2.findContours(sep_local_instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                sep_local_instance_bbox = cv2.boundingRect(sep_local_instance_contours[0])
                sep_instance_mask = global2part_image(sep_local_instance_mask, sep_local_instance_bbox)
                sep_instance_contour = part2global_contour(sep_local_instance_contours, instance_bbox)[0]
                sep_instance_bbox = part2global_bbox(sep_local_instance_bbox, instance_bbox)
                output.append((sep_instance_mask, sep_instance_contour, sep_instance_bbox))

        return output


def draw_single_mask(part_points, bbox, value):
    temp_mask = np.zeros((bbox[3], bbox[2]), dtype=np.uint8)
    cv2.fillPoly(temp_mask, [part_points], 1)
    temp_mask = temp_mask.astype(int) * value
    return temp_mask, bbox


def draw_single_contour(part_points, bbox, value):
    temp_mask = np.zeros((bbox[3], bbox[2]), dtype=np.uint8)
    temp_mask = cv2.polylines(temp_mask, [part_points], True, color=int(value), thickness=1)
    return temp_mask, bbox


DEF_PARAMS = {
    'linewidth': 3,
    'key_linewidth': 5,
}
MPL_RC = {
    'axes.facecolor': 'black',
    'axes.labelcolor': 'white',
    'axes.labelsize': 26,
    'axes.labelweight': 10,
    'axes.linewidth': DEF_PARAMS['linewidth'],
    'boxplot.boxprops.color': 'white',
    'boxplot.boxprops.linewidth': DEF_PARAMS['linewidth'],
    'boxplot.capprops.color': 'white',
    'boxplot.capprops.linewidth': DEF_PARAMS['linewidth'],
    'boxplot.flierprops.markeredgecolor': 'none',
    'boxplot.flierprops.markerfacecolor': 'blue',
    'boxplot.flierprops.markersize': 8,
    'boxplot.meanline': True,
    'boxplot.meanprops.linewidth': DEF_PARAMS['key_linewidth'],
    'boxplot.medianprops.linewidth': DEF_PARAMS['key_linewidth'],
    'boxplot.showmeans': True,
    'boxplot.whiskerprops.color': 'white',
    'boxplot.whiskerprops.linewidth': DEF_PARAMS['linewidth'],
    'figure.constrained_layout.use': True,
    'figure.constrained_layout.w_pad': 0.0,
    'figure.dpi': 100,
    'figure.edgecolor': 'black',
    'figure.facecolor': 'black',
    'font.family': ['serif'],
    'font.weight': 10,
    'hist.bins': 20,
    'legend.fontsize': 14,
    'legend.labelcolor': 'white',
    'lines.linewidth': DEF_PARAMS['key_linewidth'],
    'xtick.color': 'white',
    'xtick.labelsize': 26,
    'xtick.major.width': DEF_PARAMS['linewidth'],
    'xtick.minor.width': 0.5 * DEF_PARAMS['linewidth'],
    'ytick.color': 'white',
    'ytick.labelsize': 26,
    'ytick.major.size': 20,
    'ytick.major.width': DEF_PARAMS['linewidth'],
    'ytick.minor.size': 10,
    'ytick.minor.width': 0.5 * DEF_PARAMS['linewidth'],
    'figure.titlesize': 32,
    'figure.titleweight': 10,
    'axes.titlecolor': 'white',
    'axes.titlesize': 26,
    'axes.titleweight': 10,
}


def draw_legend(image, anchor, size, statistic_dict, quantity, mode, colormap, labelparam, landscape_mode=False, rc=MPL_RC):
    max_quantity = statistic_dict[quantity]['max']
    min_quantity = statistic_dict[quantity]['min']

    if quantity == 'area':
        title = 'Area(mm${^2}$)'
    elif quantity == 'volume':
        title = 'Vol.(mm${^3}$)'
    elif quantity == 'length':
        title = 'Len.(mm)'

    fig = Figure(figsize=(size[1] / rc['figure.dpi'], size[0] / rc['figure.dpi']), layout='tight')
    axes = fig.gca()
    if mode == 'linear':
        norm = mpl.colors.Normalize(vmin=min_quantity, vmax=max_quantity)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
        cbar = fig.colorbar(mapper, cax=axes, orientation='vertical')
    elif mode == 'log':
        norm = mpl.colors.LogNorm(vmin=min_quantity, vmax=max_quantity)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
        cbar = fig.colorbar(mapper, cax=axes, orientation='vertical')
        axes.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=[0.2, 0.5, 1.0]))
        axes.yaxis.set_major_formatter(mpl.ticker.LogFormatter(
            base=10, labelOnlyBase=labelparam['only_base'], minor_thresholds=labelparam['minor_thresh']))
    axes.spines[:].set_color('white')
    axes.spines[:].set_linewidth(rc['axes.linewidth'])
    axes.set_xlabel(str(round(min_quantity, 2)))
    axes.set_title(str(round(max_quantity)))
    axes.yaxis.set_ticks_position('right')
    if landscape_mode:
        axes.yaxis.set_ticks_position('left')
        axes.yaxis.set_label_position('left')
        axes.set_ylabel(title, fontsize=mpl.rcParams['figure.titlesize'], rotation=-90, verticalalignment='top')
        axes.yaxis.set_tick_params(labelrotation=-90)
    else:
        fig.suptitle(title, x=0.02, ha='left', color='white')

    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(-0)
    legend_image = cv2.imdecode(np.asarray(bytearray(buf.read())), cv2.IMREAD_COLOR)

    image[anchor[0]: anchor[0] + size[0],
    anchor[1]: anchor[1] + size[1], :
    ] = legend_image[:, :, :]


def draw_boxplot(image, anchor, size, quantity_series, label, yaxis_pos='left', log_yaxis=False, landscape_mode=False, rc=MPL_RC):
    fig = Figure(figsize=(size[1] / rc['figure.dpi'], size[0] / rc['figure.dpi']))
    axes = fig.gca()
    axes.spines[:].set_color('white')
    axes.spines[:].set_linewidth(rc['axes.linewidth'])
    axes.yaxis.set_ticks_position(yaxis_pos)
    if log_yaxis:
        axes.set_yscale('log')
    axes.boxplot(quantity_series.values, labels=[label], widths=0.75)
    if landscape_mode:
        axes.set_xticks([])
        axes.set_ylabel(label, rotation=-90, verticalalignment='top')
        axes.yaxis.set_tick_params(labelrotation=-90)

    buf = BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = cv2.imdecode(np.asarray(bytearray(buf.read())), cv2.IMREAD_COLOR)

    image[
    anchor[0]: anchor[0] + size[0], anchor[1]: anchor[1] + size[1], :
    ] = img[:, :, :]


def draw_vhistogram(image, anchor, size, quantity_series, label, yaxis_pos='left', landscape_mode=False, rc=MPL_RC):
    fig = Figure(figsize=(size[1] / rc['figure.dpi'], size[0] / rc['figure.dpi']))
    axes = fig.gca()
    axes.spines[:].set_color('white')
    axes.spines[:].set_linewidth(rc['axes.linewidth'])
    axes.yaxis.set_ticks_position(yaxis_pos)
    axes.hist(quantity_series.values, orientation='horizontal', rwidth=0.9)
    axes.set_xlabel(label)
    if landscape_mode:
        axes.yaxis.set_tick_params(labelrotation=-90)

    buf = BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = cv2.imdecode(np.asarray(bytearray(buf.read())), cv2.IMREAD_COLOR)

    image[
    anchor[0]: anchor[0] + size[0], anchor[1]: anchor[1] + size[1], :
    ] = img[:, :, :]


def draw_grain_dist_curve(image, anchor, size, grain_dist_dict, grain_dist_parameters, landscape_mode=False, rc=MPL_RC):
    if landscape_mode:
        figsize1 = (int(size[0] * 0.6) / rc['figure.dpi'], size[1] / rc['figure.dpi'])
        figsize2 = ((size[0] - int(size[0] * 0.6) - 30) / rc['figure.dpi'], size[1] / rc['figure.dpi'])
    else:
        figsize1 = (size[1] / rc['figure.dpi'], int(size[0] * 0.6) / rc['figure.dpi'])
        figsize2 = (size[1] / rc['figure.dpi'], (size[0] - int(size[0] * 0.6) - 30) / rc['figure.dpi'])

    y = [0, ] + [index for index in grain_dist_dict.index]
    x_area = [0.0, ] + [grain_dist_dict['cum_area_percentage'].loc[index] for index in grain_dist_dict.index]
    x_volume = [0.0, ] + [grain_dist_dict['cum_volume_percentage'].loc[index] for index in grain_dist_dict.index]

    fig = Figure(figsize=figsize1)
    axes = fig.gca()
    axes.spines[:].set_color('white')
    axes.spines[:].set_linewidth(rc['axes.linewidth'])
    if landscape_mode:
        axes.set_ylabel('Cum. Prop.')
        axes.set_yticks([0.0, 0.5, 1.0], labels=['0', '0.5', '1'])
        axes.set_xlabel('Length (mm)')
        # axes.set_xscale('log')
        axes.tick_params(axis='x', direction='in')
        axes.plot(y[:-1], x_area[:-1], color='blue', label='Area')
        axes.plot(y[:-1], x_volume[:-1], color='orange', label='Volume')
        axes.legend()
    else:
        axes.set_xlabel('Cum. Prop.')
        axes.xaxis.set_label_position('top')
        axes.xaxis.set_ticks_position('top')
        axes.set_xticks([0.0, 0.5, 1.0], labels=['0', '0.5', '1'])
        axes.set_ylabel('Length (mm)')
        # axes.set_yscale('log')
        axes.tick_params(axis='y', direction='in')
        axes.plot(x_area[:-1], y[:-1], color='blue', label='Area')
        axes.plot(x_volume[:-1], y[:-1], color='orange', label='Volume')
        axes.legend()

    buf = BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = cv2.imdecode(np.asarray(bytearray(buf.read())), cv2.IMREAD_COLOR)
    if landscape_mode:
        img = np.rot90(img, k=-1)

    image[
    anchor[0]: anchor[0] + int(size[0] * 0.6),
    anchor[1]: anchor[1] + size[1], :
    ] = img[:, :, :]

    fig2 = Figure(figsize=figsize2)
    axes1, axes2 = fig2.subplots(1, 2)
    axes1.spines[:].set_color('white')
    axes1.spines[:].set_linewidth(rc['axes.linewidth'])
    scatter_cu = axes1.scatter(
        [1, 2], [grain_dist_parameters['area']['Cu'], grain_dist_parameters['volume']['Cu']],
        s=48, c=['blue', 'orange'], marker='o'
    )
    scatter_cc = axes1.scatter(
        [1, 2], [grain_dist_parameters['area']['Cc'], grain_dist_parameters['volume']['Cc']],
        s=48, c=['blue', 'orange'], marker='X'
    )
    scatter_ci = axes1.scatter(
        [1, 2], [grain_dist_parameters['area']['CI'], grain_dist_parameters['volume']['CI']],
        s=48, c=['blue', 'orange'], marker='P'
    )
    axes1.legend([scatter_cu, scatter_cc, scatter_ci], ['Cu', 'Cc', 'CI'], fontsize=10)
    axes1.set_xticks([1, 2], ['Area', 'Vol.'])
    axes1.tick_params(axis='y', direction='in', length=3.5, labelsize=18)
    axes1.tick_params(axis='x', direction='in', labelsize=18)
    ylim1 = axes1.get_ylim()
    # axes1.set_ylim(bottom=ylim1[0], top=ylim1[0] + 1.5 * (ylim1[1] - ylim1[0]))
    axes1.set_xlim(left=0.5, right=2.5)

    axes2.spines[:].set_color('white')
    axes2.spines[:].set_linewidth(rc['axes.linewidth'])
    scatter_d10 = axes2.scatter(
        [1, 2], [grain_dist_parameters['area']['D10'], grain_dist_parameters['volume']['D10']],
        s=48, c=['blue', 'orange'], marker='o'
    )
    scatter_d30 = axes2.scatter(
        [1, 2], [grain_dist_parameters['area']['D30'], grain_dist_parameters['volume']['D30']],
        s=48, c=['blue', 'orange'], marker='X'
    )
    scatter_d50 = axes2.scatter(
        [1, 2], [grain_dist_parameters['area']['D50'], grain_dist_parameters['volume']['D50']],
        s=48, c=['blue', 'orange'], marker='P'
    )
    scatter_d60 = axes2.scatter(
        [1, 2], [grain_dist_parameters['area']['D60'], grain_dist_parameters['volume']['D60']],
        s=48, c=['blue', 'orange'], marker='D'
    )
    axes2.legend([scatter_d10, scatter_d30, scatter_d50, scatter_d60], ['D10', 'D30', 'D50', 'D60'],
                 loc='upper center', fontsize=10)
    axes2.set_xticks([1, 2], ['Area', 'Vol.'])
    axes2.tick_params(axis='y', direction='in', length=3.5, labelsize=18)
    axes2.tick_params(axis='x', direction='in', labelsize=18)
    ylim2 = axes2.get_ylim()
    axes2.set_ylim(bottom=ylim2[0], top=ylim2[0] + 1.5 * (ylim2[1] - ylim2[0]))
    axes2.set_xlim(left=0.5, right=2.5)

    buf = BytesIO()
    fig2.savefig(buf)
    buf.seek(0)
    img = cv2.imdecode(np.asarray(bytearray(buf.read())), cv2.IMREAD_COLOR)
    if landscape_mode:
        img = np.rot90(img, k=-1)

    image[
    anchor[0] + int(size[0] * 0.6) + 30: anchor[0] + size[0],
    anchor[1]: anchor[1] + size[1], :
    ] = img[:, :, :]

# EOF

