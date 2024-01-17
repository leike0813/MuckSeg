import json
import math
import pandas as pd
from PIL import Image as PILImage
from torchvision.transforms.functional import to_tensor
from pathlib import Path
import numpy as np
import cv2
from lib.parameterizedImantics import Dataset, Image, Annotation, Category, Color, Default_Config


def write_statistics(output_dir, result_statistics):
    def quantity_statistic_columns(quantity_name):
        return [
            'max_{}'.format(quantity_name),
            'min_{}'.format(quantity_name),
            'avg_{}'.format(quantity_name),
            'median_{}'.format(quantity_name),
            'up_quartile_{}'.format(quantity_name),
            'lo_quartile_{}'.format(quantity_name),
            'total_{}'.format(quantity_name),
        ]

    def grain_dist_columns(type):
        return [
            'Cu_{}'.format(type),
            'Cc_{}'.format(type),
            'CI_{}'.format(type),
            'D10_{}'.format(type),
            'D30_{}'.format(type),
            'D50_{}'.format(type),
            'D60_{}'.format(type),
            'dist_curve_{}'.format(type),
        ]

    def grain_dist_curve_json(type, grain_dist_dict):
        return {k: v for k, v in zip(grain_dist_dict.index, grain_dist_dict['cum_{}_percentage'.format(type)])}

    quantities = ['area', 'length', 'perimeter', 'sphericity', 'slenderness', 'volume']

    output_columns = ['id', 'image_name', 'image_width', 'image_height', 'muck_count']
    for quantity in quantities:
        output_columns.extend(quantity_statistic_columns(quantity))

    output_columns.extend(grain_dist_columns('area'))
    output_columns.extend(grain_dist_columns('volume'))

    output_dict = pd.DataFrame(columns=output_columns)
    output_dict.set_index('id', drop=True, inplace=True)

    for i, (img_name, statistics) in enumerate(result_statistics.items()):
        statistic_dict = statistics[1]
        grain_dist_dict = statistics[2]
        grain_dist_parameters = statistics[3]

        record = [img_name, statistic_dict['image_width'], statistic_dict['image_height'], statistic_dict['muck_count']]
        for quantity in quantities:
            record.extend(list(statistic_dict[quantity].values()))

        record.extend(list(grain_dist_parameters['area'].values()))
        record.append(grain_dist_curve_json('area', grain_dist_dict))
        record.extend(list(grain_dist_parameters['volume'].values()))
        record.append(grain_dist_curve_json('volume', grain_dist_dict))

        output_dict.loc[i] = (record)

    output_file_path = output_dir / 'muck_statistics.xlsx'
    output_dict.to_excel(output_file_path)
    output_json_path = output_dir / 'muck_statistics.json'
    output_dict.to_json(output_json_path, orient='index')
    return output_file_path, output_json_path


def export_annotations(output_dir, result_statistics, simplify=True, simplify_eps=1):
    config = Default_Config.clone()
    config.defrost()
    config.ANNOTATION.POLYGON.EXPORT_SIMPLIFIED = simplify
    config.ANNOTATION.POLYGON.EXPORT_SIMPLIFY_EPSILON = simplify_eps
    config.freeze()

    dataset = Dataset('Muck Dataset', image_root=output_dir.as_posix(), config=config)
    category = Category('Muck', id=1, color=Color(rgb=(244, 108, 59)), config=config)
    annotation_count = 0
    for i, (img_name, statisitcs) in enumerate(result_statistics.items()):
        contour_dict = statisitcs[0]
        statistic_dict = statisitcs[1]
        image = Image(
            id=i + 1, path=(output_dir / (img_name + '.jpg')).as_posix(),
            width=statistic_dict['image_width'],
            height=statistic_dict['image_height'],
            dataset=dataset, config=config
        )
        dataset.add(image)
        for contour_index in contour_dict.index:
            annotation = Annotation(
                id=annotation_count + 1, image=image, category=category,
                polygons=[contour_dict['points'].loc[contour_index].reshape(-1)],
                config=config
            )
            dataset.add(annotation)
            annotation_count += 1

    coco_obj = dataset.coco()
    annotation_path = output_dir / 'annotations.json'
    with open(annotation_path, 'w') as f:
        json.dump(coco_obj, f)

    return annotation_path


def import_annotations(annotation_path, category='muck', orig_img_folder=None):
    if orig_img_folder:
        orig_img_folder = Path(orig_img_folder)

    with open(annotation_path, 'r') as f:
        anno_json = json.load(f)

    categories = pd.DataFrame(anno_json['categories']).ffill()
    images = pd.DataFrame(anno_json['images']).ffill()
    annotations = pd.DataFrame(anno_json['annotations']).ffill()
    categories.set_index('id', drop=True, inplace=True)
    images.set_index('id', drop=True, inplace=True)
    annotations.set_index('id', drop=True, inplace=True)

    category_found = False
    for idx in categories.index:
        if categories['name'].loc[idx].lower() == category:
            category_found = True
            category_id = idx

    if not category_found:
        raise ValueError('Could not find category {}'.format(category))

    output = {}
    for idx in images.index:
        img_name = images['file_name'].loc[idx]
        if orig_img_folder:
            try:
                orig_image = PILImage.open(orig_img_folder / img_name).convert('L')
            except FileNotFoundError:
                if 'path' in images.columns:
                    orig_image = PILImage.open(images['path'].loc[idx]).convert('L')
                else:
                    raise FileNotFoundError('Could not find original image file {}'.format(images['file_name'].loc[idx]))
        else:
            orig_image = PILImage.open(orig_img_folder / img_name).convert('L')

        orig_image = to_tensor(orig_image)

        image_annotations = annotations.loc[annotations['image_id'] == idx].loc[annotations['category_id'] == category_id]
        contour_dict = pd.DataFrame(columns=['id', 'points', 'bbox', 'center'])
        contour_dict.set_index('id', drop=True, inplace=True)
        cd_idx = 1
        for anno_idx in image_annotations.index:
            points = image_annotations['segmentation'].loc[anno_idx]
            points = np.frompyfunc(round, 1, 1)(points).reshape(-1, 1, 2).astype(int)
            if points.shape[0] > 2:
                bbox = _ = image_annotations['bbox'].loc[anno_idx]
                bbox[0] = math.floor(_[0])
                bbox[1] = math.floor(_[1])
                bbox[2] = math.ceil(_[2])
                bbox[3] = math.ceil(_[3])
                m = cv2.moments(points)
                cx = int(m['m10'] / m['m00'])
                cy = int(m['m01'] / m['m00'])
                center = (cx, cy)
                contour_dict.loc[cd_idx] = [points, bbox, center]
                cd_idx += 1

        output[img_name] = {'orig_image': orig_image, 'contour_dict': contour_dict}

    return output

# EOF