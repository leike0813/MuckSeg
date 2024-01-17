import os
import random
from pathlib import Path
import multiprocessing as mul
from PIL import Image
from tqdm import tqdm
import json
import torch
import lib.pytorch_framework.transforms as CT


__all__ = [
    'preprocess_stage1',
    'preprocess_stage2',
    'preprocess_stage3',
]


def stage1_atom(img_path, category_name, num_repeats, croppers, transform_cutter, num_proc, overwrite, preprocessed,
                bound_simple, region_simple, gt_boundary_fld, gt_region_fld, gt_boundary_simple_fld, gt_region_simple_fld,
                output_fld, output_gt_boundary_fld, output_gt_region_fld, output_validation_fld,
                output_gt_boundary_simple_fld, output_gt_region_simple_fld):
    if (not overwrite and img_path.name not in preprocessed['preprocessed']) or overwrite:
        img_name = img_path.stem

        img_gt_boundary_path = gt_boundary_fld / '{img}_{cat}_boundary.png'.format(img=img_name, cat=category_name)
        img_gt_region_path = gt_region_fld / '{img}_{cat}_region.png'.format(img=img_name, cat=category_name)
        img_validation = Image.open(img_path)
        img = img_validation.convert('L')
        img_gt_boundary = Image.open(img_gt_boundary_path).convert('L')
        img_gt_region = Image.open(img_gt_region_path).convert('L')
        imgs_to_process = [img, img_gt_boundary, img_gt_region, img_validation]
        if bound_simple:
            img_gt_boundary_simple_path = gt_boundary_simple_fld / '{img}_{cat}_boundary_simple.png'.format(
                img=img_name,
                cat=category_name)
            img_gt_boundary_simple = Image.open(img_gt_boundary_simple_path).convert('L')
            imgs_to_process.append(img_gt_boundary_simple)
        if region_simple:
            img_gt_region_simple_path = gt_region_simple_fld / '{img}_{cat}_region_simple.png'.format(img=img_name,
                                                                                                      cat=category_name)
            img_gt_region_simple = Image.open(img_gt_region_simple_path).convert('L')
            imgs_to_process.append(img_gt_region_simple)

        image_sets = [croppers[i](imgs_to_process) for i in range(3)]
        for j in range(3):
            for k in range(num_repeats):
                img_output_name = '{img}_S{ns}_R{nr}'.format(img=img_name, ns=j + 1, nr=k)
                img_output_path = output_fld / '{img}.jpg'.format(img=img_output_name)
                img_output_gt_boundary_path = output_gt_boundary_fld / '{img}_{cat}_boundary.png'.format(
                    img=img_output_name, cat=category_name)
                img_output_gt_region_path = output_gt_region_fld / '{img}_{cat}_region.png'.format(img=img_output_name,
                                                                                                   cat=category_name)
                img_output_validation_path = output_validation_fld / '{img}_validation.jpg'.format(img=img_output_name)
                if bound_simple:
                    img_output_gt_boundary_simple_path = output_gt_boundary_simple_fld / '{img}_{cat}_boundary_simple.png'.format(
                        img=img_output_name, cat=category_name)
                if region_simple:
                    img_output_gt_region_simple_path = output_gt_region_simple_fld / '{img}_{cat}_region_simple.png'.format(
                        img=img_output_name, cat=category_name)

                transformed = transform_cutter(image_sets[j])
                transformed[0].save(img_output_path)
                transformed[1].save(img_output_gt_boundary_path),
                transformed[2].save(img_output_gt_region_path)
                transformed[3].save(img_output_validation_path)
                if num_proc == 5:
                    transformed[4].save(
                        img_output_gt_boundary_simple_path if bound_simple else img_output_gt_region_simple_path)
                elif num_proc == 6:
                    transformed[4].save(img_output_gt_boundary_simple_path)
                    transformed[5].save(img_output_gt_region_simple_path)
        return img_path


def preprocess_stage1(fld, num_repeats, rotation_degrees, distortion_scale,
                      p_perspective, p_hflip, p_vflip, overwrite=False, category_name='Muck'):
    if not isinstance(fld, Path):
        fld = Path(fld)
    if not overwrite:
        try:
            with open(fld / '.preprocessed_stage1.json') as f:
                preprocessed = json.load(f)
        except Exception:
            preprocessed = {'preprocessed': []}
    else:
        preprocessed = {'preprocessed': []}
    gt_boundary_fld = fld / 'GT_Boundary'
    gt_region_fld = fld / 'GT_Region'
    gt_boundary_simple_fld = fld / 'GT_Boundary_Simple'
    gt_region_simple_fld = fld / 'GT_Region_Simple'
    output_fld = fld / 'stage1_output'
    if not output_fld.exists():
        os.makedirs(output_fld)
    output_gt_boundary_fld = output_fld / 'GT_Boundary'
    output_gt_region_fld = output_fld / 'GT_Region'
    output_gt_boundary_simple_fld = output_fld / 'GT_Boundary_Simple'
    output_gt_region_simple_fld = output_fld / 'GT_Region_Simple'
    output_validation_fld = output_fld / 'validation'
    if not output_gt_boundary_fld.exists():
        os.makedirs(output_gt_boundary_fld)
    if not output_gt_region_fld.exists():
        os.makedirs(output_gt_region_fld)
    if not output_validation_fld.exists():
        os.makedirs(output_validation_fld)
    num_proc = 4
    if gt_boundary_simple_fld.exists():
        bound_simple = True
        num_proc += 1
        if not output_gt_boundary_simple_fld.exists():
            os.makedirs(output_gt_boundary_simple_fld)
    else:
        bound_simple = False
    if gt_region_simple_fld.exists():
        region_simple = True
        num_proc += 1
        if not output_gt_region_simple_fld.exists():
            os.makedirs(output_gt_region_simple_fld)
    else:
        region_simple = False

    img_paths = [f for f in fld.glob('*.jpg')]
    total_images = len(img_paths)
    croppers = [CT.Crop(0, 300, 1365, 1365), CT.Crop(1365, 300, 1365, 1365), CT.Crop(2730, 300, 1366, 1365)]
    transform_cutter = CT.RandomTransformAndCutoff(
        degrees=rotation_degrees, distortion_scale=distortion_scale,
        p_perspective=p_perspective, p_hflip=p_hflip, p_vflip=p_vflip, mask=[3])
    print('Stage 1 in progress...')
    img_infos = []
    for i in range(total_images):
        img_infos.append((img_paths[i], category_name, num_repeats, croppers, transform_cutter, num_proc, overwrite, preprocessed,
                bound_simple, region_simple, gt_boundary_fld, gt_region_fld, gt_boundary_simple_fld, gt_region_simple_fld,
                output_fld, output_gt_boundary_fld, output_gt_region_fld, output_validation_fld,
                output_gt_boundary_simple_fld, output_gt_region_simple_fld))
    pool = mul.Pool(processes=os.cpu_count())
    res_async = pool.starmap_async(stage1_atom, img_infos)
    res_async.wait()
    ret = res_async.get()
    pool.close()
    pool.join()

    for i in range(total_images):
        img_path = img_paths[i]
        if (not overwrite and img_path.name not in preprocessed['preprocessed']) or overwrite:
            preprocessed['preprocessed'].append(img_path.name)

    with open(fld / '.preprocessed_stage1.json', 'w') as f:
        json.dump(preprocessed, f)

    return output_fld


def stage2_atom(img_path, category_name, num_repeats, transformer, num_proc, overwrite, preprocessed,
                bound_simple, region_simple, gt_boundary_fld, gt_region_fld, gt_boundary_simple_fld, gt_region_simple_fld,
                output_fld, output_gt_boundary_fld, output_gt_region_fld,
                output_gt_boundary_simple_fld, output_gt_region_simple_fld):
    if (not overwrite and img_path.name not in preprocessed['preprocessed']) or overwrite:
        img_name = img_path.stem
        img_gt_boundary_path = gt_boundary_fld / '{img}_{cat}_boundary.png'.format(img=img_name, cat=category_name)
        img_gt_region_path = gt_region_fld / '{img}_{cat}_region.png'.format(img=img_name, cat=category_name)
        img = Image.open(img_path).convert('L')
        img_gt_boundary = Image.open(img_gt_boundary_path).convert('L')
        img_gt_region = Image.open(img_gt_region_path).convert('L')
        imgs_to_process = [img, img_gt_boundary, img_gt_region]
        if bound_simple:
            img_gt_boundary_simple_path = gt_boundary_simple_fld / '{img}_{cat}_boundary_simple.png'.format(
                img=img_name,
                cat=category_name)
            img_gt_boundary_simple = Image.open(img_gt_boundary_simple_path).convert('L')
            imgs_to_process.append(img_gt_boundary_simple)
        if region_simple:
            img_gt_region_simple_path = gt_region_simple_fld / '{img}_{cat}_region_simple.png'.format(img=img_name,
                                                                                                      cat=category_name)
            img_gt_region_simple = Image.open(img_gt_region_simple_path).convert('L')
            imgs_to_process.append(img_gt_region_simple)

        for k in range(num_repeats):
            img_output_name = '{img}_{nc}'.format(img=img_name, nc=str(k).zfill(3))
            img_output_path = output_fld / '{img}.jpg'.format(img=img_output_name)
            img_output_gt_boundary_path = output_gt_boundary_fld / '{img}_{cat}_boundary.png'.format(
                img=img_output_name, cat=category_name)
            img_output_gt_region_path = output_gt_region_fld / '{img}_{cat}_region.png'.format(
                img=img_output_name, cat=category_name)
            if bound_simple:
                img_output_gt_boundary_simple_path = output_gt_boundary_simple_fld / '{img}_{cat}_boundary_simple.png'.format(
                    img=img_output_name, cat=category_name)
            if region_simple:
                img_output_gt_region_simple_path = output_gt_region_simple_fld / '{img}_{cat}_region_simple.png'.format(
                    img=img_output_name, cat=category_name)

            transformed = transformer(imgs_to_process)
            transformed[0].save(img_output_path)
            transformed[1].save(img_output_gt_boundary_path),
            transformed[2].save(img_output_gt_region_path)
            if num_proc == 4:
                transformed[3].save(
                    img_output_gt_boundary_simple_path if bound_simple else img_output_gt_region_simple_path)
            elif num_proc == 5:
                transformed[3].save(img_output_gt_boundary_simple_path)
                transformed[4].save(img_output_gt_region_simple_path)


def preprocess_stage2(fld, num_repeats, size, scale_relative, aspect_ratio,
                      p_hflip, p_vflip, overwrite=False, category_name='Muck'):
    if not isinstance(fld, Path):
        fld = Path(fld)
    image_fld = fld / 'stage1_output'
    if not image_fld.exists():
        print('Cannot find stage1 outputs.')
        return
    gt_boundary_fld = image_fld / 'GT_Boundary'
    gt_region_fld = image_fld / 'GT_Region'
    gt_boundary_simple_fld = image_fld / 'GT_Boundary_Simple'
    gt_region_simple_fld = image_fld / 'GT_Region_Simple'
    output_fld = fld / 'stage2_output_{w}x{h}'.format(w=size[0], h=size[1])
    if not overwrite:
        try:
            with open(output_fld / '.preprocessed_stage2.json') as f:
                preprocessed = json.load(f)
        except Exception:
            preprocessed = {'preprocessed': []}
    else:
        preprocessed = {'preprocessed': []}
    output_gt_boundary_fld = output_fld / 'GT_Boundary'
    output_gt_region_fld = output_fld / 'GT_Region'
    output_gt_boundary_simple_fld = output_fld / 'GT_Boundary_Simple'
    output_gt_region_simple_fld = output_fld / 'GT_Region_Simple'
    if not output_fld.exists():
        os.makedirs(output_fld)
    if not output_gt_boundary_fld.exists():
        os.makedirs(output_gt_boundary_fld)
    if not output_gt_region_fld.exists():
        os.makedirs(output_gt_region_fld)
    num_proc = 3
    if gt_boundary_simple_fld.exists():
        bound_simple = True
        num_proc += 1
        if not output_gt_boundary_simple_fld.exists():
            os.makedirs(output_gt_boundary_simple_fld)
    else:
        bound_simple = False
    if gt_region_simple_fld.exists():
        region_simple = True
        num_proc += 1
        if not output_gt_region_simple_fld.exists():
            os.makedirs(output_gt_region_simple_fld)
    else:
        region_simple = False

    img_paths = [f for f in image_fld.glob('*.jpg')]
    total_images = len(img_paths)
    transformer = CT.Compose([
        CT.RandomResizedCrop_Relative(size, scale_relative=scale_relative, ratio=aspect_ratio),
        CT.RandomHorizontalFlip(p=p_hflip),
        CT.RandomVerticalFlip(p=p_vflip),
    ])
    print('Stage 2 in progress...')
    img_infos = []
    for i in range(total_images):
        img_infos.append((img_paths[i], category_name, num_repeats, transformer, num_proc, overwrite, preprocessed,
                bound_simple, region_simple, gt_boundary_fld, gt_region_fld, gt_boundary_simple_fld, gt_region_simple_fld,
                output_fld, output_gt_boundary_fld, output_gt_region_fld,
                output_gt_boundary_simple_fld, output_gt_region_simple_fld))
    pool = mul.Pool(processes=os.cpu_count())
    res_async = pool.starmap_async(stage2_atom, img_infos)
    res_async.wait()
    ret = res_async.get()
    pool.close()
    pool.join()

    for i in range(total_images):
        img_path = img_paths[i]
        if (not overwrite and img_path.name not in preprocessed['preprocessed']) or overwrite:
            preprocessed['preprocessed'].append(img_path.name)

    with open(image_fld / '.preprocessed_stage2.json', 'w') as f:
        json.dump(preprocessed, f)

    return output_fld


def preprocess_stage3(fld, mode='statistic_partial', category_name='Muck'):
    def convert_to_tensor(img_path, tensors=None, gt_include=False):
        img_name = img_path.stem
        img_gt_boundary_path = gt_boundary_fld / '{img}_{cat}_boundary.png'.format(img=img_name, cat=category_name)
        img_gt_region_path = gt_region_fld / '{img}_{cat}_region.png'.format(img=img_name, cat=category_name)
        img_gt_boundary_simple_path = gt_boundary_simple_fld / '{img}_{cat}_boundary_simple.png'.format(img=img_name, cat=category_name)
        img_gt_region_simple_path = gt_region_simple_fld / '{img}_{cat}_region_simple.png'.format(img=img_name, cat=category_name)
        img = Image.open(img_path).convert('L')
        convert_list = [img]
        num_proc = 1
        if gt_include:
            img_gt_boundary = Image.open(img_gt_boundary_path).convert('L')
            img_gt_region = Image.open(img_gt_region_path).convert('L')
            convert_list.append(img_gt_boundary)
            convert_list.append(img_gt_region)
            num_proc += 2
            if bound_simple:
                img_gt_boundary_simple = Image.open(img_gt_boundary_simple_path).convert('L')
                convert_list.append(img_gt_boundary_simple)
                num_proc += 1
            if region_simple:
                img_gt_region_simple = Image.open(img_gt_region_simple_path).convert('L')
                convert_list.append(img_gt_region_simple)
                num_proc += 1
        transformed = trans_tensor(convert_list)
        if not tensors:
            tensors = [transformed[i].unsqueeze(0) for i in range(num_proc)]
        else:
            tensors = [torch.cat((tensors[i], transformed[i].unsqueeze(0)), dim=0) for i in range(num_proc)]
        return tensors

    if not isinstance(fld, Path):
        fld = Path(fld)
    gt_boundary_fld = fld / 'GT_Boundary'
    gt_region_fld = fld / 'GT_Region'
    gt_boundary_simple_fld = fld / 'GT_Boundary_Simple'
    gt_region_simple_fld = fld / 'GT_Region_Simple'

    bound_simple = True if gt_boundary_simple_fld.exists() else False
    region_simple = True if gt_region_simple_fld.exists() else False

    img_paths = [f for f in fld.glob('*.jpg')]
    total_images = len(img_paths)
    trans_tensor = CT.ToTensor()
    if mode == 'statistic_partial':
        total_images_statistic = min(total_images, 200)
        print('Calculating partial statistics of images.')
        for i in tqdm(range(total_images_statistic)):
            idx = random.randint(0, total_images - 1 - i)
            img_path = img_paths.pop(idx)
            if i == 0:
                tensors = None
            tensors = convert_to_tensor(img_path, tensors)
    elif mode == 'statistic' or mode == 'pack_tensor':
        print('Converting images to tensor.' if mode == 'pack_tensor' else 'Calculating statistics of images.')
        for i in tqdm(range(total_images)):
            img_path = img_paths[i]
            if i == 0:
                tensors = None
            tensors = convert_to_tensor(img_path, tensors, gt_include=True if mode == 'pack_tensor' else False)
    else:
        raise NotImplementedError

    img_tensor = tensors[0]
    img_tensor.to('cuda')
    std, mean = torch.std_mean(img_tensor, dim=(0, 2, 3))
    with open(fld / 'statistics.json', 'w') as f:
        json.dump({
            'mean': [i.item() for i in mean],
            'std': [i.item() for i in std]
        }, f)

    if mode == 'pack_tensor':
        output_fld = fld / 'tensor'
        if not output_fld.exists():
            os.makedirs(output_fld)

        with open(output_fld / 'statistics.json', 'w') as f:
            json.dump({
                'mean': [i.item() for i in mean],
                'std': [i.item() for i in std]
            }, f)
        print('Packing tensors.')
        torch.save(img_tensor, output_fld / 'image.pt')
        torch.save(tensors[1], output_fld / 'image_GT_Boundary.pt')
        torch.save(tensors[2], output_fld / 'image_GT_Region.pt')
        if len(tensors) == 4:
            torch.save(tensors[3], output_fld / ('image_GT_Boundary_Simple.pt' if bound_simple else 'image_GT_Region_Simple.pt'))
        elif len(tensors) == 5:
            torch.save(tensors[3], output_fld / 'image_GT_Boundary_Simple.pt')
            torch.save(tensors[4], output_fld / 'image_GT_Region_Simple.pt')

# EOF