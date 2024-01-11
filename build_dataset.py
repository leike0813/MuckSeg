import argparse
import re
from pathlib import Path
from data.data_preprocess import *


def rotation_degree_parser(string):
    seq_expr = re.compile(r'^[\(\[]\s*([-+]?\d*\.?\d*)\s*,?\s*([-+]?\d*\.?\d*)?\s*[\)\]]$')
    result = re.findall(seq_expr, string)
    if len(result) == 1:
        deg_lb = float(result[0][0])
        assert deg_lb >= -360.0 and deg_lb <= 360.0
        if result[0][1] != '':
            deg_ub = float(result[0][1])
            assert deg_ub >= max(-360.0, deg_lb) and deg_ub <= 360.0
            return deg_lb, deg_ub
        else:
            assert deg_lb >= 0.0 and deg_lb <= 360.0
            return -deg_lb, deg_lb
    else:
        deg = float(string)
        assert deg >= 0 and deg <= 360.0
        return -deg, deg


def ratio_range_check(string):
    expr = re.compile(r'^[\(\[]\s*([-+]?\d*\.?\d*)\s*,?\s*([-+]?\d*\.?\d*)?\s*[\)\]]$')
    result = re.findall(expr, string)
    ratio = (float(result[0][0]), float(result[0][1]))
    assert ratio[0] > 0 and ratio[1] > ratio[0]
    return ratio


def possibility_check(string):
    p = float(string)
    if p < 0 or p > 1:
        raise ValueError
    return p


def path_check(string):
    path = Path(string)
    assert path.is_dir() and path.exists()
    return path


def parse_option():
    parser = argparse.ArgumentParser('MuckSeg image data preprocess script')
    parser.add_argument('--data-path', type=path_check, required=True, help='path to image data')
    parser.add_argument('--category-name', type=str, default='Muck', help='category name')
    parser.add_argument('--stages', type=int, required=True,
                        choices=[1, 2, 3], nargs='+',
                        help='stages contained in intended preprocess procedure')
    parser.add_argument('--stage3-mode', type=str, default='statistic_partial',
                        choices=['statistic_partial', 'statistic', 'pack_tensor'],
                        help='mode of stage 3: image statistic and tensor packing')
    parser.add_argument('--image-size', type=int, required=True, help='target image size')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite exist files')
    parser.add_argument('--num-repeats', type=int, nargs='+', required=True,
                        help="times to repeating data augmentation pipeline")
    parser.add_argument('--rotation-degrees', type=rotation_degree_parser, default=(-30, 30),
                        help='range of random rotation degrees')
    parser.add_argument('--distortion-scale', type=possibility_check, default=0.5,
                        help='distortion scale of random perspective transformation')
    parser.add_argument('--p-perspective', type=possibility_check, default=1.0,
                        help='possibility to perform random perspective transformation')
    parser.add_argument('--p-hflip', type=possibility_check, default=[0.5], nargs='+',
                        help='possibility to perform random horizontal flip')
    parser.add_argument('--p-vflip', type=possibility_check, default=[0.5], nargs='+',
                        help='possibility to perform random vertical flip')
    parser.add_argument('--scale-relative', type=ratio_range_check, default=(0.5, 1.2),
                        help='range of scale factor relative to target image size for random resized cut')
    parser.add_argument('--aspect-ratio', type=ratio_range_check, default=(0.5, 2),
                        help='range of aspect ratio for random resized cut')
    # return parser
    args, unparsed = parser.parse_known_args()

    return args


if __name__ == '__main__':
    args = parse_option()
    output_fld = args.data_path
    if 1 in args.stages:
        output_fld = preprocess_stage1(
            fld=args.data_path,
            num_repeats=args.num_repeats[0],
            rotation_degrees=args.rotation_degrees,
            distortion_scale=args.distortion_scale,
            p_perspective=args.p_perspective,
            p_hflip=args.p_hflip[0],
            p_vflip=args.p_vflip[0],
            overwrite=args.overwrite,
            category_name=args.category_name
        )
    if 2 in args.stages:
        output_fld = preprocess_stage2(
            fld=args.data_path,
            num_repeats=args.num_repeats[1 if (len(args.num_repeats) >= 2) and (1 in args.stages) else 0],
            size=(args.image_size, args.image_size),
            scale_relative=args.scale_relative,
            aspect_ratio=args.aspect_ratio,
            p_hflip=args.p_hflip[1 if (len(args.p_hflip) >= 2) and (1 in args.stages) else 0],
            p_vflip=args.p_vflip[1 if (len(args.p_vflip) >= 2) and (1 in args.stages) else 0],
            overwrite=args.overwrite,
            category_name=args.category_name
        )
    if 3 in args.stages:
        preprocess_stage3(
            fld=output_fld,
            mode=args.stage3_mode,
            category_name=args.category_name
        )

# EOF