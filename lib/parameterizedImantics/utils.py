import numpy as np
import os
from pathlib import Path

from .config import _C


_default_config = _C.clone()


def json_default(o):
    print(o)
    if isinstance(o, np.int64):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    
    type_name = o.__class__.__name__
    raise TypeError("Object of type {} is not JSON serializable".format(type_name))

# 20230616: Feature added by Joshua Reed
def resolve_relative_image_path(image_path, image_root):
    return Path(os.path.relpath(image_path, image_root)).as_posix()


def resolve_absolute_image_path(rel_image_path, image_root):
    return (Path(image_root) / Path(rel_image_path)).absolute().resolve().as_posix()


def replace_absolute_image_path(image_path, old_image_root, new_image_root):
    return resolve_absolute_image_path(resolve_relative_image_path(image_path, old_image_root), new_image_root)