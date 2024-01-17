from enum import IntEnum
import cv2


__all__ = [
    'KernelShape',
    'ErosionMode',
    'PredictionType',
    'VisualizationMode',
]


class KernelShape(IntEnum):
    Ellipse = cv2.MORPH_ELLIPSE
    Rectangle = cv2.MORPH_RECT
    Cross = cv2.MORPH_CROSS


class ErosionMode(IntEnum):
    Iterative = 1
    Direct = 2


class PredictionType(IntEnum):
    Possibility = 1
    Binary = 2


class VisualizationMode(IntEnum):
    Area = 1
    LogArea = 2
    Volume = 3
    LogVolume = 4
    Length = 5
    LogLength = 6
    ID = 7

    @staticmethod
    def get_quantity_name(visualization_mode):
        if visualization_mode in [VisualizationMode.Area, VisualizationMode.LogArea]:
            return 'area'
        elif visualization_mode in [VisualizationMode.Volume, VisualizationMode.LogVolume]:
            return 'volume'
        elif visualization_mode in [VisualizationMode.Length, VisualizationMode.LogLength]:
            return 'length'
        else:
            raise ValueError("Can't fetch quantity name for visualization mode: {}".format(visualization_mode))

    @staticmethod
    def get_mode(visualization_mode):
        if visualization_mode in [VisualizationMode.Area, VisualizationMode.Volume, VisualizationMode.Length]:
            return 'linear'
        elif visualization_mode in [VisualizationMode.LogArea, VisualizationMode.LogVolume, VisualizationMode.LogLength]:
            return 'log'
        else:
            raise ValueError("Can't fetch mode for visualization mode: {}".format(visualization_mode))

# EOF