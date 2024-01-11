import random as rand
import numpy as np
import colorsys
from .config import _C


_default_config = _C.clone()


class Color:

    @classmethod
    def create(cls, color, config=_default_config):
        """
        Creates color class

        string - generates color from hex
        tuple and values between [0, 1] - generates from hls
        tuple and values between [0, 255] - generates from rgb

        :param color: tuple, list, str
        :returns: color class
        """
        if isinstance(color, str):
            return cls(hex=color, config=config)
        
        if isinstance(color, (list, tuple)):
            if np.any(np.array(color) > 1):
                return cls(rgb=color, config=config)
            return cls(hls=color, config=config)
        
        if isinstance(color, Color):
            return color

        return cls().random(config=config)

    @classmethod
    def random(cls, h=None, l=None, s=None, config=_default_config):
        """
        Generates a random color

        :param l: range for lightness
        :type l: tuple
        :param h: range for hue
        :type h: tuple
        :param s: range for saturation
        :type s: tuple
        :returns: randomly generated color 
        :rtype: :class:`Color`
        """
        if not h:
            h = config.COLOR.DEFAULT_RANDOM_HUE_RANGE
        if not l:
            l = config.COLOR.DEFAULT_RANDOM_LIGHTNESS_RANGE
        if not s:
            s = config.COLOR.DEFAULT_RANDOM_SATURATION_RANGE
        h = rand.uniform(h[0], h[1])
        l = rand.uniform(l[0], l[1])
        s = rand.uniform(s[0], s[1])
        return cls(hls=(h, l, s), config=config)
    
    def __init__(self, hls=None, rgb=None, hex=None, config=_default_config):
        self._hls = hls
        self._rgb = rgb
        self._hex = hex
        self.config = config
    
    @property
    def hex(self):
        """
        Hex representation of color
        """
        if not self._hex:
            r, g, b = self.rgb
            self._hex = '#%02x%02x%02x' % (r, g, b)

        return self._hex 

    @property
    def hls(self):
        """
        HLS representation of color
        """
        if not self._hls:
            self._hls = colorsys.rgb_to_hls(*[i/255 for i in self.rgb])
                        
        return self._hls
    
    @property
    def rgb(self):
        """
        RGB representation of color
        """
        if not self._rgb:
            if self._hex:
                h = self.hex.lstrip('#')
                self._rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
            else:
                self._rgb = tuple([int(i*255) for i in colorsys.hls_to_rgb(*self.hls)])

        return self._rgb
    

__all__ = ['Color']
