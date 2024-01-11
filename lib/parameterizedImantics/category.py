from .basic import Semantic
from .color import Color
from .config import _C


_default_config = _C.clone()


class Category(Semantic):


    @classmethod
    def from_coco(cls, coco, config=_default_config):
        data = {
            'name': coco.get('name'),
            'metadata': coco.get('metadata', {}),
            'id': coco.get('id', 0),
            'parent': coco.get('supercategory'),
            'color': coco.get('color')
        }

        return cls(config=config, **data)


    def __init__(self, name, parent=None, metadata={}, id=0, color=None, config=_default_config):
        self.id = id
        self.name = name
        self.parent = None
        self.color = Color.create(color, config)

        super(Category, self).__init__(id, metadata, config)

    def coco(self, include=True):
        if self.config.COLOR.DEFAULT_EXPORT_FORMAT == 'rgb':
            color = self.color.rgb
        elif self.config.COLOR.DEFAULT_EXPORT_FORMAT == 'hls':
            color = self.color.hls
        elif self.config.COLOR.DEFAULT_EXPORT_FORMAT == 'hex':
            color = self.color.hex
        else:
            raise RuntimeError('Invalid color format: {fmt}'.format(
                fmt=self.config.COLOR.DEFAULT_EXPORT_FORMAT))

        # category = {
        #     'id': self.id,
        #     'name': self.name,
        #     'supercategory': self.parent.name if self.parent else None,
        #     'metadata': self.metadata,
        #     'color': color
        # }

        category = {
            'id': self.id,
            'name': self.name,
            'supercategory': self.parent.name if self.parent else None,
            'color': color
        } # 20230626: Modified by Joshua Reed
        if self.config.CATEGORY.KEYS.METADATA:
            category['metadata'] = self.metadata

        if include:
            return {
                'categories': [category]
            }

        return category


__all__ = ["Category"]
