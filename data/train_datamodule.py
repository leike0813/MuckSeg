import torch
from pathlib import Path
from PIL import Image
import json
import pandas as pd
from torch.utils import data
from torchvision.transforms import ToTensor, Normalize, Compose
from lightning import LightningDataModule


class MuckSeg_DataModule(LightningDataModule):
    train_ds = None
    valid_ds = None
    test_ds = None
    _STAGE = 1

    def __init__(
            self, data_path, batch_size, config, val_volume=100, test_volume=1000, num_workers=0,
            shuffle=True, pin_memory=True, split_seed=None
    ):
        super(MuckSeg_DataModule, self).__init__()
        self.config = config
        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            raise ValueError('Invalid input data path {datapath}.'.format(datapath=data_path))
        self.batch_size = batch_size
        self.val_volume = val_volume
        self.test_volume = test_volume
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        if isinstance(split_seed, int):
            self.generator = torch.Generator().manual_seed(split_seed)
        else:
            self.generator = torch.Generator()

    def setup(self, stage):
        if stage == 'fit':
            dataset = MuckSeg_Dataset(self.data_path, self.config, self._STAGE)
            self.train_ds, self.valid_ds, self.test_ds = data.random_split(
                dataset,
                [len(dataset) - self.val_volume - self.test_volume, self.val_volume, self.test_volume],
                self.generator
            )
            print("image count in train dataset :{}".format(len(self.train_ds)))
            print("image count in validation dataset :{}".format(len(self.valid_ds)))
        if stage == 'test':
            if not self.test_ds:
                self.setup('fit')
            print("image count in test dataset :{}".format(len(self.test_ds)))

    def train_dataloader(self):
        return data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return data.DataLoader(
            dataset=self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return data.DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )


class MuckSeg_Dataset(data.Dataset):
    def __init__(self, base_fld, config, stage=1, category_name='Muck'):
        self.base_fld = Path(base_fld)
        self.config = config
        self._STAGE = stage
        self.gt_boundary_fld = self.base_fld / 'GT_Boundary'
        self.gt_region_fld = self.base_fld / 'GT_Region'
        self.gt_region_simple_fld = self.base_fld / 'GT_Region_Simple'
        self.category_name = category_name
        try:
            with open(self.base_fld / 'statistics.json', 'r') as f:
                statistics = json.load(f)
            self.image_mean = statistics['mean']
            self.image_std = statistics['std']
        except Exception:
            self.image_mean = self.config.PREDICT.IMAGE_MEAN
            self.image_std = self.config.PREDICT.IMAGE_STD
        self.normalizer = Normalize(mean=self.image_mean, std=self.image_std, inplace=True)
        self.tensor_converter = ToTensor()

        self.image_lib = pd.DataFrame(columns=['image_path', 'gt_boundary_path', 'gt_region_path', 'gt_region_simple_path'])
        self.image_count = 0
        for img_path in self.base_fld.glob('*.jpg'):
            img_name = img_path.stem
            img_gt_boundary_path = self.gt_boundary_fld / '{img}_{cat}_boundary.png'.format(img=img_name, cat=category_name)
            img_gt_region_path = self.gt_region_fld / '{img}_{cat}_region.png'.format(img=img_name, cat=category_name)
            img_gt_region_simple_path = self.gt_region_simple_fld / '{img}_{cat}_region_simple.png'.format(img=img_name, cat=category_name)
            if img_gt_boundary_path.exists() and img_gt_region_path.exists() and img_gt_region_simple_path.exists():
                self.image_lib.loc[self.image_count] = [img_path.as_posix(), img_gt_boundary_path.as_posix(),
                                                        img_gt_region_path.as_posix(), img_gt_region_simple_path.as_posix()]
                self.image_count += 1

    def __len__(self):
        return self.image_count

    def __getitem__(self, item):
        img_info = self.image_lib.loc[item]
        img = Image.open(img_info['image_path']).convert('L')
        img = self.tensor_converter(img)
        img = self.normalizer(img)
        if self._STAGE == 1:
            img_gt_simple_region = Image.open(img_info['gt_region_simple_path']).convert('L')
            img_gt_simple_region = self.tensor_converter(img_gt_simple_region)
            return img, img_gt_simple_region
        elif self._STAGE == 2:
            img_gt_boundary = Image.open(img_info['gt_boundary_path']).convert('L')
            img_gt_region = Image.open(img_info['gt_region_path']).convert('L')
            img_gt_boundary = self.tensor_converter(img_gt_boundary)
            img_gt_region = self.tensor_converter(img_gt_region)
            return img, img_gt_boundary, img_gt_region

    def denormalizer(self, x):
        assert isinstance(x, torch.Tensor), 'Input must be torch.Tensor'
        if x.ndim == 2: # assuming H, W
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3: # assuming C, H, W
            x = x.unsqueeze(0)
        elif x.ndim == 4: # assuming B, C, H, W
            pass
        else:
            raise ValueError('Number of dimensions of input must be 2 to 4')
        return torch.cat([
            (x[:, i] * self.image_std[i] + self.image_mean[i]).unsqueeze(1) for i in range(x.shape[1])
        ], dim=1)

# EOF