"""Gear shard descriptor."""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import cv2
from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from openfl.utilities import validate_file_hash

class GearShardDataset(ShardDataset):
    """Gear Shard dataset class."""

    def __init__(self, dataset_dir: Path, rank=1, worldsize=1, enforce_image_hw=None, image_extension="bmp", mask_extension="png", mask_tag="_label_ground-truth_semantic.", num_classes=4):
        """Initialize GearShardDataset."""
        self.rank = rank
        self.worldsize = worldsize
        self.dataset_dir = dataset_dir
        self.enforce_image_hw = enforce_image_hw
        self.images_path = self.dataset_dir / 'segmented-images' / 'images'
        self.masks_path = self.dataset_dir / 'segmented-images' / 'masks'
        self.image_extension = image_extension
        self.mask_extension = mask_extension
        self.mask_tag = mask_tag
        self.num_classes=num_classes
        self.images_names = [
            img_name
            for img_name in sorted(os.listdir(self.images_path))
            if len(img_name) > 3 and img_name[-3:] == self.image_extension
        ]
        # Sharding
        self.images_names = self.images_names[self.rank - 1::self.worldsize]

    def __getitem__(self, index):
        """Return a item by the index."""
        name = self.images_names[index]
        # Reading data
        name_mask = name.replace("."+self.image_extension, self.mask_tag+self.mask_extension)
        image_path = os.path.join(self.images_path, name)
        mask_path = os.path.join(self.masks_path, name_mask)

        img = Image.open(image_path)
        mask = ImageOps.grayscale(Image.open(mask_path))

        if self.enforce_image_hw is not None:
            # If we need to resize data
            # PIL accepts (w,h) tuple, not (h,w)
            img = img.resize(self.enforce_image_hw[::-1])
            mask = mask.resize(self.enforce_image_hw[::-1])

        img = np.asarray(img)
        mask = np.reshape(np.asarray(mask).astype(np.uint8), (self.enforce_image_hw[1], self.enforce_image_hw[0]))
        
        # transform pixel mask (400, 400) into (400, 400, num_classes)
        if np.min(mask) == 0:
            mask+=1

        shape = (self.enforce_image_hw[1], self.enforce_image_hw[0], self.num_classes)

        #print("MIN MASK {} MAX {} Name {}".format(np.min(mask), np.max(mask), name))
        #print("[*] Number of classes {}, creating new samples of masks array {}".format(self.num_classes, shape))
        masks = np.zeros((shape))

        for i in range(self.num_classes):
            masks[:,:,i] = np.where(mask == i, 1, 0)
        # check rgb
        assert img.shape[2] == 3 
        return img, masks.astype(np.uint8)
        #return img, mask[:, :, 0].astype(np.uint8)

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.images_names) 

class GearShardDescriptor(ShardDescriptor):
    """Shard descriptor class."""

    def __init__(self, data_folder: str = 'dataset',
                 rank_worldsize: str = '1,1',
                 enforce_image_hw: str = None) -> None:
        """Initialize GearShardDescriptor."""
        super().__init__()
        # Settings for sharding the dataset
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

        self.data_folder = Path.cwd() / data_folder
        #self.download_data(self.data_folder)

        # Settings for resizing data
        self.enforce_image_hw = None
        if enforce_image_hw is not None:
            self.enforce_image_hw = tuple(int(size) for size in enforce_image_hw.split(','))

        # Calculating data and target shapes
        ds = self.get_dataset()
        sample, target = ds[0]
        self._sample_shape = [str(dim) for dim in sample.shape]
        self._target_shape = [str(dim) for dim in target.shape]

    def get_dataset(self, dataset_type='train'):
        """Return a shard dataset by type."""
        return GearShardDataset(
            dataset_dir=self.data_folder,
            rank=self.rank,
            worldsize=self.worldsize,
            enforce_image_hw=self.enforce_image_hw
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return self._sample_shape

    @property
    def target_shape(self):
        """Return the target shape info."""
        return self._target_shape

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Gear dataset, shard number {self.rank} '
                f'out of {self.worldsize}')


