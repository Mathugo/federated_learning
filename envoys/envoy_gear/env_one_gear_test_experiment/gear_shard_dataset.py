import os
import PIL
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms as tsf
from openfl.interface.interactive_api.experiment import DataInterface
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class GearShardDataset(Dataset):
    
    def __init__(self, dataset, num_classes: int=4, img_size: tuple = (400, 400)):
        self._dataset = dataset
        self.img_size = img_size
        print("[*] Dataset len {}".format(len(dataset)))
        # Prepare transforms
        self.img_trans = tsf.Compose([
            tsf.ToPILImage(),
            tsf.Resize(img_size),
            tsf.ToTensor(),
            # normalized settings for deeplab3
            tsf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            #tsf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.mask_trans = tsf.Compose([
            tsf.ToPILImage(),
            tsf.Resize(img_size, interpolation=PIL.Image.NEAREST),
            tsf.ToTensor()])

        self.transforms = tsf.Compose([
            #tsf.ToPILImage(),
            tsf.RandomHorizontalFlip(),
            tsf.RandomVerticalFlip(),
            tsf.RandomCrop((224, 224)),
            tsf.Resize(img_size),
            tsf.ToTensor(),
            # normalized settings for deeplab3
            tsf.Normalize(mean=[0.485, 0.456, 0.406, 0], std=[0.229, 0.224, 0.225, 1])])

    def __getitem__(self, index):
        img, mask = self._dataset[index]
        img, mask = self._dataset[index]
        #img = self.img_trans(img).numpy()
        #mask = self.mask_trans(mask).numpy()
        # Concatenate image and label, to apply same transformation on both
        
        image_np = np.asarray(img)
        label_np = np.asarray(mask)

        new_shape = (image_np.shape[0], image_np.shape[1], image_np.shape[2] + 1)
        image_and_label_np = np.zeros(new_shape, image_np.dtype)
        image_and_label_np[:, :, 0:3] = image_np
        image_and_label_np[:, :, 3] = label_np

        # Convert to PIL
        image_and_label = Image.fromarray(image_and_label_np)

        # Apply Transforms
        image_and_label = self.transforms(image_and_label)

        # Extract image and label
        image = np.reshape(image_and_label[0:3, :, :], (3, 400, 400))
        label = image_and_label[3, :, :].unsqueeze(0)

        # Normalize back from [0, 1] to [0, 255]
        label = label * 255
        #  Convert to int64 and remove second dimension
        label = label.long().squeeze()

        return image, label
        
        """
        new_shape = (image_np.shape[0], image_np.shape[1], image_np.shape[2] + 1)
        image_and_label_np = np.zeros(new_shape, image_np.dtype)
        image_and_label_np[:, :, 0:3] = image_np
        image_and_label_np[:, :, 3] = label_np

        # Convert to PIL
        image_and_label = Image.fromarray(image_and_label_np)

        # Apply Transforms
        image_and_label = self.transforms(image_and_label)

        # Extract image and label
        image = image_and_label[0:3, :, :]
        label = image_and_label[3, :, :].unsqueeze(0)

        #image = np.reshape(image, (3, self.img))

        # Normalize back from [0, 1] to [0, 255]
        label = label * 255
        #  Convert to int64 and remove second dimension
        label = label.long().squeeze()
        print("[*] Image shape {} Mask {}".format(img.shape, mask.shape))
        return img, mask
        """
    
    def __len__(self):
        return len(self._dataset)

    

# Now you can implement you data loaders using dummy_shard_desc
class GearSD(DataInterface):

    def __init__(self, validation_fraction=1/8, **kwargs):
        super().__init__(**kwargs)
        
        self.validation_fraction = validation_fraction
        
    @property
    def shard_descriptor(self):
        return self._shard_descriptor
        
    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        """
        Describe per-collaborator procedures or sharding.
        This method will be called during a collaborator initialization.
        Local shard_descriptor  will be set by Envoy.
        """
        self._shard_descriptor = shard_descriptor
        self._shard_dataset = GearShardDataset(shard_descriptor.get_dataset('train'))
        
        validation_size = max(1, int(len(self._shard_dataset) * self.validation_fraction))
        
        self.train_indeces = np.arange(len(self._shard_dataset) - validation_size)
        self.val_indeces = np.arange(len(self._shard_dataset) - validation_size, len(self._shard_dataset))
    
    def get_train_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks with optimizer in contract
        """
        train_sampler = SubsetRandomSampler(self.train_indeces)
        return DataLoader(
            self._shard_dataset,
            num_workers=8,
            batch_size=self.kwargs['train_bs'],
            sampler=train_sampler
        )

    def get_valid_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks without optimizer in contract
        """
        val_sampler = SubsetRandomSampler(self.val_indeces)
        return DataLoader(
            self._shard_dataset,
            num_workers=8,
            batch_size=self.kwargs['valid_bs'],
            sampler=val_sampler
        )

    def get_train_data_size(self):
        """
        Information for aggregation
        """
        return len(self.train_indeces)

    def get_valid_data_size(self):
        """
        Information for aggregation
        """
        return len(self.val_indeces)