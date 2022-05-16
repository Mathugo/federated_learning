import os
import PIL
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from openfl.interface.interactive_api.experiment import DataInterface
from torchvision import transforms as tsf


class KvasirShardDataset(Dataset):
    
    def __init__(self, dataset):
        self._dataset = dataset
        
        # Prepare transforms
        self.img_trans = tsf.Compose([
            tsf.ToPILImage(),
            tsf.Resize((332, 332)),
            tsf.ToTensor(),
            tsf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.mask_trans = tsf.Compose([
            tsf.ToPILImage(),
            tsf.Resize((332, 332), interpolation=PIL.Image.NEAREST),
            tsf.ToTensor()])
        
    def __getitem__(self, index):
        img, mask = self._dataset[index]
        img = self.img_trans(img).numpy()
        mask = self.mask_trans(mask).numpy()
        return img, mask
    
    def __len__(self):
        return len(self._dataset)

    

# Now you can implement you data loaders using dummy_shard_desc
class KvasirSD(DataInterface):

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
        self._shard_dataset = KvasirShardDataset(shard_descriptor.get_dataset('train'))
        
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