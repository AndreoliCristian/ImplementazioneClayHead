# datasets/datamodule.py

import os
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchvision import transforms
from datasets.dataLoader import ObjectDetectionDataset

class ObjectDetectionDataModule(pl.LightningDataModule):
    def __init__(self, images_dir: str, metadata_dir: str, annotation_dir: str, batch_size=8, num_workers=4, val_split=0.2):
        super().__init__()
        self.images_dir = images_dir
        self.metadata_dir = metadata_dir
        self.annotation_dir = annotation_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        # Define transformations # TODO: APPLICARE LE CORRETTE TRANSFORMAZIONI
        self.transform = None

    def setup(self, stage=None):
        # Called on every GPU separately - so you can set state here
        # that you'd like to use in training/validation/testing

        # Load the full dataset
        full_dataset = ObjectDetectionDataset(
            images_dir=self.images_dir,
            metadata_dir=self.metadata_dir,
            annotation_dir=self.annotation_dir,
            transform=self.transform # TODO: APPLICARE LE CORRETTE TRANSFORMAZIONI
        )

        # Split the dataset into training and validation
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn # TODO: DEFINIRE UNA FUNZIONE DI COLLATE FN
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn # TODO: DEFINIRE UNA FUNZIONE DI COLLATE FN
        )

    def test_dataloader(self):
        # If you have a test dataset
        pass

    @staticmethod
    def collate_fn(batch): # TODO: DEFINIRE UNA FUNZIONE DI COLLATE FN, deve gestire gsd e waves in modo corretto.
        # Since targets can have different sizes (number of objects), we need to define a custom collate_fn
        return tuple(zip(*batch))
