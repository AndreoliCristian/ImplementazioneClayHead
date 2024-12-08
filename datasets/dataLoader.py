# datasets/dataloader.py

import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.transforms import ToTensor, v2
from datetime import datetime
import math


class ObjectDetectionDataset(Dataset):
    def __init__(self, images_dir, metadata_dir, annotation_dir, transform=None): # TODO: TRASFORMAZIONI COME DATA AUGMENTATION
        self.images_dir = images_dir
        self.metadata_dir = metadata_dir
        self.annotation_dir = annotation_dir
        self.transform = v2.Compose(
            [
                MinMaxNormalize(min_val=0, max_val=1),
            ]
        )
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # Load image
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        # open the image file
        image = Image.open(image_path)
        # convert image to tensor
        image = torch.from_numpy(np.array(image).astype(np.float32))
        # unsqueeze the image tensor if it has only 2 dimensions
        if image.dim() == 2:
            image = image.unsqueeze(0)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # annotation file
        annotation_file = os.path.splitext(image_name)[0] + '.txt'
        annotation_path = os.path.join(self.annotation_dir, annotation_file)
        # open the annotation file
        annotation = np.loadtxt(annotation_path, dtype=np.float32) # [num_objects, 5]
        # add a dummy dimension if there is only one object
        if annotation.ndim == 1:
            annotation = np.expand_dims(annotation, axis=0)
        # create a dictionary with the annotation
        targets = {'boxes': torch.tensor(annotation[:, 1:], dtype=torch.float32),
                   'labels': torch.tensor(annotation[:, 0], dtype=torch.int64)}


        # Load and parse metadata
        xml_name = os.path.splitext(image_name)[0] + '.xml'
        xml_path = os.path.join(self.metadata_dir, xml_name)
        metadata = self.parse_xml(xml_path)

        # Prepare the data cube
        datacube = self.prepare_datacube(image, metadata)

        return datacube, targets


    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Read metadata from the XML
        #tree = ET.parse(self.metadata_dir)
        #root = tree.getroot()
        lat = float(root.findtext('.//LAT_OFF'))
        lon = float(root.findtext('.//LON_OFF'))
        time = root.findtext('.//time')

        metadata = {
            'time': time,
            'latlon': [lat, lon],
        }

        return metadata

    def prepare_datacube(self, image, metadata):
        time = time_transform(metadata['time'])
        latlon = latlon_transform(metadata['latlon'][0], metadata['latlon'][1])


        datacube = {
            'pixels': image,  # [C, H, W]
            'time': time, # [2]
            'latlon': latlon,  # [2]
        }
        return datacube


class MinMaxNormalize:
    def __init__(self, min_val=0.0, max_val=1.0):
        """
        Inizializza la trasformazione MinMaxNormalize.

        Args:
            min_val (float): Valore minimo desiderato dopo la normalizzazione.
            max_val (float): Valore massimo desiderato dopo la normalizzazione.
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        """
        Applica la normalizzazione min-max al tensore di input.

        Args:
            tensor (torch.Tensor): Immagine come tensore di dimensioni (C, H, W).

        Returns:
            torch.Tensor: Immagine normalizzata come tensore.
        """
        tensor_min = tensor.min()
        tensor_max = tensor.max()

        # Evita la divisione per zero
        if tensor_max - tensor_min == 0:
            return torch.full(tensor.size(), self.min_val)

        # Normalizza il tensore
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        scaled_tensor = normalized_tensor * (self.max_val - self.min_val) + self.min_val
        return scaled_tensor

def time_transform(time) -> torch.Tensor:
    datetime = convert_time_to_datetime(time)
    time = normalize_timestamp(datetime)

    week_norm = [dat[0] for dat in time]
    hour_norm = [dat[1] for dat in time]
    time = torch.tensor((week_norm, hour_norm), dtype=torch.float32)

    return time

def latlon_transform(LAT_OFF, LON_OFF) -> torch.Tensor:
    # Coordinates normalization
    lat_norm, lon_norm = normalize_latlon(LAT_OFF, LON_OFF)

    latlon = torch.tensor((lat_norm, lon_norm), dtype=torch.float32)

    return latlon


def normalize_latlon(lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180

    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))


# Function to convert time string to datetime object
def convert_time_to_datetime(time_str):
    try:
        # Define the format of the input time string
        time_format = "%Y-%m-%dT%H:%M:%S.%f"
        # Convert the time string to a datetime object
        time_dt = datetime.strptime(time_str, time_format)
        return time_dt
    except ValueError as e:
        print(f"Error converting time '{time_str}': {e}")
        return None


def normalize_timestamp(date):
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24

    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))