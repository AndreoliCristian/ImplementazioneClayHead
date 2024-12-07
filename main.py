from datasets.datamodule import ObjectDetectionDataModule
from models.clayEncoder import Encoder
import yaml
import torch


def main():
    # Leggi il file YAML
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    '''datamodule = ObjectDetectionDataModule(
        images_dir=config["data"]["images_dir"],
        metadata_dir=config["data"]["metadata_dir"],
        annotation_dir=config["data"]["annotation_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        val_split=config["data"]["val_split"]
    )'''

    encoder = Encoder(
        mask_ratio=config["model"]["mask_ratio"],
        patch_size=config["model"]["patch_size"],
        shuffle=config["model"]["shuffle"],
        dim=config["model"]["dim"],
        depth=config["model"]["depth"],
        heads=config["model"]["heads"],
        dim_head=config["model"]["dim_head"],
        mlp_ratio=config["model"]["mlp_ratio"],
    )

    # Load encoder weights
    encoder_weights = torch.load(config["model"]["encoder_weights_path"])
    encoder.load_state_dict(encoder_weights)


if __name__ == '__main__':
    main()