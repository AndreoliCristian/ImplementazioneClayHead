# scripts/train.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.dataloader import ObjectDetectionDataset
from models import ObjectDetectionModel
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint

# TODO: Utilize callbacks
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
# TODO: Utilize loggers
from pytorch_lightning.loggers import TensorBoardLogger



def main():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Prepare dataset and dataloader
    dataset = ObjectDetectionDataset(
        images_dir='data/images',
        metadata_dir='data/metadata',
        transform=transform
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='checkpoints/',
        filename='best_model',
        save_top_k=1,
        mode='min',
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # Initialize model
    model = ObjectDetectionModel(
        encoder_weights_path='path/to/pretrained_weights.pth',
        num_classes=80,  # Adjust based on your dataset
        learning_rate=1e-3
    )

    # Training
    trainer = pl.Trainer(
        max_epochs=20,
        gpus=1 if torch.cuda.is_available() else 0,
        default_root_dir='checkpoints/',
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, dataloader)

    # Save the final model
    trainer.save_checkpoint('checkpoints/final_model.ckpt')


if __name__ == '__main__':
    main()
