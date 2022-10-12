from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
import pandas as pd
import ast


wandb.init(project="ecg", 
           entity="multi-modal-fsdl2022",
           config = {
                "learning_rate": 0.001,
                "epochs": 100,
                "train_batch_size": 64,
                "val_batch_size": 16,
           })
wandb_logger = WandbLogger(log_model="all")
checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch', mode='min')

# Necessary paths
project_path = Path(__file__).parent
data_dir = project_path/'data/coherent/csv'
pretrained_weight_path = project_path/'weights/trained_weight_mitdb.pth'


class ECGDataset(Dataset):

    def __init__(self, csv_path: Path) -> None:
        df = pd.read_csv(csv_path)

        ecg_str = df['value']        
        self.ecg = []
        for ecg in ecg_str:
            self.ecg.append([float(x) for x in ecg.split()])
        self.ecg = torch.Tensor(self.ecg)
        # self.ecg = torch.unsqueeze(self.ecg, dim=0)
        
        labels = df['label'].values
        self.labels = []
        for l in labels:
            self.labels.append(ast.literal_eval(l))
        self.labels = torch.Tensor(self.labels)

    def __getitem__(self, idx):
        """
        Return torch Tensor of an ECG signal and its label
        """
        return torch.unsqueeze(self.ecg[idx], dim=0), self.labels[idx]

    def __len__(self):
        return len(self.labels)


class ECGModel(pl.LightningModule):
    def __init__(self, pretrained_weight_path: Path, lr=0.001):
        super().__init__()
        self.lr = lr
        # load the pretrained model on the MIT-BIH dataset
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3)  # [bz, 16, 401]
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2)  # [bz, 16, 200]
        self.conv2 = nn.Conv1d(16, 16, 5, padding=2)  # [bz, 16, 200]
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(2)  # [bz, 16, 100]
        self.load_init_weights(pretrained_weight_path)  
        self.linear1 = nn.Linear(16*100, 512)
        self.relu3 = nn.LeakyReLU()
        self.linear2 = nn.Linear(512, 128)
        self.relu4 = nn.LeakyReLU()
        self.linear3 = nn.Linear(128, 5)

    def load_init_weights(self, init_weight_path: Path):
        checkpoint = torch.load(init_weight_path)
        self.conv1.weight.data = checkpoint["conv1.weight"]
        self.conv1.bias.data = checkpoint["conv1.bias"]
        self.conv2.weight.data = checkpoint["conv2.weight"]
        self.conv2.bias.data = checkpoint["conv2.bias"]
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 16*100)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        x = self.relu4(x)
        x = self.linear3(x)
        return x

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        y_hats = self.forward(xs)
        loss = F.binary_cross_entropy_with_logits(y_hats, ys)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        y_hats = self.forward(xs)
        loss = F.binary_cross_entropy_with_logits(y_hats, ys)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    config = wandb.config
    
    train_dataset = ECGDataset(csv_path=data_dir/'train_ecg.csv')
    val_dataset = ECGDataset(csv_path=data_dir/'val_ecg.csv')
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=config.train_batch_size, 
                              shuffle=True, # images are loaded in random order
                              num_workers=12)                         
    val_loader = DataLoader(val_dataset, 
                            batch_size=config.val_batch_size,
                            num_workers=12)
    
    model = ECGModel(pretrained_weight_path)
    trainer = pl.Trainer(accelerator='gpu', devices=int(torch.cuda.is_available()), 
                         max_epochs=config.epochs, logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)