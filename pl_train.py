# Import libraries
import torch, torchmetrics, wandb, timm, argparse, yaml, os, pickle
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torchmetrics import F1Score, Precision, Accuracy
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torchvision import transforms as tfs
from dataset import CustomDataset, get_dls
from transformations import get_tfs

class CustomModel(pl.LightningModule):
    
    """"
    
    This class gets several arguments and returns a model for training.
    
    Parameters:
    
        input_shape  - shape of input to the model, tuple -> int;
        model_name   - name of the model from timm library, str;
        num_classes  - number of classes to be outputed from the model, int;
        lr           - learning rate value, float.
    
    """
    
    def __init__(self, input_shape, model_name, num_classes, lr):
        super().__init__()
        
        # log hyperparameters
        self.save_hyperparameters()
        self.lr = lr
        self.f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.model = timm.create_model(model_name, pretrained = True, num_classes = num_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr = self.lr)
    
    def forward(self, inp): return self.model(inp)
    
    def training_step(self, batch, batch_idx):
        
        ims, gts = batch
        
        preds = self.model(ims)
        
        loss = self.ce_loss(preds, gts)
        
        # Train metrics
        pred_clss = torch.argmax(preds, dim = 1)
        acc = self.accuracy(pred_clss, gts)
        f1 = self.f1(pred_clss, gts)
        
        self.log("train_loss", loss, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        self.log("train_acc", acc, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        self.log("train_f1", f1, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        ims, gts = batch
        
        preds = self.model(ims)
        
        loss = self.ce_loss(preds, gts)
        
        # Train metrics
        pred_clss = torch.argmax(preds, dim = 1)
        acc = self.accuracy(pred_clss, gts)
        f1 = self.f1(pred_clss, gts)
        
        self.log("val_loss", loss, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        self.log("val_acc", acc, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        self.log("val_f1", f1, on_step = False, on_epoch = True, logger = True, sync_dist = True)
        
        return loss
    
#     def test_step(self, batch, batch_idx):
        
#         ims, gts = batch
#         preds = self.model(ims)
        
#         loss = self.ce_loss(preds, gts)
        
#         # Train metrics
#         pred_clss = torch.argmax(preds, dim = 1)
#         acc = self.accuracy(pred_clss, gts)
#         f1 = self.f1(pred_clss, gts)
        
#         self.log("test_loss", loss, on_step = False, on_epoch = True, logger = True, sync_dist = True)
#         self.log("test_acc", acc, on_step = False, on_epoch = True, logger = True, sync_dist = True)
#         self.log("test_f1", f1, on_step = False, on_epoch = True, logger = True, sync_dist = True)
            
class ImagePredictionLogger(Callback):
    
    def __init__(self, val_samples, cls_names = None, num_samples = 8):
        super().__init__()
        self.num_samples, self.cls_names = num_samples, cls_names
        self.val_imgs, self.val_labels = val_samples
        
    def on_validation_epoch_end(self, trainer, pl_module):
        
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device = pl_module.device)
        val_labels = self.val_labels.to(device = pl_module.device)
        
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        
        # Log the images as wandb Image
        if self.cls_names != None:
            trainer.logger.experiment.log({
                "Sample Validation Prediction Results":[wandb.Image(x, caption = f"Predicted class: {list(self.cls_names.values())[pred.item()]}, Ground truth class: {list(self.cls_names.values())[y]}") 
                               for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                     preds[:self.num_samples], 
                                                     val_labels[:self.num_samples])]})

