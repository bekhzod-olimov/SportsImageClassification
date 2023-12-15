# Import libraries
import torch, torchvision, os
from torch.utils.data import random_split, Dataset, DataLoader
from torch import nn; from PIL import Image
from torchvision import transforms as T; from glob import glob
# Set the manual seed
torch.manual_seed(2023)

class CustomDataset(Dataset):
    
    """

    This class gets several parameters and returns dataset to train an AI model.

    Parameters:

        root             - path to data, str;
        transformations  - transformations to be applied, torchvision object.    
    
    """
    
    def __init__(self, root, data, transformations = None):

        """

        This function gets several parameters and initiates the class.

        Parameters:

            root             - path to data, str;
            data             - data name, str;
            transformations  - transformations to be applied, transforms object.
        
        """
        
        self.transformations, self.data, self.root = transformations, data, root
        
        self.cls_names = {}
        df = pd.read_csv(f"{root}/{data}.csv")
        if data == "train":
            self.im_ids, self.lbls = df["image_ID"], df["label"] 
            classes = [cl for cl in np.unique(self.lbls)]
            self.cls_names = {value: key for key, value in enumerate(classes)}
        else: self.im_ids = df["image_ID"]
        
    def read_im(self, idx): return Image.open(os.path.join(self.root, self.data, self.im_ids[idx])).convert("RGB")

    def get_im_gt(self, idx, data): return (self.read_im(idx), self.cls_names[self.lbls[idx]]) if data == "train" else (self.read_im(idx), None)
    
    def __len__(self): return len(self.im_ids)

    def __getitem__(self, idx):
        
        im, gt = self.get_im_gt(idx, data = self.data)
        
        if self.transformations is not None: im = self.transformations(im)
        
        return im, gt if self.data == "train" else im
    
def get_dls(root, transformations, bs, split = [0.9, 0.05], ns = 4):
    
    tr_ds = CustomDataset(root = root, data = "train", transformations = transformations)
    cls_names = tr_ds.cls_names
    ts_ds = CustomDataset(root = root, data = "test", transformations = transformations)
    
    all_len = len(tr_ds); tr_len = int(all_len * split[0]); val_len = all_len - tr_len
    tr_ds, val_ds = random_split(dataset = tr_ds, lengths = [tr_len, val_len])
    
    tr_dl, val_dl, ts_dl = DataLoader(tr_ds, batch_size = bs, shuffle = True, num_workers = ns), DataLoader(val_ds, batch_size = bs, shuffle = False, num_workers = ns), DataLoader(ts_ds, batch_size = 1, shuffle = False, num_workers = ns)
    
    return tr_dl, val_dl, ts_dl, cls_names
