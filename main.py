# Import libraries
import torch, wandb, argparse, yaml, os, pickle, pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from dataset import get_dls
from transformations import get_tfs
from time import time
from train import train_setup, train
from utils import DrawLearningCurves

def run(args):
    
    """
    
    This function runs the main script based on the arguments.
    
    Parameter:
    
        args - parsed arguments.
        
    Output:
    
        train process.
    
    """
    
    # Get train arguments 
    argstr = yaml.dump(args.__dict__, default_flow_style = False)
    print(f"\nTraining Arguments:\n\n{argstr}")
    
    os.makedirs(args.dls_dir, exist_ok=True); os.makedirs(args.stats_dir, exist_ok=True)
    
    transformations = get_tfs()
    tr_dl, val_dl, ts_dl, classes = get_dls(root = args.root, transformations = transformations, bs = args.batch_size)
    
    if os.path.isfile(f"{args.dls_dir}/tr_dl") and os.path.isfile(f"{args.dls_dir}/val_dl") and os.path.isfile(f"{args.dls_dir}/ts_dl"): pass
    else:
        torch.save(tr_dl,   f"{args.dls_dir}/tr_dl")
        torch.save(val_dl,  f"{args.dls_dir}/val_dl")
        torch.save(ts_dl, f"{args.dls_dir}/test_dl")
    
    tr_dl, val_dl = torch.load(f"{args.dls_dir}/tr_dl"), torch.load(f"{args.dls_dir}/val_dl")
    
    cls_names_file = f"{args.dls_dir}/cls_names.pkl"
    if os.path.isfile(cls_names_file): pass
    else:
        with open(f"{cls_names_file}", "wb") as f: 
            pickle.dump(classes, f)

    m, epochs, device, loss_fn, optimizer = train_setup(model_name = args.model_name, epochs = args.epochs, classes = classes, device = args.device)
    results = train(tr_dl = tr_dl, val_dl = val_dl, m = m, device = args.device, 
                    loss_fn = loss_fn, optimizer = optimizer, epochs = args.epochs, 
                    save_dir = "saved_models", save_prefix = f"{args.root.split('/')[-2]}")
    
    DrawLearningCurves(results, args.stats_dir).save_learning_curves()
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = 'Image Classification Training Arguments')
    
    # Add arguments to the parser
    parser.add_argument("-r", "--root", type = str, default = "/mnt/data/dataset/bekhzod/im_class/sports/dataset", help = "Path to data")
    parser.add_argument("-bs", "--batch_size", type = int, default = 64, help = "Mini-batch size")
    parser.add_argument("-is", "--inp_im_size", type = tuple, default = (224, 224), help = "Input image size")
    parser.add_argument("-mn", "--model_name", type = str, default = "rexnet_150", help = "Model name for backbone")
    parser.add_argument("-d", "--device", type = str, default = "cuda:2", help = "GPU device name")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-3, help = "Learning rate value")
    parser.add_argument("-e", "--epochs", type = int, default = 10, help = "Train epochs number")
    parser.add_argument("-sm", "--save_model_path", type = str, default = 'saved_models', help = "Path to the directory to save a trained model")
    parser.add_argument("-sd", "--stats_dir", type = str, default = "stats", help = "Path to dir to save train statistics")
    parser.add_argument("-dl", "--dls_dir", type = str, default = "saved_dls", help = "Path to dir to save dataloaders")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)
