# Import libraries
import torch, wandb, argparse, yaml, os, pickle, pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from dataset import get_dls
from transformations import get_tfs
from time import time
from pl_train import CustomModel, ImagePredictionLogger
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
    
    os.system(f"wandb login --relogin {args.wandb_key}")
    os.makedirs(args.dls_dir, exist_ok=True); os.makedirs(args.stats_dir, exist_ok=True)
    
    transformations = get_tfs()
    tr_dl, val_dl, ts_dl, classes = get_dls(root = args.root, transformations = transformations, bs = args.batch_size)
    
    if os.path.isfile(f"{args.dls_dir}/tr_dl") and os.path.isfile(f"{args.dls_dir}/val_dl") and os.path.isfile(f"{args.dls_dir}/ts_dl"): pass
    else:
        torch.save(tr_dl,   f"{args.dls_dir}/tr_dl")
        torch.save(val_dl,  f"{args.dls_dir}/val_dl")
        torch.save(ts_dl, f"{args.dls_dir}/test_dl")
    
    tr_dl, val_dl, ts_dl = torch.load(f"{args.dls_dir}/tr_dl"), torch.load(f"{args.dls_dir}/val_dl"), torch.load(f"{args.dls_dir}/test_dl")
    
    cls_names_file = f"{args.dls_dir}/cls_names.pkl"
    if os.path.isfile(cls_names_file): pass
    else:
        with open(f"{cls_names_file}", "wb") as f: 
            pickle.dump(classes, f)
    
    data_name = args.root.split('/')[-2]
    
    if args.train_framework == "pl":
        
        ckpt_name = f"{data_name}_best_model_{args.model_name}_{args.train_framework}"
        # Samples required by the custom ImagePredictionLogger callback to log image predictions. 
        val_samples = next(iter(val_dl))

        model = CustomModel(input_shape = args.inp_im_size, model_name = args.model_name, num_classes = len(classes), lr = args.learning_rate) 

        # Initialize wandb logger
        wandb_logger = WandbLogger(project = f"{data_name}", job_type = "train", name = f"{args.model_name}_{data_name}_{args.batch_size}_{args.learning_rate}")

        # Initialize a trainer
        trainer = pl.Trainer(max_epochs = args.epochs, accelerator="gpu", devices = args.devices, strategy = "ddp", 
                             logger = wandb_logger, 
                             # fast_dev_run = True,
                             callbacks = [EarlyStopping(monitor = "val_loss", mode = "min", patience = 5), ImagePredictionLogger(val_samples, classes),
                                          ModelCheckpoint(monitor = "val_loss", dirpath = args.save_model_path, filename = ckpt_name)])
        
        # Train the model
        trainer.fit(model, tr_dl, val_dl)

        # Test the model
        # trainer.test(ckpt_path = f"{{args.save_model_path}/ckpt_name}", dataloaders = ts_dl)

        # Close wandb run
        wandb.finish()
    
    elif args.train_framework == "py":
        
        m, epochs, device, loss_fn, optimizer = train_setup(model_name = args.model_name, epochs = args.epochs, classes = classes, device = args.device)
        results = train(tr_dl = tr_dl, val_dl = val_dl, m = m, device = args.device, 
                        loss_fn = loss_fn, optimizer = optimizer, epochs = args.epochs, 
                        save_dir = args.save_model_path, save_prefix = data_name, train_framework = args.train_framework)

        DrawLearningCurves(results, args.stats_dir).save_learning_curves()
        
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = 'Image Classification Training Arguments')
    
    # Add arguments to the parser
    parser.add_argument("-r", "--root", type = str, default = "/mnt/data/dataset/bekhzod/im_class/sports/dataset", help = "Path to data")
    parser.add_argument("-bs", "--batch_size", type = int, default = 8, help = "Mini-batch size")
    parser.add_argument("-is", "--inp_im_size", type = tuple, default = (224, 224), help = "Input image size")
    parser.add_argument("-mn", "--model_name", type = str, default = "rexnet_150", help = "Model name for backbone")
    parser.add_argument("-ds", "--devices", type = int, default = 2, help = "GPU devices number")
    parser.add_argument("-d", "--device", type = str, default = "cuda:1", help = "GPU device name")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-3, help = "Learning rate value")
    parser.add_argument("-e", "--epochs", type = int, default = 10, help = "Train epochs number")
    parser.add_argument("-sm", "--save_model_path", type = str, default = 'saved_models', help = "Path to the directory to save a trained model")
    parser.add_argument("-sd", "--stats_dir", type = str, default = "stats", help = "Path to dir to save train statistics")
    parser.add_argument("-wk", "--wandb_key", type = str, default = "3204eaa1400fed115e40f43c7c6a5d62a0867ed1", help = "Wandb key can be obtained from wandb.ai")
    parser.add_argument("-dl", "--dls_dir", type = str, default = "saved_dls", help = "Path to dir to save dataloaders")
    parser.add_argument("-tf", "--train_framework", type = str, default = "pl", help = "Framework to be used for training an AI model")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)
