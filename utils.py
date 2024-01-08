import cv2, torch, random, numpy as np
from collections import OrderedDict as OD
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm; from time import time
from torchvision import transforms as T

# Inverse transformations
inv_fn = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                T.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

def get_state_dict(checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = OD()
    for k, v in checkpoint["state_dict"].items():
        name = k.replace("model.", "") # remove `model.`
        new_state_dict[name] = v
    return new_state_dict

def tn2np(t, inv_fn=None): return (inv_fn(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8) if inv_fn is not None else (t * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def get_preds(model, test_dl, device):
    model.eval()
    print("Start inference...")
    
    all_ims, all_preds, all_gts, acc = [], [], [], 0
    start_time = time()
    for idx, batch in tqdm(enumerate(test_dl)):
        if idx == 100: break
        ims, _ = batch
        # all_ims.extend(ims); all_gts.extend(gts);        
        all_ims.extend(ims);        
        preds = model(ims.to(device))
        pred_clss = torch.argmax(preds.data, dim = 1)
        all_preds.extend(pred_clss)
        # acc += (pred_clss == gts.to(device)).sum().item()
        
    print(f"Inference is completed in {(time() - start_time):.3f} secs!")
    # print(f"Accuracy of the model is {acc / len(test_dl.dataset) * 100:.3f}%")
    
    # return all_ims, all_preds, all_gts
    return all_ims, all_preds
    
# def visualize(all_ims, all_preds, all_gts, num_ims, rows, cls_names, save_path, save_name):
def visualize(all_ims, all_preds, num_ims, rows, cls_names, save_path, save_name):
    
    print("Start visualization...")
    plt.figure(figsize = (20, 20))
    indices = [random.randint(0, len(all_ims) - 1) for _ in range(num_ims)]
    
    for idx, ind in enumerate(indices):
        
        im = all_ims[ind]
        # gt = all_gts[ind].item()
        pr = all_preds[ind].item()
        
        plt.subplot(rows, num_ims // rows, idx + 1)
        plt.imshow(tn2np(im, inv_fn = inv_fn))
        plt.axis("off")
        # plt.title(f"GT: {list(cls_names.keys())[gt]} | Pred: {list(cls_names.keys())[pr]}")
        plt.title(f"Pred: {list(cls_names.keys())[pr]}")
    
    plt.savefig(f"{save_path}/{save_name}_preds.png")
    print(f"The visualization can be seen in {save_path} directory.")
    
def grad_cam(model, all_ims, num_ims, rows, save_path, save_name):
    
    print("\nStart GradCAM visualization...")
    plt.figure(figsize = (20, 20))
    indices = [random.randint(0, len(all_ims) - 1) for _ in range(num_ims)]
    
    for idx, ind in enumerate(indices):
        im = all_ims[ind]
        ori_cam = tn2np(im, inv_fn = inv_fn) / 255
        cam = GradCAM(model = model, target_layers = [model.features[-1]], use_cuda = True)
        grayscale_cam = cam(input_tensor = im.unsqueeze(0))[0, :]
        vis = show_cam_on_image(ori_cam, grayscale_cam, image_weight = 0.6, use_rgb = True)
        
        plt.subplot(rows, num_ims // rows, idx + 1)
        plt.imshow(vis)
        plt.axis("off")
        plt.title("GradCAM Visualization")
        
    plt.savefig(f"{save_path}/{save_name}_gradcam.png")
    print(f"The GradCAM visualization can be seen in {save_path} directory.")
    
class DrawLearningCurves():
    
    def __init__(self, learning_curves, save_path):
        
        self.lc, self.save_path = learning_curves, save_path
        
    def plot(self, metric, label):  plt.plot(self.lc[metric], label = label)
        
    def decorate(self, ylabel, title): plt.title(title); plt.xlabel("Epochs"); plt.ylabel(ylabel); plt.legend()
    
    def save_learning_curves(self):
        
        self.visualize(metric1 = "tr_losses", metric2 = "val_losses", label1 = "Train Loss", 
                       label2 = "Validation Loss", title = "Loss Learning Curve", 
                       ylabel = "Loss Scores", fname = "loss_learning_curves")
        
        self.visualize(metric1 = "tr_accs", metric2 = "val_accs", label1 = "Train Accuracy", 
                       label2 = "Validation Accuracy", title = "Accuracy Score Learning Curve", 
                       ylabel = "Accuracy Scores", fname = "acc_learning_curves")
    
    def visualize(self, metric1, metric2, label1, label2, title, ylabel, fname):
        
        plt.figure(figsize=(10, 5))
        self.plot(metric1, label1); self.plot(metric2, label2)
        self.decorate(ylabel, title); plt.xticks(np.arange(len(self.lc[metric1])), [i for i in range(1, len(self.lc[metric1]) + 1)])               
        plt.savefig(f"{self.save_path}/{fname}.png")
