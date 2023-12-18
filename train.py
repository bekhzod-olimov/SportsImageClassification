import os, timm, torch 
from tqdm import tqdm

def train_setup(model_name, epochs, classes, device, lr = 3e-4): 
    m = timm.create_model(model_name, pretrained = True, num_classes = len(classes))  
    return m.to(device), epochs, device, torch.nn.CrossEntropyLoss(), torch.optim.Adam(params = m.parameters(), lr = lr)

def to_device(batch, device): return batch[0].to(device), batch[1].to(device)

def get_metrics(model, ims, gts, loss_fn, epoch_loss, epoch_acc): preds = model(ims); loss = loss_fn(preds, gts); return loss, epoch_loss + (loss.item()), epoch_acc + (torch.argmax(preds, dim = 1) == gts).sum().item()

def train(tr_dl, val_dl, m, device, loss_fn, optimizer, epochs, threshold = 0.01, save_dir = "saved_models", save_prefix = "med", train_framework = "py"):
    print("Start training...")
    learning_curves, tr_losses, val_losses, tr_accs, val_accs = {}, [], [], [], []
    best_loss = float(torch.inf)
    
    for epoch in range(epochs):

        epoch_loss, epoch_acc = 0, 0
        for idx, batch in tqdm(enumerate(tr_dl)):
            
            ims, gts = to_device(batch, device)
            
            loss, epoch_loss, epoch_acc = get_metrics(m, ims, gts, loss_fn, epoch_loss, epoch_acc)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        tr_loss_to_track = epoch_loss / len(tr_dl)
        tr_acc_to_track  = epoch_acc / len(tr_dl.dataset)
        tr_losses.append(tr_loss_to_track); tr_accs.append(tr_acc_to_track)
        
        print(f"{epoch + 1}-epoch train process is completed!")
        print(f"{epoch + 1}-epoch train loss          -> {tr_loss_to_track:.3f}")
        print(f"{epoch + 1}-epoch train accuracy      -> {tr_acc_to_track:.3f}")

        m.eval()
        with torch.no_grad():
            val_epoch_loss, val_epoch_acc = 0, 0
            for idx, batch in enumerate(val_dl):
                ims, gts = batch
                ims, gts = ims.to(device), gts.to(device)

                preds = m(ims)
                loss = loss_fn(preds, gts)
                pred_cls = torch.argmax(preds.data, dim = 1)
                val_epoch_acc += (pred_cls == gts).sum().item()
                val_epoch_loss += loss.item()

            val_loss_to_track = val_epoch_loss / len(val_dl)
            val_acc_to_track  = val_epoch_acc / len(val_dl.dataset)
            val_losses.append(val_loss_to_track); val_accs.append(val_acc_to_track)
            
            print(f"{epoch + 1}-epoch validation process is completed!")
            print(f"{epoch + 1}-epoch validation loss     -> {val_loss_to_track:.3f}")
            print(f"{epoch + 1}-epoch validation accuracy -> {val_acc_to_track:.3f}")

            if val_loss_to_track < (best_loss + threshold):
                os.makedirs(save_dir, exist_ok = True)
                best_loss = val_loss_to_track
                torch.save(m.state_dict(), f"{save_dir}/{save_prefix}_best_model_{train_framework}.pth")
                
    learning_curves["tr_losses"] = tr_losses; learning_curves["val_losses"] = val_losses; learning_curves["tr_accs"] = tr_accs; learning_curves["val_accs"] = val_accs
    
    return learning_curves
