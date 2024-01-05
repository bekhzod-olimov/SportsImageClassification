# Import libraries
import torch, pickle, timm, argparse, streamlit as st
from transformations import get_tfs  
from PIL import Image, ImageFont
from utils import get_state_dict
from torchvision.datasets import ImageFolder
st.set_page_config(layout = 'wide')

def run(args):
    
    """
    
    This function gets parsed arguments and runs the script.
    
    Parameter:
    
        args   - parsed arguments, argparser object;
        
    """
    
    # Get class names for later use
    with open("saved_dls/cls_names.pkl", "rb") as f: cls_names = pickle.load(f)
    
    # Get number of classes
    num_classes = len(cls_names)
    
    # Initialize transformations to be applied
    transformations = get_tfs()
    
    # Set a default path to the image
    default_path = "sample_ims/soccer.jpg"
    
    # Load classification model
    m = load_model(args.model_name, num_classes, args.checkpoint_path)
    
    # Set title and decorate streamlit page
    st.title("Fruit Classifier")
    file = st.file_uploader("Please upload your image")

    # Get image and predicted class
    im, result = predict(m = m, path = file, tfs = transformations, cls_names = cls_names) if file else predict(m = m, path = default_path, tfs = transformations, cls_names = cls_names)
    # Write on the streamlit page
    st.write(f"INPUT IMAGE: "); st.image(im); st.write(f"PREDICTED AS -> {result.upper()}")
        
# @st.cache_data
def load_model(model_name, num_classes, checkpoint_path): 
    
    """
    
    This function gets several parameters and loads a classification model.
    
    Parameters:
    
        model_name      - name of a model from timm library, str;
        num_classes     - number of classes in the dataset, int;
        checkpoint_path - path to the trained model, str;
        
    Output:
    
        m              - a model with pretrained weights and in an evaluation mode, torch model object;
    
    """
    
    # Load an AI model
    m = timm.create_model(model_name, num_classes = num_classes)
    # Load the pretrained trainable parameters for the model
    m.load_state_dict(get_state_dict(args.checkpoint_path))
    
    # Change to evaluation model and return the model
    return m.eval()

def predict(m, path, tfs, cls_names):

    """

    This function gets several parameters and returns an original image and the corresponding predicted class by the AI model.

    Parameters:

        m           - a pretrained AI model, timm object;
        path        - path to an image, str;
        tfs         - transformations to be applied, transforms object;
        cls_names   - class names from the pretrained dataset, list/dict. 
    
    """
    
    im = Image.open(path).convert("RGB")
    cls_names = list(cls_names.keys()) if isinstance(cls_names, dict) else cls_names
    
    return im, cls_names[torch.argmax(m(tfs(im).unsqueeze(0)), dim = 1).item()]
        
if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = "Sport Types Classification Demo")
    
    # Add arguments
    parser.add_argument("-mn", "--model_name", type = str, default = "rexnet_150", help = "Model name for backbone")
    parser.add_argument("-cp", "--checkpoint_path", type = str, default = "./saved_models/sports_best_model_rexnet_150_pl.ckpt", help = "Path to the checkpoint")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args) 
