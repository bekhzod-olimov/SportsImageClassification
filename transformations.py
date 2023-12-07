# Import library
from torchvision import transforms as T

# Get transformations based on the input image dimensions
def get_tfs(im_size = (224, 224), imagenet_normalization = True):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean = mean, std = std)])