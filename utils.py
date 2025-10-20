import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


def get_transform(args):
    """
    Create image transformations for preprocessing
    Args:
        args: Arguments containing image_size and other parameters
    
    Returns:
        preprocess: Transform for input images
        target_transform: Transform for target masks (can be None)
    """
    
    # Custom transform that handles RGBA to RGB conversion
    class ConvertToRGB:
        def __call__(self, img):
            return img.convert('RGB')
    
    # Standard ImageNet normalization values
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    # Create preprocessing pipeline with RGB conversion
    preprocess = transforms.Compose([
        ConvertToRGB(),  # Ensure RGB format
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    # For target masks (ground truth), we typically just resize and convert to tensor
    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
    
    return preprocess, target_transform


class MockArgs:
    """Mock arguments class for compatibility"""
    def __init__(self, image_size=518):
        self.image_size = image_size
        self.dataset = 'mvtec'
        self.features_list = [6, 12, 18, 24]
        self.sigma = 4
