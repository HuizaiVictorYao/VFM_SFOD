import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import ImageFilter
import random

import cv2 as cv

from torchvision.transforms import RandAugment

class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        # Convert torch tensor to PIL image
        x_pil = TF.to_pil_image(x)
        
        # Randomly choose sigma
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        
        # Apply Gaussian blur
        x_blurred_pil = x_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        # Convert blurred PIL image back to torch tensor
        x_blurred = TF.to_tensor(x_blurred_pil).cuda()
        
        return x_blurred
        

def scale_tensor(image, h, w):
    scaled_tensor = F.interpolate(image.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
    scaled_tensor = scaled_tensor.squeeze(0)

    return scaled_tensor

def save_tensor_as_image(tensor, file_path):
    array = tensor.permute(1, 2, 0).cpu().numpy()

    array = (array * 255).astype(np.uint8)

    image = Image.fromarray(array)

    image.save(file_path)

    

def weak_aug(images):
    transform = T.RandomHorizontalFlip(p=0.5)
        
    transformed_images = torch.stack([transform(image) for image in images])

    return transformed_images


def strong_aug(images):
    augmentation = []
    augmentation.append(T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
    augmentation.append(T.RandomGrayscale(p=0.2))
    augmentation.append(T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

    randcrop_transform = T.RandomApply([
            T.RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"),
            T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"),
            T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"),
        ], p=1.0)
    augmentation.append(randcrop_transform)

    transformed_images = torch.stack([T.Compose(augmentation)(image) for image in images])
    return transformed_images

