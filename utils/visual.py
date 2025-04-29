import matplotlib.pyplot as plt
import numpy as np
import torch


def unnormalize_image(tensor, mean, std):
    """Reverse the normalization for visualization."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
def visualization(image, mask, predication):
    fig, axes = plt.subplots(1, 3,  figsize=(15, 3))
    
    image = image.clone().cpu()  
    image = unnormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = image.permute(1, 2, 0).numpy()

    #* Convert mask and prediction
    mask = mask.squeeze().cpu().numpy()
    predication = predication.squeeze().cpu().numpy()

    
    
    
    #* display the original image 
    axes[0].imshow(image)
    #axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    #* displaying the ground truth mask image
    axes[1].imshow(mask, cmap="viridis")
    axes[1].set_title("Ground Truth Mask")  
    axes[1].axis("off")
    
    axes[2].imshow(predication, cmap="viridis")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    plt.show()    