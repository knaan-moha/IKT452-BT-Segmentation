from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CustomDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


#* this function  creates  train, val and test datasets 
#* Data preprocessing 

def create_data_loaders(train_image_paths, train_mask_paths, val_image_paths, val_mask_paths, test_image_paths, test_mask_paths, batch_size):
    #* transforms 
    transform = A.Compose([
        A.Resize(256, 256), 
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.ColorJitter(brightness=0.5, contrast=0.5, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
        
    ])
    
    test_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    train_dataset = CustomDataset(train_image_paths, train_mask_paths, transform=transform)
    val_dataset = CustomDataset(val_image_paths, val_mask_paths, transform= test_transform)
    test_dataset = CustomDataset(test_image_paths, test_mask_paths, transform= test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader