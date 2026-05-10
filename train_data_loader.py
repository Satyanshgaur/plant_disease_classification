import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import multiprocessing

def get_data_loaders(train_dir, val_dir, batch_size=32, image_size=(224, 224), num_workers=None):
    """
    Prepares and returns training and validation data loaders.
    
    Args:
        train_dir (str): Path to the training data directory.
        val_dir (str): Path to the validation (test) data directory.
        batch_size (int): Number of images per batch.
        image_size (tuple): Target size for the images (height, width).
        num_workers (int): Number of parallel cores to use for batch generation. 
                           Defaults to the number of available CPU cores.
    
    Returns:
        train_loader (DataLoader): PyTorch DataLoader for the training set.
        val_loader (DataLoader): PyTorch DataLoader for the validation set.
    """
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    # Define transformations for training and validation
    # 1. Resize images (normalize pixel size)
    # 2. Convert to tensor (normalizes pixel values to [0, 1])
    # 3. Normalize with ImageNet mean and std (standard for transfer learning)
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),  # Augmentation for better training
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])

    # Create DataLoaders
    # Shuffle is enabled for training data
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_DIR = os.path.join(BASE_DIR, "data/train")
    VAL_DIR = os.path.join(BASE_DIR, "data/test")
    
    # Configuration
    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)
    NUM_WORKERS = multiprocessing.cpu_count()
    
    print(f"Initializing DataLoaders with:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Image Size: {IMAGE_SIZE}")
    print(f"  Parallel Cores (num_workers): {NUM_WORKERS}")
    
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print(f"Error: Training or validation directory not found. Please run 'prepare_data_grouped.py' first.")
    else:
        train_loader, val_loader = get_data_loaders(
            TRAIN_DIR, 
            VAL_DIR, 
            batch_size=BATCH_SIZE, 
            image_size=IMAGE_SIZE,
            num_workers=NUM_WORKERS
        )
        
        print(f"DataLoaders successfully created.")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        
        # Test a single batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
