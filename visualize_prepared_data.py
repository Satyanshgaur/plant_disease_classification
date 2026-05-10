import os
import torch
import matplotlib.pyplot as plt
from train_data_loader import get_data_loaders
from torchvision import transforms

def show_image(image, label, dataset):
    """
    Displays an image and prints its label.
    """
    print("Label :" + dataset.classes[label] + "(" + str(label) + ")")
    
    # Denormalize the image for correct visualization
    # The training data was normalized with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    image = inv_normalize(image)
    image = torch.clamp(image, 0, 1) # Ensure pixel values are in valid [0, 1] range
    
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')

if __name__ == "__main__":
    # Setup paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_DIR = os.path.join(BASE_DIR, "data/train")
    VAL_DIR = os.path.join(BASE_DIR, "data/test")
    FIGURES_DIR = os.path.join(BASE_DIR, "reports/figures")
    
    # Ensure reports directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Initialize data loaders
    # We use a batch size of 5 for convenience to extract the 5 samples
    train_loader, _ = get_data_loaders(
        TRAIN_DIR, 
        VAL_DIR, 
        batch_size=5, 
        image_size=(224, 224)
    )
    
    # Get the dataset object for class names
    train_dataset = train_loader.dataset
    
    # Get one batch of 5 images
    images, labels = next(iter(train_loader))
    
    print(f"Inspecting training data and saving samples to {FIGURES_DIR}...\n")
    
    for i in range(5):
        plt.figure(figsize=(8, 8))
        
        # Use the requested format
        show_image(images[i], labels[i].item(), train_dataset)
        
        # Save the image
        save_path = os.path.join(FIGURES_DIR, f"prepared_sample_{i}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_path}")

    print("\nVisualization complete.")
