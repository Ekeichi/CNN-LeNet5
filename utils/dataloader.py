import numpy as np
import torchvision
from torchvision import transforms
import random
import torch

def load_mnist(data_dir="./data", train_limit=None, test_limit=None):
    """
    Télécharge le dataset MNIST et le retourne sous forme de tableaux NumPy.
    """

    sample_ratio = 0.1
    # conversion pour numpy
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(padding=2, fill=0),
        transforms.Lambda(lambda x: x.numpy())
    ])
    
    # télécharger données
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True
    )


    # Sous-échantillonnage
    if train_limit:
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_limit))
    if test_limit:
        test_dataset = torch.utils.data.Subset(test_dataset, range(test_limit))
    
    # Filtrer pour garder uniquement les classes 0, 1, 2, 3
    train_dataset = [(img, label) for img, label in train_dataset if label in [0, 1, 2, 3]]
    test_dataset = [(img, label) for img, label in test_dataset if label in [0, 1, 2, 3]]

    # Extraire les images et les labels
    train_images = np.array([data[0] for data in train_dataset])  # Images d'entraînement
    train_labels = np.array([data[1] for data in train_dataset])  # Labels d'entraînement
    
    test_images = np.array([data[0] for data in test_dataset])  # Images de test
    test_labels = np.array([data[1] for data in test_dataset])  # Labels de test


    
    return (train_images, train_labels), (test_images, test_labels)


class DataLoader:
    def __init__(self, data, labels, batch_size, shuffle=True):
        """
        Initialisation du DataLoader.
        """
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.indices = np.arange(self.num_samples)
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        """Crée un itérateur pour parcourir les données."""
        self.current_idx = 0
        
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        return self
    
    def __next__(self):
        """Renvoie le prochain batch."""
        if self.current_idx >= self.num_samples:
            raise StopIteration
        
        end_idx = min(self.current_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[self.current_idx:end_idx]
        
        batch_data = self.data[batch_indices]
        batch_labels = self.labels[batch_indices]

        # Si le batch est incomplet, ajouter un padding
        if len(batch_data) < self.batch_size:
            raise StopIteration
        
        self.current_idx = end_idx
        
        return batch_data, batch_labels
    

    
