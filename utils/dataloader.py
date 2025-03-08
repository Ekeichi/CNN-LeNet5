import numpy as np
import torchvision
from torchvision import transforms
import random
import torch

def load_mnist(data_dir="./data", train_limit=None, test_limit=None):
    """
    Télécharge le dataset MNIST et le retourne sous forme de tableaux NumPy.
    
    Args:
        data_dir (str): Dossier où télécharger les données.

    Returns:
        (tuple): (train_images, train_labels), (test_images, test_labels)
    """

    sample_ratio = 0.1
    # Transformation pour convertir les images en tenseurs puis en tableaux NumPy
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convertit en tenseur PyTorch (C, H, W) normalisé entre [0, 1]
        transforms.Pad(padding=2, fill=0),
        transforms.Lambda(lambda x: x.numpy())  # Convertit en tableau NumPy
    ])
    
    # Télécharger les données
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

    # # Sous-échantillonner
    # train_indices = random.sample(range(len(train_dataset)), int(len(train_dataset) * sample_ratio))
    # test_indices = random.sample(range(len(test_dataset)), int(len(test_dataset) * sample_ratio))

    # train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    # test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

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
        
        :param data: numpy array des données (e.g., images de taille (N, H, W))
        :param labels: numpy array des labels (e.g., (N,))
        :param batch_size: taille des batchs
        :param shuffle: bool, si les données doivent être mélangées à chaque epoch
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
    

    