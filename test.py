import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from utils.dataloader import load_mnist, DataLoader
from models.cnn import CNN
from evaluate import softmax

def test_model(model_path=None, test_samples=1000, batch_size=32, display_examples=True):
    """
    Teste un modèle CNN sur l'ensemble de test MNIST.
    
    Args:
        model_path: Chemin vers le modèle sauvegardé (.npz)
        test_samples: Nombre d'échantillons de test à utiliser
        batch_size: Taille du batch pour le test
        display_examples: Afficher quelques exemples de prédictions
    """
    # Charger les données de test
    print("Chargement des données de test...")
    (_, _), (test_images, test_labels) = load_mnist(test_limit=test_samples)
    test_loader = DataLoader(test_images, test_labels, batch_size=batch_size, shuffle=True)
    
    # Créer et charger le modèle
    model = CNN()
    if model_path and os.path.exists(model_path):
        print(f"Chargement du modèle depuis {model_path}...")
        model.load_model(model_path)
    else:
        print("Utilisation d'un modèle non entraîné (résultats aléatoires attendus)")
    
    # Évaluer le modèle
    correct_predictions = 0
    total_predictions = 0
    class_correct = np.zeros(4)  # 4 classes (0, 1, 2, 3)
    class_total = np.zeros(4)  # 4 classes (0, 1, 2, 3)
    
    # Pour l'affichage des exemples
    examples_to_show = []
    
    for batch in test_loader:
        images_batch, labels_batch = batch
        logits = model.forward(images_batch)
        probabilities = softmax(logits)
        predictions = np.argmax(probabilities, axis=1)
        
        # Précision globale
        correct_predictions += np.sum(predictions == labels_batch)
        total_predictions += len(labels_batch)
        
        # Précision par classe
        for i in range(len(labels_batch)):
            label = labels_batch[i]
            class_total[label] += 1
            if predictions[i] == label:
                class_correct[label] += 1
            
            # Collecter quelques exemples pour affichage
            if len(examples_to_show) < 10 and i % 10 == 0:
                examples_to_show.append((
                    images_batch[i, 0],  # Image (prendre le premier canal)
                    label,               # Vrai label
                    predictions[i],      # Prédiction
                    probabilities[i]     # Probabilités
                ))
    
    # Afficher les résultats
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\nPrécision globale: {accuracy:.2f}%")
    
    print("\nPrécision par classe:")
    for i in range(4):  # 4 classes (0, 1, 2, 3)
        if class_total[i] > 0:
            class_accuracy = (class_correct[i] / class_total[i]) * 100
            print(f"Classe {i}: {class_accuracy:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
    
    # Afficher des exemples de prédictions
    if display_examples and examples_to_show:
        plt.figure(figsize=(15, 8))
        for i, (img, true_label, pred_label, probs) in enumerate(examples_to_show):
            if i >= 5:  # Limiter à 5 exemples
                break
                
            plt.subplot(1, 5, i+1)
            plt.imshow(img, cmap='gray')
            
            # Colorier le titre en fonction de la prédiction (vert si correct, rouge si incorrect)
            title_color = 'green' if true_label == pred_label else 'red'
            plt.title(f"Vrai: {true_label}\nPréd: {pred_label}", color=title_color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('test_examples.png')
        plt.show()
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tester le modèle LeNet-5")
    parser.add_argument("--model", type=str, default="saved_models/lenet5_best.npz", 
                        help="Chemin vers le modèle sauvegardé")
    parser.add_argument("--samples", type=int, default=1000, 
                        help="Nombre d'échantillons de test à utiliser")
    args = parser.parse_args()
    
    test_model(model_path=args.model, test_samples=args.samples)