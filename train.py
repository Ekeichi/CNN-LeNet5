import numpy as np
import time
import os

from utils.dataloader import load_mnist, DataLoader
from models.cnn import CNN
from utils.metrics import cross_entropy_loss, one_hot_encode
from evaluate import predict_batch, softmax
import matplotlib.pyplot as plt

# Créer dossier pour sauvegarder les modèles
os.makedirs('saved_models', exist_ok=True)

# Hyperparamètres
batch_size = 32
epochs = 10  # Augmenté à 10 époques pour un meilleur apprentissage
learning_rate = 0.001
train_limit = 5000  # Utiliser plus d'exemples pour un meilleur entraînement
test_limit = 1000

# Charger les données MNIST
print("Chargement des données MNIST...")
(train_images, train_labels), (test_images, test_labels) = load_mnist(train_limit=train_limit, test_limit=test_limit)
train_loader = DataLoader(train_images, train_labels, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_images, test_labels, batch_size=batch_size, shuffle=True)

print(f"Nombre d'images d'entraînement: {len(train_images)}")
print(f"Nombre d'images de test: {len(test_images)}")

# Initialiser le modèle
model = CNN()

loss_values = []
accuracy_values = []
best_accuracy = 0

start_time = time.time()

for epoch in range(epochs):
    epoch_start_time = time.time()
    epoch_loss = 0
    batch_count = 0
    
    # Parcourir tous les batchs de l'époque
    train_loader_iter = iter(train_loader)
    for batch in train_loader_iter:
        images_batch, labels_batch = batch
        
        # Forward pass
        output = model.forward(images_batch)
        Y_true = one_hot_encode(labels_batch, num_classes=4)  # 4 classes (0, 1, 2, 3)
        loss, grad = cross_entropy_loss(Y_true, output)
        
        # Backward pass
        model.backward(grad, learning_rate)
        epoch_loss += loss
        batch_count += 1
    
    # Stocker la perte moyenne de l'époque
    avg_loss = epoch_loss / batch_count
    loss_values.append(avg_loss)
    
    # Évaluer sur l'ensemble de test
    correct_predictions = 0
    total_predictions = 0
    
    for batch in test_loader:
        images_batch, labels_batch = batch
        logits = model.forward(images_batch)
        probabilities = softmax(logits)
        predictions = np.argmax(probabilities, axis=1)
        
        correct_predictions += np.sum(predictions == labels_batch)
        total_predictions += len(labels_batch)
    
    accuracy = (correct_predictions / total_predictions) * 100
    accuracy_values.append(accuracy)
    
    epoch_time = time.time() - epoch_start_time
    
    print(f"Époque {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}% - Temps: {epoch_time:.2f}s")
    
    # Sauvegarder le meilleur modèle
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        model.save_model(f'saved_models/lenet5_best.npz')

# Sauvegarder le modèle final
model.save_model(f'saved_models/lenet5_final.npz')

total_time = time.time() - start_time
print(f"\nEntraînement terminé en {total_time:.2f} secondes!")
print(f"Meilleure précision: {best_accuracy:.2f}%")

# Afficher la courbe de perte
plt.figure(figsize=(12, 5))

# Courbe de perte
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), loss_values, marker='o', label='Loss', color='blue')
plt.title('Courbe de perte', fontsize=14)
plt.xlabel('Époque', fontsize=12)
plt.ylabel('Perte', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Courbe de précision
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), accuracy_values, marker='o', label='Accuracy', color='green')
plt.title('Courbe de précision', fontsize=14)
plt.xlabel('Époque', fontsize=12)
plt.ylabel('Précision (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# Évaluation finale
print("\nÉvaluation finale sur l'ensemble de test:")
correct_predictions = 0
total_predictions = 0

class_correct = np.zeros(4)  # 4 classes (0, 1, 2, 3) 
class_total = np.zeros(4)  # 4 classes (0, 1, 2, 3)

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

accuracy = (correct_predictions / total_predictions) * 100
print(f"Précision globale: {accuracy:.2f}%")

# Afficher la précision par classe
print("\nPrécision par classe:")
for i in range(4):  # 4 classes (0, 1, 2, 3)
    if class_total[i] > 0:
        class_accuracy = (class_correct[i] / class_total[i]) * 100
        print(f"Classe {i}: {class_accuracy:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")