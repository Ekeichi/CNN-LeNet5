import numpy as np

from utils.dataloader import load_mnist, DataLoader
from models.cnn import CNN
from utils.metrics import cross_entropy_loss, one_hot_encode
from evaluate import predict_batch, softmax
import matplotlib.pyplot as plt



# Charger les données MNIST
(train_images, train_labels), (test_images, test_labels) = load_mnist(train_limit=1000, test_limit=100)
train_loader = DataLoader(train_images, train_labels, batch_size=32, shuffle=True)
test_loader = DataLoader(test_images, test_labels, batch_size=32, shuffle=True)

epochs = 2
model = CNN()

loss_values = []

for epoch in range(epochs):
    lr = 0.001
    epoch_loss = 0
    
    # Parcourir tous les batchs de l'époque
    train_loader_iter = iter(train_loader)
    for batch in train_loader_iter:

        images_batch, labels_batch = batch
        
        output = model.forward(images_batch)
        Y_true = one_hot_encode(labels_batch)
        loss, grad = cross_entropy_loss(Y_true, output)
        
        model.backward(grad, lr)
        epoch_loss += loss
    
    # Stocker la perte moyenne de l'époque
    avg_loss = epoch_loss / 32 # taille du batch, a modifer en cas de changement
    loss_values.append(avg_loss)
    
    if epoch % 1 == 0:
        print(f"{100*epoch/epochs:.0f}% iter: {epoch}, loss: {avg_loss}")

e = range(1, len(loss_values) + 1)
plt.figure(figsize=(10, 6))
plt.plot(e, loss_values, marker='o', label='Loss per epoch', color='blue')
plt.title('Loss Curve During Training', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.show()



images_batch, labels_batch = next(iter(test_loader))
# logits = model.forward(images_batch)  # (N, n_classes)
    
# # Appliquer la softmax pour obtenir les probabilités
# probabilities = softmax(logits)  # (N, n_classes)
    
# # Prédire la classe (indice de la probabilité maximale)
# predictions = np.argmax(probabilities, axis=1)  # (N,)
# # Comparer les prédictions aux labels réels
# accuracy = np.mean(predictions == labels_batch) * 100

# print(f"Précision sur le batch : {accuracy:.2f}%")


correct_predictions = 0
total_predictions = 0

# Parcourir tous les batchs de test
for batch in test_loader:
    images_batch, labels_batch = batch
    logits = model.forward(images_batch)
    probabilities = softmax(logits)
    predictions = np.argmax(probabilities, axis=1)
    
    correct_predictions += np.sum(predictions == labels_batch)
    total_predictions += len(labels_batch)
    
accuracy = (correct_predictions / total_predictions) * 100
print(f"Précision sur l'ensemble de test : {accuracy:.2f}%")