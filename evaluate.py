import numpy as np

def softmax(logits):
    """
    Applique la fonction softmax à chaque élément d'un batch.
    :param logits: Tableau de logits de forme (batch, n_classes)
    :return: Probabilités de forme (batch, n_classes)
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Stabilisation numérique
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def predict_batch(model, batch_images):
    """
    Effectue des prédictions sur un batch d'images.
    
    :param model: Le modèle CNN.
    :param batch_images: Batch d'images de forme (N, C, H, W).
    :return: Prédictions des classes pour chaque image dans le batch.
    """
    # Forward pass
    logits = model.forward(batch_images)  # (N, n_classes)
    
    # Appliquer la softmax pour obtenir les probabilités
    probabilities = softmax(logits)  # (N, n_classes)
    
    # Prédire la classe (indice de la probabilité maximale)
    predictions = np.argmax(probabilities, axis=1)  # (N,)
    return predictions


