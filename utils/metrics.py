import numpy as np

def one_hot_encode(labels, num_classes = 10):
    one_hot_labels = np.zeros((len(labels), num_classes))  # Initialisation d'un tableau de zéros
    one_hot_labels[np.arange(len(labels)), labels] = 1  # Affectation de 1 aux positions des classes
    return one_hot_labels

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Stabilité numérique
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def cross_entropy_loss(y_true, logits):
    # Calcul des probabilités avec la softmax
    probs = softmax(logits)
    # Calcul de la perte pour chaque exemple
    loss = -np.sum(y_true * np.log(probs + 1e-9)) / y_true.shape[0]  # Ajout de 1e-9 pour éviter log(0)
    # Gradient de la perte par rapport aux logits
    grad = (probs - y_true) / y_true.shape[0]
    return loss, grad


