# LeNet-5 en NumPy

J'ai developpé une implémentation de Lenet-5 (https://en.wikipedia.org/wiki/LeNet) en utilisant seulement Numpy pour faire chaque couche de l'architecture. J'ai également utilisé torchvision pour récupérer le dataset du MNIST. En reduisant le nombre de chiffres présent dans le dataset, j'arrive à le faire tourner relativement vite sur un MacBook Pro M1 de 2020. 

## Architecture

L'architecture LeNet-5 implémentée dans ce projet comprend:

- Couche convolutive (6 filtres, noyau 5x5)
- Pooling moyen 2x2
- Activation ReLU
- Couche convolutive (16 filtres, noyau 5x5)
- Pooling moyen 2x2
- Activation ReLU
- Couche convolutive (120 filtres, noyau 5x5)
- Couche entièrement connectée (84 neurones)
- Activation ReLU
- Couche de sortie (4 classes)

## Utilisation

### Entraînement

Pour entraîner le modèle:

```bash
python3 train.py
```

Cela entraînera le modèle avec les paramètres par défaut et sauvegardera:
- Le meilleur modèle (selon la précision) dans `saved_models/lenet5_best.npz`
- Le modèle final dans `saved_models/lenet5_final.npz`
- Une visualisation des courbes d'apprentissage dans `training_curves.png`

### Courbes d'entraînement

Après l'entraînement, les courbes suivantes sont générées:

![Courbes d'entraînement](training_curves.png)


### Test

Pour tester un modèle entraîné:

```bash
python3 test.py --model saved_models/lenet5_best.npz --samples 1000
```


### Exemples de prédictions

![Exemples de prédictions](test_examples.png)


## Performances

Sur un MacBook Pro M1, le modèle atteint généralement:
- Une précision d'environ 85 - 90% sur les chiffres 0, 1, 2 et 3
- Un temps d'entraînement inférieur à 15 minutes pour 10 époques

