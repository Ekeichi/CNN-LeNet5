import numpy as np
from models.layers import Convolutional, Fc, avg_pooling, sigmoid, Relu


class CNN:
    def __init__(self):
        # Initialisation des couches
        self.conv1 = Convolutional((32, 1, 32, 32), 5, 6, stride=1, padding=0)
        self.pool1 = avg_pooling()
        self.sig1 = Relu()
        self.conv2 = Convolutional((32, 1, 14, 14), 5, 16, stride=1, padding=0)
        self.pool2 = avg_pooling()
        self.sig2 = Relu()
        self.conv3 = Convolutional((32, 1, 5, 5), 5, 120, stride=1, padding=0)
        self.Fc1 = Fc(120, 84)
        self.sig3 = Relu()
        self.Fc2 = Fc(84, 10)

    def forward(self, X):
        # Passer les données à travers les couches
        X = self.conv1.forward(X)
        X = self.pool1.forward(X, 2, 2)
        X = self.sig1.forward(X)
        X = self.conv2.forward(X)
        X = self.pool2.forward(X, 2, 2)
        X = self.sig2.forward(X)
        X = self.conv3.forward(X)

        self.X_shape_before_reshape = X.shape

        X = X.reshape(X.shape[0], -1)
        X = self.Fc1.forward(X)
        X = self.sig3.forward(X)
        X = self.Fc2.forward(X)
        return X

    def backward(self, d_out, lr):
        # Calcul des gradients
        d_out = self.Fc2.backward(d_out, lr)
        d_out = self.sig3.backward(d_out)
        
        d_out = self.Fc1.backward(d_out, lr)
        d_out = d_out.reshape(self.X_shape_before_reshape)
        d_out = self.conv3.backward(d_out, lr)
        d_out = self.sig2.backward(d_out)
        d_out = self.pool2.backward(d_out)
        d_out = self.conv2.backward(d_out, lr)
        d_out = self.sig1.backward(d_out)
        d_out = self.pool1.backward(d_out)
        d_out = self.conv1.backward(d_out, lr)


    def save_model(self, filename):
        # Sauvegarde de tous les paramètres du modèle
        np.savez(filename, 
            conv1_weights=self.conv1_weights,
            conv1_bias=self.conv1_bias,
            conv2_weights=self.conv2_weights,
            conv2_bias=self.conv2_bias,
            fc1_weights=self.fc1_weights,
            fc1_bias=self.fc1_bias,
            fc2_weights=self.fc2_weights,
            fc2_bias=self.fc2_bias,
            output_weights=self.output_weights,
            output_bias=self.output_bias
        )
        print(f"Modèle LeNet-5 sauvegardé dans {filename}")