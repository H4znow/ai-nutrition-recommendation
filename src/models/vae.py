"""
 A deep generative network architecture to create weekly meal plans by employing sophisticated 
 loss functions to align the network with well-founded nutritional guidelines
 src : "AI nutrition recommendation using a deep generative model and ChatGPT", Ilias Papastratis & al., 2024
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, random


class Encoder(Model):
    def __init__(self, latent_dim, hidden_units):
        super(Encoder, self).__init__()
        self.danse1 = layers.Dense(hidden_units, activation='relu')
        self.danse2 = layers.Dense(hidden_units, activation='relu')
        self.fc_mu = layers.Dense(latent_dim)
        self.fc_logvar = layers.Dense(latent_dim)

    def call(self, x):
        h = self.dense1(x)
        h = self.dense2(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

def reparametrization(mu, logvar):
    epsilon = random.normal(shape=mu.shape, mean=0.0, stddev=1.0)
    z = mu + tf.exp(logvar * 0.5) * epsilon
    return z