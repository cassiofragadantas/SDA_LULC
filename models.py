# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:44:56 2021

@author: emmanuel
"""

# importation des modules nécessaires
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Dense, Activation, Dropout
from tensorflow.keras.layers import BatchNormalization, RNN

tf.keras.backend.set_floatx('float32')


class MLP(Layer):
  def __init__(self, drop_val=0.5, **kwargs):
    super(MLP, self).__init__(**kwargs) # Appel du constructeur parent
    self.dense1 = layers.Dense(64, activation="relu")
    self.dropout = layers.Dropout(drop_val)
    self.dense2 = layers.Dense(64, activation="relu")

  @tf.function#(experimental_relax_shapes=True)
  def call(self, inputs, training=False):

    emb = layers.Flatten()(inputs)
    emb = self.dense1(emb)
    emb = self.dropout(emb, training=training)
    emb = self.dense2(emb)

    return emb



###############################
# Définition d'un classifieur pour nb_class classes
# avec une couche complètement connectée de nb_units neurones
# sans couches de Batch Normalization
class Classifier(Layer):

  def __init__(self, nb_class, nb_units, drop_val=0.5, **kwargs):
    super(Classifier, self).__init__(**kwargs) # Appel du constructeur parent

    self.dense = layers.Dense(nb_units)
    self.act = layers.Activation('relu')
    self.dropout = layers.Dropout(drop_val)

    # couche de 8 neurones, 1 sortie par classe, la valeur de chaque sortie
    # = probabilité estimée que l'entrée corresponde ait cette classe
    # (fonction softmax)
    self.output_ = layers.Dense(nb_class, activation="softmax")

  @tf.function#(experimental_relax_shapes=True)
  def call(self, inputs, training=False):

    dense = self.dense(inputs)
    act = self.act(dense)
    dropout = self.dropout(act, training=training)
    return self.output_(dropout)


# Définition d'un bloc de convolution 1D
# sans couches de Batch Normalization
class Conv1D_bloc_Model(Layer):

  def __init__(self, filters_nb, kernel_size, drop_val, **kwargs):
    super(Conv1D_bloc_Model, self).__init__(**kwargs) # Appel du constructeur parent

    # Ajout d'un bloc de convolution comprenant :
    # - une couche de convolution avec 'filters_nb' filtres
    #   de taille 'kernel_size' et bord à zéro
    # - une couche d'activation avec la fonction rectifieur linéaire
    # - une couche de mise à zéro aléatoire avec un taux de 'drop_val'
    #   des valeurs en entrée
    self.conv1D = layers.Conv1D(filters_nb, kernel_size, padding="same",  kernel_initializer='he_normal')
    self.act = layers.Activation('relu')
    self.output_ = layers.Dropout(drop_val)

  @tf.function#(experimental_relax_shapes=True)
  def call(self, inputs, training=False):
    conv1D = self.conv1D(inputs)
    act = self.act(conv1D)
    return self.output_(act, training=training)


# Définition de l'encodeur du TempCNN avec 3 blocs de convolution 1D
# sans couches de Batch Normalization
class TempCNN_Encoder2(Layer):
#class TempCNN_Encoder(keras.Model):

  def __init__(self, drop_val=0.5, **kwargs):
    super(TempCNN_Encoder2, self).__init__(**kwargs) # Appel du constructeur parent

    self.conv_bloc1 = Conv1D_bloc_Model(64, 5, drop_val)
    self.conv_bloc2 = Conv1D_bloc_Model(64, 5, drop_val)
    self.conv_bloc3 = Conv1D_bloc_Model(64, 5, drop_val)

    #self.flatten = tf.keras.layers.GlobalAveragePooling1D()
    #self.avgP = layers.GlobalAveragePooling1D()
    self.flatten = layers.Flatten()

  @tf.function#(experimental_relax_shapes=True)
  def call(self, inputs, training=False):

    conv1 = self.conv_bloc1(inputs, training=training)
    conv2 = self.conv_bloc2(conv1, training=training)
    conv3 = self.conv_bloc3(conv2, training=training)
    #print(conv3.get_shape())

    #flatten = self.avgP(conv3)
    flatten = self.flatten(conv3)
    #print(flatten.get_shape())

    return flatten





# Définition de l'encodeur du TempCNN avec 3 blocs de convolution 1D
# sans couches de Batch Normalization
class TempCNN_Encoder(Layer):
#class TempCNN_Encoder(keras.Model):

  def __init__(self, drop_val=0.5, **kwargs):
    super(TempCNN_Encoder, self).__init__(**kwargs) # Appel du constructeur parent

    self.conv_bloc1 = Conv1D_bloc_Model(64, 5, drop_val)
    self.conv_bloc2 = Conv1D_bloc_Model(64, 5, drop_val)
    self.conv_bloc3 = Conv1D_bloc_Model(64, 5, drop_val)

    #self.flatten = tf.keras.layers.GlobalAveragePooling1D()
    self.flatten = layers.Flatten()

  @tf.function#(experimental_relax_shapes=True)
  def call(self, inputs, training=False):
      #print(inputs.get_shape())
      conv1 = self.conv_bloc1(inputs, training=training)
      #print(conv1.get_shape())
      conv2 = self.conv_bloc2(conv1, training=training)
      #print(conv2.get_shape())
      conv3 = self.conv_bloc3(conv2, training=training)
      #print(conv3.get_shape())

      flatten = self.flatten(conv3)

      return flatten



@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad


class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)

# Définition du modèle TempCNN avec 3 blocs de convolution 1D
# sans couches de Batch Normalization
class DANN(keras.Model):
  def __init__(self, nb_class, drop_val=0.5, **kwargs):
    super(DANN, self).__init__(**kwargs) # Appel du constructeur parent

    self.encoder = TempCNN_Encoder()
    #self.encoder = TempCNN_Encoder2()
    #self.encoder = MLP()
    self.labelClassif = Classifier(nb_class, 256)
    self.grl = GradReverse()
    self.domainClassif = Classifier(2, 256)
    self.jigsawClassif = Classifier(6, 256)
    #self.projHead = layers.Dense(256, activation=None)

  @tf.function#(experimental_relax_shapes=True)
  def call(self, inputs, training=False):
    enc_out = self.encoder(inputs, training=training)
    #enc_out = tf.math.l2_normalize(enc_out)
    labelClassif = self.labelClassif(enc_out)
    grl = self.grl(enc_out)
    #grl = enc_out
    #return self.labelClassif(enc_out), self.domainClassif(grl), self.projHead(enc_out), self.jigsawClassif(enc_out)
    return labelClassif, self.domainClassif(grl), enc_out, self.jigsawClassif(enc_out)
