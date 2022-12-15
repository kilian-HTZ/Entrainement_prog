# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:19:00 2022

@author: Legion5
"""


#import des librairies
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers #contient les méthode d'optimisation (DSG,DG etc)
from tensorflow.keras.models import Sequential #construction sequentiel (couche par couche)
from tensorflow.keras.layers import Dense #couche de neurones dense 



#import des données
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical #fonction nous permetant de créer la matrice des classes binaire (1=(0,1,0,0,0,0,0,0,0,0))

(X_train_data,Y_train_data),(X_test_data,Y_test_data)=mnist.load_data()


#######
#Données d'entrainement du réseau de neurone 
#######

#traitement des données pour les rendres exploitablent par le réseau
N = X_train_data.shape[0]
X_train=np.reshape(X_train_data,(N,784))#création des vecteurs images
X_train=X_train/255 #♣normalisation des données

#création du vecteur de sortie du réseaux
Y_train=to_categorical(Y_train_data,num_classes=10)#création de la matrice binaires de sortie du réseau


######
#Données de test du réseau
######

#données test entrée
X_test=np.reshape(X_test_data,(X_test_data.shape[0],784))
X_test=X_test/255 #on pourrait aussi chercher le max et ne pas utiliser le max théorique afin d'éviter le problème si les images sont sous exposées

#données test de sortie
Y_test= to_categorical(Y_test_data,num_classes=10)



#######
#création du réseau de neurones et recherche des poids
#######


p=8

modele=Sequential()
modele.add(Dense(p,input_dim=784,activation='sigmoid')) #couche d'entré dense (input vecteur image de dim 784 pour 8 neurones d'activation sigma)
modele.add(Dense(p,activation='sigmoid'))#couche 2 dense de 8 neurones d'activation sigma
modele.add(Dense(10,activation='softmax'))#couche de sortie dense output 10 car nolmbre de (0 à 9) activation softmax 

#recherche des poids du réseau en minimisant l'entropie croisée via descente de gradiant stochastique 

modele.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy']) #accuracy car on prend en compte la précision
modele.fit(X_train,Y_train,batch_size=100,epochs=60)#paramétrage du grad stochastique pas batch de 32

#résultats
resultat = modele.evaluate(X_test, Y_test, verbose=0) 
print('Valeur de l''erreur sur les données de test (loss):', resultat[0])
print('Précision sur les données de test (accuracy):', resultat[1])


Y_predict = modele.predict(X_test)

# Un exemple
i = 48 # numéro de l'image
chiffre_predit = np.argmax(Y_predict[i]) # prend la proba max de la sortie 
print("Sortie réseau", Y_predict[i])
print("Chiffre attendu :", Y_test_data[i])
print("Chiffre prédit :", chiffre_predit)
plt.imshow(X_test_data[i], cmap='Greys')
plt.show()