# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 21:06:54 2022

@author: Legion5
"""


#Import des blibliothèques 
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from mpl_toolkits.mplot3d import axes3d



"""
Ce première exemple concerne un réseaux de deuxcouches de neurones dense avec 1 entrée et une sortie.
La fonction d'activation est Relu.

"""
 
#création des couches

modele=Sequential()#définition du type d'architecture 

modele.add(Dense(2,input_dim=1,activation='relu'))  #création de la première couche dense avec 1 entrée"
modele.add(Dense(1,activation='relu')) #création de la deuxième couche dense avec 1 neurones

print(modele.summary()) #affiche le résumé de l'architectue du réseaux

#initialisation des poids

#couche 0
coeff=np.array([[1,-0.5]])
biais=np.array([-1,1])
poids=[coeff,biais]
modele.layers[0].set_weights(poids) #on définit les paramètres de la couche 0

#couche1
coeff=np.array([[1],[1]])
biais=np.array([0])
poids=[coeff,biais]
modele.layers[1].set_weights(poids) #on définit les paramètres de la couche 0

#utilisation du réseaux

entree=np.array([[3.0]]) #on test avec 3 en entrée la sortie doit être 2
sortie= modele.predict(entree)
print (sortie)

liste_x=np.linspace(-2,3,num=100)
entree=np.array([[x] for x in liste_x])
sortie=modele.predict(entree)
liste_y=np.array([y[0] for y in sortie])
plt.plot(liste_x,liste_y)
plt.show()

"""
On va créer un réseau de neurones disosant de deux entrée x et y et d'une sortie F(x,y)
composé de deux couches la premières composée de 3 neurones de fonc d'activation sigmoid
la deuxième permettant d'aditioner les sorties de la couche 1
"""

#création des couches
modele1=Sequential()
modele1.add(Dense(3,input_dim=2,activation='sigmoid'))
modele1.add(Dense(1,activation='sigmoid'))
print(modele.summary())

#couche0
coeff=np.array([[1,3,-5],[2,-4,-6]]) #preier elt sont les coeffs de x et le deuxième elt sont les coeff de y
biais=np.array([-1,0,1])
poids=[coeff,biais]
modele1.layers[0].set_weights(poids)

#couche1
coeff=np.array([[1],[1],[1]])
biais=np.array([-3])
poids=[coeff,biais]
modele1.layers[1].set_weights(poids)


#utilisation tracé du graph
VX = np.linspace(-5, 5, 100)
VY = np.linspace(-5, 5, 100)
X,Y = np.meshgrid(VX, VY)
entree = np.c_[X.ravel(), Y.ravel()]
sortie = modele1.predict(entree)
Z = sortie.reshape(X.shape)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()