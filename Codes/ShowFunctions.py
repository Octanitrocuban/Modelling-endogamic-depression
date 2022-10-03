# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 22:53:41 2022

@author: Matthieu Nougaret
"""
import numpy as np
import matplotlib.pyplot as plt
#=============================================================================
def plotEvolGen(ArrOfMut, FgSz=(14, 6)):
	"""
	input:
		ArrNnMut: array du nombre de la proportion d'individus n'ayant aucune
		mutation par génération.
		Arr1Mut: array du nombre de la proportion d'individus ayant une seule
		mutation par génération.
		Arr2Mut: array du nombre de la proportion d'individus ayant deux
		mutation par génération.
		NbGéné: int, nombre de générations modélisé.
		FgSz: tuple|list|array d'int & float. Largeur et hauteur de la figure.
	"""
	Nnb = np.sum(ArrOfMut, axis=1)
	plt.figure(figsize=FgSz)
	plt.title("Proportion of mutant per generation", fontsize = 22)
	plt.grid(True, zorder=1)
	for i in range(ArrOfMut.shape[1]):
		plt.plot(ArrOfMut[:, i]/Nnb, '.-', label=str(i)+" mutation", zorder=2)
	plt.plot([0, len(Nnb)-1], [0.5, 0.5], 'k', zorder=2)
	plt.xlabel("Generations", fontsize=15)
	plt.ylabel("% of the population in the n-th generation", fontsize=15)
	plt.legend(fontsize=13, loc=[1.01, -0.1])
	plt.show()
	return

def EvolLenPopShow(ArrLenPop, ArrSumPop, FgSz=(10, 7)):
	"""Represents changes in population size over generations.

	Parameters
	----------
		ArrLenPop: array|list|tuple
			Nombre d'individus par génération 
		ArrSumPop: array|list|tuple
			Somme commulé du nombre d'individus par génération
		FgSz: tuple|list|array d'int & float
			Largeur et hauteur de la figure.
	"""
	plt.figure(figsize=FgSz)
	plt.title("Evolution of the population", fontsize=22)
	plt.grid(True, zorder=1)
	plt.plot(ArrLenPop, '.-', label="Cumulated", zorder=2)
	plt.plot(ArrSumPop, '.-', label="By generation", zorder=2)
	plt.legend(fontsize=15, loc='upper right')
	plt.xlabel("Generation", fontsize=15)
	plt.ylabel("Population in the n-th generation", fontsize=15)
	plt.show()
	return
