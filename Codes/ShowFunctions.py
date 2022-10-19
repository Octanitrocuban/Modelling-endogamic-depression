# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 22:53:41 2022

@author: Matthieu Nougaret
"""
import numpy as np
import matplotlib.pyplot as plt
#=============================================================================
def plotEvolGen(ArrOfMut, FgSz=(16, 6), Leg=False):
	"""
	This fuction is used to show the evolution of the proportion of mutation
	in the population through generation.

	Parameters
	----------
	ArrOfMut : numpy.ndarray
		A 2 dimensional array, where the n-th line correspond to the n-th
		generation, and the m-th columns correspond to the m-th number of
		people with m mutation. It correspond to the Mut_N output from
		PopulationsFunctions.Evoluteur().
	FgSz : tuple, optional
		Tuple or any other type of 1 dimensions vector with 2 elements. These
		elements can be int and/or float. They are the width and high of the
		figure. The default is (14, 6).
	Leg : bool, optional
		Boolean to show (True) or not (False) le legend. The default is False.

	Returns
	-------
	None.

	"""
	Nnb = np.sum(ArrOfMut, axis=1)
	plt.figure(figsize=FgSz)
	plt.title("Proportion of mutant per generation", fontsize = 22)
	plt.grid(True, zorder=1)
	for i in range(ArrOfMut.shape[1]):
		plt.plot(ArrOfMut[:, i]/Nnb, '.-', label=str(i), zorder=2)
	plt.plot([0, len(Nnb)-1], [0.5, 0.5], 'k', zorder=2)
	plt.xlabel("Generations", fontsize=15)
	plt.ylabel("% of the population in the n-th generation", fontsize=15)
	plt.xticks(range(len(ArrOfMut)), range(len(ArrOfMut)),
			   fontsize=14, rotation=90)
	if Leg:
		plt.legend(fontsize=13)
	plt.xlim(-.1, len(ArrOfMut)-.9)
	plt.show()
	return

def EvolLenPopShow(ArrLenPop, ArrSumPop, FgSz=(10, 7)):
	"""
	Represents changes in population size over generations.

	Parameters
	----------
		ArrLenPop: array|list|tuple
			Nombre d'individus par génération 
		ArrSumPop: array|list|tuple
			Somme commulé du nombre d'individus par génération
		FgSz: tuple|list|array d'int & float
			Largeur et hauteur de la figure.

	Returns
	-------
	None.

	"""
	plt.figure(figsize=FgSz)
	plt.subplot(1, 2, 1)
	plt.title("Evolution of the population", fontsize=22)
	plt.grid(True, zorder=1)
	plt.plot(ArrLenPop, '.-', label="Cumulated", zorder=2)
	plt.legend(fontsize=15, loc='upper left')
	plt.xlabel("Generation", fontsize=15)
	plt.ylabel("Population", fontsize=15)
	plt.xticks(range(len(ArrLenPop)), range(len(ArrLenPop)),
			   fontsize=14, rotation=90)
	plt.yticks(fontsize=14)
	plt.subplot(1, 2, 2)
	plt.title("Cumulative evolution of the population", fontsize=22)
	plt.grid(True, zorder=1)
	plt.plot(ArrSumPop, '.-', label="By generation", zorder=2)
	plt.legend(fontsize=15, loc='upper left')
	plt.xlabel("Generation", fontsize=15)
	plt.ylabel("Cumulaive population", fontsize=15)
	plt.xticks(range(len(ArrLenPop)), range(len(ArrLenPop)),
			   fontsize=14, rotation=90)
	plt.yticks(fontsize=14)
	plt.show()
	return

def GenealogicTree(ArrOfPop, Figsize=(16, 10), STitle=20):
	"""
	Represents the family tree from the LUCA ancestor (identifier -1) of all
	individuals generated during the simulation. Gender and descent/ancestry
	links are indicated.

	Parameters
	----------
	ArrOfPop : numpy.ndarray
		A 2 dimensions array. It is the array of all of the peoples generated
		during simulation. Corresponds to the name variable 'Tree' rendered
		by the function Evoluteur().
	Figsize : tuple, optional
		Size of the fisure. The default is (16, 10).
	STitle : int, optional
		Size of the title. The default is 20.

	Returns
	-------
	None

	"""
	NCaract = np.array([0, 2, 3, 4])#(self, mother, father, generation) number
	ArrGene = (ArrOfPop[:, NCaract]).astype(int).T
	cent = 2
	Tree = np.array([ArrGene[0], ArrGene[3], ArrGene[1], ArrGene[2]]).T
	TreeCut, PopG = [], []
	for i in range(Tree[-1, 1]+1):
		TreeCut.append(Tree[Tree[:, 1] == i])
		PopG.append(len(TreeCut[i]))
	TreeCut = np.array(TreeCut, dtype=object)
	L = np.max(PopG)
	#Centered on cent, between 0.5 and 1.5
	plt.figure(figsize=Figsize)
	plt.title("Genealogic tree", fontsize=STitle)
	plt.plot(-1, cent, 'k.', label="Peoples", markersize=12, zorder=2)
	plt.plot(-1, cent, 'k*', label="'LUCA'", markersize=14, zorder=3)
	for i in range(len(TreeCut)):
		brc1 = np.copy(TreeCut[i])
		y1 = (brc1[:, 0]-brc1[0, 0])/L+cent
		y1 = y1-np.max(y1)/2+(cent/2)
		x1 = np.array([i]*PopG[i])
		plt.plot(x1, y1, 'k.', markersize=8, zorder=3)
		# dico relation number
		if i == 0:
			link_x1 = np.zeros(PopG[i])-1
			link_x2 = np.zeros(PopG[i])
			link_y1 = np.zeros(PopG[i])+cent
			link_y2 = y1.astype(float)
			for j in range(PopG[i]):
				plt.plot([link_x1[j], link_x2[j]], [link_y1[j], link_y2[j]],
						 '-', color=[1, 0, 1], zorder=2)
		else:
			brc2 = np.copy(TreeCut[i-1])#preious generation
			min_r2 = np.min(brc2[:, 0])#min self n° previous
			min_r1 = np.min(brc1[:, 0])#min self n° current
			y2 = (brc2[:, 0]-brc2[0, 0])/L+cent
			y2 = y2-np.max(y2)/2+(cent/2)
			x2 = i-1
			for j in range(len(brc1)):
				ym = y2[int(brc1[j, 2]-min_r2)]
				yp = y2[int(brc1[j, 3]-min_r2)]
				yi = y1[int(brc1[j, 0]-min_r1)]
				plt.plot([x2, x1[0]], [ym, yi], 'b-', zorder=2)
				plt.plot([x2, x1[0]], [yp, yi], 'r-', zorder=2)
	plt.plot(-1, cent, "b-",
		  label="From left to right :\nMother->child")
	plt.plot(-1, cent, "r-",
		  label="From left to right :\nFather->child")
	plt.xlabel("Generations", fontsize=18)
	plt.legend(fontsize=15, loc='upper left')
	plt.xticks(range(-1, Tree[-1, 1]+1), range(-1, Tree[-1, 1]+1),
			   fontsize=14)
	plt.yticks(fontsize=0)
	plt.show()
	return

def InformativLinearTree(ArrOfPop, Figsize=(18, 14), STitl=22, Sscat=30,
						 Cmap='jet'):
	"""
	This fuction used to show the genealogical link between the peoples
	created during the simulation, and their mutation rate (=number of mutated
	gens/total number of gens).

	Parameters
	----------
	ArrOfPop : numpy.ndarray
		A 2 dimensions array. It is the array of all of the peoples generated
		during simulation. Corresponds to the name variable 'Tree' rendered
		by the function Evoluteur().
	Figsize : tuple, optional
		Size of the fisure. The default is (16, 10).
	STitl : int, optional
		Size of the title. The default is 20.
	Sscat : int, optional
		Size of the dots. The default is 30.
	Cmap : str, optional
		Name of the color map used for the mutation rate. The default is
		'jet'. I try different color map, and the most readeball were:
		['autumn', 'brg', 'plasma', 'gist_rainbow', 'gnuplot', 'jet',
		 'turbo', 'viridis'] 

	Returns
	-------
	None.

	"""
	PercMut = np.zeros(ArrOfPop[:, 5:].shape)
	PercMut[ArrOfPop[:, 5:] == 'aa'] = 2
	PercMut[ArrOfPop[:, 5:] == 'Aa'] = 1
	PercMut[ArrOfPop[:, 5:] == 'aA'] = 1
	PercMut = np.sum(PercMut, axis=1)/(len(ArrOfPop[0, 5:])*2)
	vals, PopG = np.unique(ArrOfPop[:, 4].astype(int), return_counts=True)
	NCaract = np.array([0, 2, 3, 4])
	ArrGene = (ArrOfPop[:, NCaract]).astype(int).T
	Tree = np.array([ArrGene[0], ArrGene[3], ArrGene[1], ArrGene[2]]).T
	TreeCut = [] ; Gend = [] ; Mutcut = []
	for i in range(Tree[-1, 1]+1):
		TreeCut.append(Tree[Tree[:, 1] == i])
		Gend.append(ArrOfPop[Tree[:, 1] == i , 1])
		Mutcut.append(PercMut[Tree[:, 1] == i])
	TreeCut = np.array(TreeCut, dtype=object)
	Mutcut = np.array(Mutcut, dtype=object)
	L = np.max(PopG) ; nbg = np.max(vals)
	cent = 2
	plt.figure(figsize=Figsize)
	plt.title("Genealogical tree & mutation rate", fontsize=STitl)
	plt.plot(-1, cent, 'k*', markersize=15, zorder=3, label='LUCA')
	plt.plot(-4, cent, 'ko', markersize=10, zorder=2, label='Woman')
	plt.plot(-4, cent, 'k^', markersize=10, zorder=1, label='Man')
	for i in range(len(TreeCut)):
		brc1 = np.copy(TreeCut[i])
		y1 = (brc1[:, 0]-brc1[0, 0])/L+cent
		y1 = y1-np.max(y1)/2+(cent/2)
		x1 = np.array([i]*PopG[i])
		plt.scatter(x1[Gend[i] == 'f'], y1[Gend[i] == 'f'], cmap=Cmap,
					s=Sscat, c=Mutcut[i][Gend[i] == 'f'], marker='o',
					zorder=3, vmin=0, vmax=1)
		plt.scatter(x1[Gend[i] == 'm'], y1[Gend[i] == 'm'], cmap=Cmap,
					s=Sscat, c=Mutcut[i][Gend[i] == 'm'], marker='^',
					zorder=3, vmin=0, vmax=1)
		if i == 0:
			link_x1 = np.zeros(PopG[i])-1
			link_x2 = np.zeros(PopG[i])
			link_y1 = np.zeros(PopG[i])+cent
			link_y2 = y1.astype(float)
			for j in range(PopG[i]):
				plt.plot([link_x1[j], link_x2[j]], [link_y1[j], link_y2[j]],
						 '-', color=[1, 0, 1], zorder=1)
		else:
			brc2 = np.copy(TreeCut[i-1])
			min_r2 = np.min(brc2[:, 0])
			min_r1 = np.min(brc1[:, 0])
			y2 = (brc2[:, 0]-brc2[0, 0])/L+cent
			y2 = y2-np.max(y2)/2+(cent/2)
			x2 = i-1
			for j in range(len(brc1)):
				ym = y2[int(brc1[j, 2]-min_r2)]
				yp = y2[int(brc1[j, 3]-min_r2)]
				yi = y1[int(brc1[j, 0]-min_r1)]
				plt.plot([x2, x1[0]], [ym, yi], 'b-', zorder=1)
				plt.plot([x2, x1[0]], [yp, yi], 'r-', zorder=1)
	plt.plot(-1, cent, 'b-',
			 label="From left to right :\nMother->child(s)")
	plt.plot(-1, cent, 'r-',
			 label="From left to right :\nFather->child(s)")
	plt.xlabel('Generation\n--------------------\nMutation Rate', fontsize=15)
	plt.xticks(range(-1, nbg+1), range(-1, nbg+1), fontsize=14)
	plt.yticks(fontsize=0)
	plt.legend(fontsize=15, loc='upper left')
	plt.xlim(-2, nbg+1)
	plt.colorbar(shrink=0.7, orientation="horizontal", pad=0.12, aspect=50)
	plt.show()
	return
