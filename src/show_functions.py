# -*- coding: utf-8 -*-
"""
Module with finction to create plots.
"""
import numpy as np
import matplotlib.pyplot as plt
#=============================================================================
def plot_evol_gen(arr_of_mut, figsize=(16, 6), leg=False, marge=0.01,
				  save=None):
	"""
	This fuction is used to show the evolution of the proportion of mutation
	in the population through generation.

	Parameters
	----------
	arr_of_mut : numpy.ndarray
		A 2 dimensional array, where the n-th line correspond to the n-th
		generation, and the m-th columns correspond to the m-th number of
		people with m mutation. It correspond to the Mut_N output from
		PopulationsFunctions.Evoluteur().
	figsize : tuple, optional
		Tuple or any other type of 1 dimensions vector with 2 elements. These
		elements can be int and/or float. They are the width and high of the
		figure. The default is (14, 6).
	leg : bool, optional
		Boolean to show (True) or not (False) le legend. The default is False.
	save : NoneType or str, optional
		If not None, it have to be str path to save the picture.

	Returns
	-------
	None.

	"""
	nnb = np.sum(arr_of_mut, axis=1)
	ampli = len(nnb)
	# x-axis limits
	lower = -ampli*marge
	upper = len(nnb)-1+ampli*marge

	plt.figure(figsize=figsize)
	plt.title("Proportion of mutant per generation", fontsize=22)
	plt.grid(True, zorder=1)
	for i in range(arr_of_mut.shape[1]):
		plt.plot(arr_of_mut[:, i]/nnb, '.-', label=str(i), zorder=2)

	plt.plot([0, len(nnb)-1], [0.5, 0.5], 'k', zorder=2)
	plt.xlabel("Generations", fontsize=15)
	plt.ylabel("% of the population", fontsize=15)
	plt.xticks(range(len(arr_of_mut)), range(len(arr_of_mut)),
			   fontsize=14, rotation=90)

	if leg:
		plt.legend(fontsize=13, ncol=2)

	plt.xlim(lower, upper)
	if type(save) == str:
		if save[-4:] not in ['.png', '.pdf', '.jpg', '.svg', '.tif']:
			save += '.png'

		plt.savefig(save, bbox_inches='tight')

	plt.show()

def plot_evol_consang(arr_of_pop, figsize=(16, 6), marge=0.01, save=None):
	"""
	This fuction is used to show the evolution of the consanguinity score's in
	the population through generation.

	Parameters
	----------
	arr_of_pop : numpy.ndarray
		A 2 dimensional array, where the n-th line correspond to the n-th
		generation, and the m-th columns correspond to the m-th number of
		people with m mutation. It correspond to the Mut_N output from
		PopulationsFunctions.Evoluteur().
	figsize : tuple, optional
		Tuple or any other type of 1 dimensions vector with 2 elements. These
		elements can be int and/or float. They are the width and high of the
		figure. The default is (14, 6).
	save : NoneType or str, optional
		If not None, it have to be str path to save the picture.

	Returns
	-------
	None.

	"""
	nnb = int(arr_of_pop[-1, 4])
	ampli = nnb
	# x-axis limits
	lower = -ampli*marge
	upper = nnb+ampli*marge
	# consaguinity score stored in the array of population
	consang = arr_of_pop[:, 5].astype(float)
	scores = np.zeros(nnb+1)
	for i in range(nnb+1):
		scores[i] = np.mean(consang[arr_of_pop[:, 4] == str(i)])

	plt.figure(figsize=figsize)
	plt.title("Evolution of consanguinity per generation", fontsize=22)
	plt.grid(True, zorder=1)
	plt.plot(scores, '.-', zorder=2)
	plt.plot([0, nnb], [0.5, 0.5], 'k', zorder=2)
	plt.xlabel("Generations", fontsize=15)
	plt.ylabel("Mean consanguinity scores", fontsize=15)
	plt.xticks(range(nnb+1), range(nnb+1), fontsize=14, rotation=90)
	plt.xlim(lower, upper)
	plt.ylim(-0.05, 1.05)
	if type(save) == str:
		if save[-4:] not in ['.png', '.pdf', '.jpg', '.svg', '.tif']:
			save += '.png'

		plt.savefig(save, bbox_inches='tight')

	plt.show()

def evol_len_pop_show(arr_len_pop, arr_sum_pop, figsize=(10, 6), marge=0.01,
					  save=None):
	"""
	Represents changes in population size over generations.

	Parameters
	----------
	arr_len_pop: array
		Nombre d'individus par génération 
	arr_sum_pop: array
		Somme commulé du nombre d'individus par génération
	figsize: tuple
		Largeur et hauteur de la figure.
	save : NoneType or str, optional
		If not None, it have to be str path to save the picture.

	Returns
	-------
	None.

	"""
	ampli = len(arr_len_pop)
	# x-axis limits
	lower = -ampli*marge
	upper = len(arr_len_pop)-1+ampli*marge
	
	plt.figure(figsize=figsize)
	plt.title("Evolution of the population", fontsize=22)
	plt.grid(True, zorder=1)
	plt.plot(arr_len_pop, '.-', label="By generation", zorder=2)
	plt.legend(fontsize=15, loc='upper left')
	plt.xlabel("Generation", fontsize=15)
	plt.ylabel("Population", fontsize=15)
	plt.xticks(range(len(arr_len_pop)), range(len(arr_len_pop)),
			   fontsize=14, rotation=90)

	plt.yticks(fontsize=14)
	plt.xlim(lower, upper)
	plt.show()

	plt.figure(figsize=figsize)
	plt.title("Cumulative evolution of the population", fontsize=22)
	plt.grid(True, zorder=1)
	plt.plot(arr_sum_pop, '.-', label="Cumulated", zorder=2)
	plt.legend(fontsize=15, loc='upper left')
	plt.xlabel("Generation", fontsize=15)
	plt.ylabel("Cumulaive population", fontsize=15)
	plt.xticks(range(len(arr_len_pop)), range(len(arr_len_pop)),
			   fontsize=14, rotation=90)

	plt.yticks(fontsize=14)
	plt.xlim(lower, upper)
	if type(save) == str:
		if save[-4:] not in ['.png', '.pdf', '.jpg', '.svg', '.tif']:
			save += '.png'

		plt.savefig(save, bbox_inches='tight')

	plt.show()

def genealogic_tree(arr_of_pop, figsize=(16, 10), size_title=20, marge=0.02,
					save=None):
	"""
	Represents the family tree from the LUCA ancestor (identifier -1) of all
	individuals generated during the simulation. Gender and descent/ancestry
	links are indicated.

	Parameters
	----------
	arr_of_pop : numpy.ndarray
		A 2 dimensions array. It is the array of all of the peoples generated
		during simulation. Corresponds to the name variable 'tree' rendered
		by the function Evoluteur().
	figsize : tuple, optional
		Size of the fisure. The default is (16, 10).
	size_title : int, optional
		Size of the title. The default is 20.
	save : NoneType or str, optional
		If not None, it have to be str path to save the picture.

	Returns
	-------
	None

	"""
	# to select (self, mother, father, generation) number
	ncaract = np.array([0, 2, 3, 4])
	# selected (self, mother, father, generation) number
	arr_gene = (arr_of_pop[:, ncaract]).astype(int).T
	# y-axis position of LUCA. All of the generations will be around this
	# value.
	cent = 2
	tree = np.array([arr_gene[0], arr_gene[3], arr_gene[1], arr_gene[2]]).T
	tree_cut, pop_g = [], []
	for i in range(tree[-1, 1]+1):
		tree_cut.append(tree[tree[:, 1] == i])
		pop_g.append(len(tree_cut[i]))

	tree_cut = np.array(tree_cut, dtype=object)
	leng = np.max(pop_g)
	ampli = len(tree_cut)-1
	# x-axis limits
	lower = -1-ampli*marge
	upper = len(tree_cut)-1+ampli*marge

	#Centered on cent, between 0.5 and 1.5
	plt.figure(figsize=figsize)
	plt.title("Genealogic tree", fontsize=size_title)
	plt.plot(-1, cent, 'k.', label="Peoples", markersize=12, zorder=2)
	plt.plot(-1, cent, 'k*', label="'LUCA'", markersize=14, zorder=3)
	for i in range(len(tree_cut)):
		brc1 = np.copy(tree_cut[i])
		y1 = (brc1[:, 0]-brc1[0, 0])/leng+cent
		y1 = y1-np.max(y1)/2+(cent/2)
		x1 = np.array([i]*pop_g[i])
		plt.plot(x1, y1, 'k.', markersize=8, zorder=3)
		# dico relation number
		if i == 0:
			link_x1 = np.zeros(pop_g[i])-1
			link_x2 = np.zeros(pop_g[i])
			link_y1 = np.zeros(pop_g[i])+cent
			link_y2 = y1.astype(float)
			for j in range(pop_g[i]):
				plt.plot([link_x1[j], link_x2[j]], [link_y1[j], link_y2[j]],
						 '-', color=[1, 0, 1], zorder=2)

		else:
			brc2 = np.copy(tree_cut[i-1])#preious generation
			min_r2 = np.min(brc2[:, 0])#min self n° previous
			min_r1 = np.min(brc1[:, 0])#min self n° current
			y2 = (brc2[:, 0]-brc2[0, 0])/leng+cent
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
	plt.xticks(range(-1, tree[-1, 1]+1), range(-1, tree[-1, 1]+1),
			   fontsize=14)

	plt.yticks(fontsize=0)
	plt.xlim(lower, upper)
	if type(save) == str:
		if save[-4:] not in ['.png', '.pdf', '.jpg', '.svg', '.tif']:
			save += '.png'

		plt.savefig(save, bbox_inches='tight')

	plt.show()

def informativ_linear_tree(arr_of_pop, figsize=(18, 14), size_title=22,
						   scat_size=30, cmap='jet', marge=0.02, save=None):
	"""
	This fuction used to show the genealogical link between the peoples
	created during the simulation, and their mutation rate (=number of mutated
	gens/total number of gens).

	Parameters
	----------
	arr_of_pop : numpy.ndarray
		A 2 dimensions array. It is the array of all of the peoples generated
		during simulation. Corresponds to the name variable 'tree' rendered
		by the function Evoluteur().
	figsize : tuple, optional
		Size of the fisure. The default is (16, 10).
	size_title : int, optional
		Size of the title. The default is 20.
	scat_size : int, optional
		Size of the dots. The default is 30.
	cmap : str, optional
		Name of the color map used for the mutation rate. The default is
		'jet'. I try different color map, and the most readeball were:
		['autumn', 'brg', 'plasma', 'gist_rainbow', 'gnuplot', 'jet',
		 'turbo', 'viridis'].
	save : NoneType or str, optional
		If not None, it have to be str path to save the picture.

	Returns
	-------
	None.

	"""

	perc_mut = np.zeros(arr_of_pop[:, 6:].shape)
	perc_mut[arr_of_pop[:, 6:] == 'aa'] = 2
	perc_mut[arr_of_pop[:, 6:] == 'Aa'] = 1
	perc_mut[arr_of_pop[:, 6:] == 'aA'] = 1
	perc_mut = np.sum(perc_mut, axis=1)/(len(arr_of_pop[0, 6:])*2)
	vals, pop_g = np.unique(arr_of_pop[:, 4].astype(int), return_counts=True)
	ncaract = np.array([0, 2, 3, 4])
	arr_gene = (arr_of_pop[:, ncaract]).astype(int).T
	tree = np.array([arr_gene[0], arr_gene[3], arr_gene[1], arr_gene[2]]).T
	tree_cut = []
	gender = []
	muta_cut = []
	for i in range(tree[-1, 1]+1):
		tree_cut.append(tree[tree[:, 1] == i])
		gender.append(arr_of_pop[tree[:, 1] == i , 1])
		muta_cut.append(perc_mut[tree[:, 1] == i])

	tree_cut = np.array(tree_cut, dtype=object)
	muta_cut = np.array(muta_cut, dtype=object)
	leng = np.max(pop_g)
	nbg = np.max(vals)
	# y-axis position of LUCA. All of the generations will be around this
	# value.
	cent = 2
	ampli = len(tree_cut)-1
	# x-axis limits
	lower = -1-ampli*marge
	upper = len(tree_cut)-1+ampli*marge

	plt.figure(figsize=figsize)
	plt.title("Genealogical tree & mutation rate", fontsize=size_title)
	plt.plot(-1, cent, 'k*', markersize=15, zorder=3, label='LUCA')
	plt.plot(-4, cent, 'ko', markersize=10, zorder=2, label='Woman')
	plt.plot(-4, cent, 'k^', markersize=10, zorder=1, label='Man')
	for i in range(len(tree_cut)):
		brc1 = np.copy(tree_cut[i])
		y1 = (brc1[:, 0]-brc1[0, 0])/leng+cent
		y1 = y1-np.max(y1)/2+(cent/2)
		x1 = np.array([i]*pop_g[i])
		plt.scatter(x1[gender[i] == 'f'], y1[gender[i] == 'f'], cmap=cmap,
					s=scat_size, c=muta_cut[i][gender[i] == 'f'], marker='o',
					zorder=3, vmin=0, vmax=1, edgecolors='k')

		plt.scatter(x1[gender[i] == 'm'], y1[gender[i] == 'm'], cmap=cmap,
					s=scat_size, c=muta_cut[i][gender[i] == 'm'], marker='^',
					zorder=3, vmin=0, vmax=1, edgecolors='k')

		if i == 0:
			link_x1 = np.zeros(pop_g[i])-1
			link_x2 = np.zeros(pop_g[i])
			link_y1 = np.zeros(pop_g[i])+cent
			link_y2 = y1.astype(float)
			for j in range(pop_g[i]):
				plt.plot([link_x1[j], link_x2[j]], [link_y1[j], link_y2[j]],
						 '-', color=[1, 0, 1], zorder=1)

		else:
			brc2 = np.copy(tree_cut[i-1])
			min_r2 = np.min(brc2[:, 0])
			min_r1 = np.min(brc1[:, 0])
			y2 = (brc2[:, 0]-brc2[0, 0])/leng+cent
			y2 = y2-np.max(y2)/2+(cent/2)
			x2 = i-1
			for j in range(len(brc1)):
				ym = y2[int(brc1[j, 2]-min_r2)]
				yp = y2[int(brc1[j, 3]-min_r2)]
				yi = y1[int(brc1[j, 0]-min_r1)]
				plt.plot([x2, x1[0]], [ym, yi], 'b-', zorder=1)
				plt.plot([x2, x1[0]], [yp, yi], 'r-', zorder=1)

	plt.plot(-1, cent, 'b-',
			 label="From left to right :\nMother -> child(s)")

	plt.plot(-1, cent, 'r-',
			 label="Father  -> child(s)")

	plt.xlabel('Generation\n--------------------\nMutation Rate', fontsize=15)
	plt.xticks(range(-1, nbg+1), range(-1, nbg+1), fontsize=14)
	plt.yticks(fontsize=0)
	plt.legend(fontsize=15, loc='upper left')
	plt.xlim(lower, upper)
	plt.colorbar(shrink=0.7, orientation="horizontal", pad=0.12, aspect=50)
	if type(save) == str:
		if save[-4:] not in ['.png', '.pdf', '.jpg', '.svg', '.tif']:
			save += '.png'

		plt.savefig(save, bbox_inches='tight')

	plt.show()

def consang_linear_tree(arr_of_pop, figsize=(18, 14), size_title=22,
						scat_size=30, cmap='jet', marge=0.02, save=None):
	"""
	This fuction used to show the genealogical link between the peoples
	created during the simulation, and their mutation rate (=number of mutated
	gens/total number of gens).

	Parameters
	----------
	arr_of_pop : numpy.ndarray
		A 2 dimensions array. It is the array of all of the peoples generated
		during simulation. Corresponds to the name variable 'tree' rendered
		by the function Evoluteur().
	figsize : tuple, optional
		Size of the fisure. The default is (16, 10).
	size_title : int, optional
		Size of the title. The default is 20.
	scat_size : int, optional
		Size of the dots. The default is 30.
	cmap : str, optional
		Name of the color map used for the mutation rate. The default is
		'jet'. I try different color map, and the most readeball were:
		['autumn', 'brg', 'plasma', 'gist_rainbow', 'gnuplot', 'jet',
		 'turbo', 'viridis'].
	save : NoneType or str, optional
		If not None, it have to be str path to save the picture.

	Returns
	-------
	None.

	"""
	vals, pop_g = np.unique(arr_of_pop[:, 4].astype(int), return_counts=True)
	ncaract = np.array([0, 2, 3, 4])
	arr_gene = (arr_of_pop[:, ncaract]).astype(int).T
	tree = np.array([arr_gene[0], arr_gene[3], arr_gene[1], arr_gene[2]]).T
	tree_cut = []
	gender = []
	muta_cut = []
	for i in range(tree[-1, 1]+1):
		tree_cut.append(tree[tree[:, 1] == i])
		gender.append(arr_of_pop[tree[:, 1] == i , 1])
		muta_cut.append(arr_of_pop[tree[:, 1] == i, 5].astype(float))

	tree_cut = np.array(tree_cut, dtype=object)
	muta_cut = np.array(muta_cut, dtype=object)
	leng = np.max(pop_g)
	nbg = np.max(vals)
	# y-axis position of LUCA. All of the generations will be around this
	# value.
	cent = 2
	ampli = len(tree_cut)-1
	# x-axis limits
	lower = -1-ampli*marge
	upper = len(tree_cut)-1+ampli*marge

	plt.figure(figsize=figsize)
	plt.title("Genealogical tree & consanguinity", fontsize=size_title)
	plt.plot(-1, cent, 'k*', markersize=15, zorder=3, label='LUCA')
	plt.plot(-4, cent, 'ko', markersize=10, zorder=2, label='Woman')
	plt.plot(-4, cent, 'k^', markersize=10, zorder=1, label='Man')
	for i in range(len(tree_cut)):
		brc1 = np.copy(tree_cut[i])
		y1 = (brc1[:, 0]-brc1[0, 0])/leng+cent
		y1 = y1-np.max(y1)/2+(cent/2)
		x1 = np.array([i]*pop_g[i])
		plt.scatter(x1[gender[i] == 'f'], y1[gender[i] == 'f'], cmap=cmap,
					s=scat_size, c=muta_cut[i][gender[i] == 'f'], marker='o',
					zorder=3, vmin=0, vmax=1, edgecolors='k')

		plt.scatter(x1[gender[i] == 'm'], y1[gender[i] == 'm'], cmap=cmap,
					s=scat_size, c=muta_cut[i][gender[i] == 'm'], marker='^',
					zorder=3, vmin=0, vmax=1, edgecolors='k')

		if i == 0:
			link_x1 = np.zeros(pop_g[i])-1
			link_x2 = np.zeros(pop_g[i])
			link_y1 = np.zeros(pop_g[i])+cent
			link_y2 = y1.astype(float)
			for j in range(pop_g[i]):
				plt.plot([link_x1[j], link_x2[j]], [link_y1[j], link_y2[j]],
						 '-', color=[1, 0, 1], zorder=1)

		else:
			brc2 = np.copy(tree_cut[i-1])
			min_r2 = np.min(brc2[:, 0])
			min_r1 = np.min(brc1[:, 0])
			y2 = (brc2[:, 0]-brc2[0, 0])/leng+cent
			y2 = y2-np.max(y2)/2+(cent/2)
			x2 = i-1
			for j in range(len(brc1)):
				ym = y2[int(brc1[j, 2]-min_r2)]
				yp = y2[int(brc1[j, 3]-min_r2)]
				yi = y1[int(brc1[j, 0]-min_r1)]
				plt.plot([x2, x1[0]], [ym, yi], 'b-', zorder=1)
				plt.plot([x2, x1[0]], [yp, yi], 'r-', zorder=1)

	plt.plot(-1, cent, 'b-',
			 label="From left to right :\nMother -> child(s)")

	plt.plot(-1, cent, 'r-',
			 label="Father  -> child(s)")

	plt.xlabel('Generation\n--------------------\nConsanguinity', fontsize=15)
	plt.xticks(range(-1, nbg+1), range(-1, nbg+1), fontsize=14)
	plt.yticks(fontsize=0)
	plt.legend(fontsize=15, loc='upper left')
	plt.xlim(lower, upper)
	plt.colorbar(shrink=0.7, orientation="horizontal", pad=0.12, aspect=50)
	if type(save) == str:
		if save[-4:] not in ['.png', '.pdf', '.jpg', '.svg', '.tif']:
			save += '.png'

		plt.savefig(save, bbox_inches='tight')

	plt.show()
