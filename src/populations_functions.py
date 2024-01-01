# -*- coding: utf-8 -*-
"""
Module with functions to model the endogamic depression.
"""
import numpy as np
#=============================================================================
def sexe_individus(p_woman, size=1):
	"""
	Sex of the people in function of the probability to drawn a woman.

	Parameters
	----------
	p_woman : float
		Probability to drawn a woman. I must be in the range of 0 to 1.

	Returns
	-------
	str
		The sex of thr people : 'f' or 'm'. Closer this number will be to 1,
		higger the the probility to draw a woman will be.

	Exemple
	-------
	In[0] : sexe_individus(0.9)
	Out[0] : 'f'

	"""
	if size == 1:
		draw = np.random.random(1)
		if draw < p_woman:
			return 'f'
	
		else:
			return 'm'

	elif size > 1:
		draw = np.random.random(size)
		sexes = np.zeros(size, dtype='<U9')
		sexes[:] = 'f'
		sexes[draw >= p_woman] = 'm'
		return sexes

def individus_init(size, pwoman, ngens, exactprop=False, pre_mut=0,
				   prop_mut=0):
	"""
	Create the initial population with size. With proportion of women is
	pwoman.

	Parameters
	----------
	size : int
		Number of people in the initial population. If the number is to low,
		there wil be risk that the pouplation does not survive.
	pwoman : float
		Probability to drawn a woman. It must be in the range of 0 to 1.
	ngens : int
		Number of gene pairs.
	exactprop : bool, optional
		If not False, the initial population will have the best proportion
		between the number of men and women following the 'pWonen' given. The
		default is False.

	Returns
	-------
	pop_gen_0 : numpy.ndarray
		A 2 dimensions array, it countain the initial population. For each
		people, their caracteristics are as it follow : {self identifiant,
		sex, mother identifiant, father identifiant, self generation,
		consanguinity of the parents, gens}.

	Exemple
	-------
	In[0] : individus_init(10, 0.6, 2, exactprop=False)
	Out[0] : np.array([['0', 'm', '-1', '-1', '0', '0', 'AA', 'AA'],
				       ['1', 'm', '-1', '-1', '0', '0', 'AA', 'AA'],
				       ['2', 'f', '-1', '-1', '0', '0', 'AA', 'AA'],
				       ['3', 'f', '-1', '-1', '0', '0', 'AA', 'AA'],
				       ['4', 'f', '-1', '-1', '0', '0', 'AA', 'AA'],
				       ['5', 'f', '-1', '-1', '0', '0', 'AA', 'AA'],
				       ['6', 'f', '-1', '-1', '0', '0', 'AA', 'AA'],
				       ['7', 'm', '-1', '-1', '0', '0', 'AA', 'AA'],
				       ['8', 'f', '-1', '-1', '0', '0', 'AA', 'AA'],
				       ['9', 'f', '-1', '-1', '0', '0', 'AA', 'AA']],
				      dtype='<U9')

	Note
	----
	The initialization of the first generation with 'pop_gen_0' use a
	numpy.ndarray with dtype = '<U9'. Consequently, it cannot countain more
	than 1e9 persons. If you try to it will cutt the input value at the 9-th
	position. This will lead to major issue when computing the evolution. If
	you want to have more persons, you can modify this parameter by increasing
	the value '9' from the dtype.

	"""
	if size >= 1e9:
		print('Warning message: you asked for a population with size='
			  + str(size)+' . Be aware taht because of how the initialization'
			  + " of the first generation with 'pop_gen_0' is done, it cannot"
			  + ' countain more than 1e9 persons. If you try to it will cutt '
			  + 'the input value at the 9-th position. This will lead to '
			  + 'major issue when computing the evolution. If you want to '
			  + 'have more persons, you can modify this parameter by '
			  + "increasing the value '9' from the dtype.")

	pop_gen_0 = np.full([size, int(6+ngens)], '', dtype='<U9')
	pop_gen_0[:, 0] = range(size) # id of the people
	if exactprop:
		num_w = int(pwoman*size)
		pop_gen_0[:num_w, 1] = 'f'
		pop_gen_0[num_w:, 1] = 'm'
	else:
		pop_gen_0[:, 1] = sexe_individus(pwoman, size)

		# to avoid having no woman or no man in the first generation
		if len(pop_gen_0[pop_gen_0[:, 1] == 'f']) == 0:
			pop_gen_0[np.random.randint(0, size), 1] = 'f'

		if len(pop_gen_0[pop_gen_0[:, 1] == 'm']) == 0:
			pop_gen_0[np.random.randint(0, size), 1] = 'm'

	pop_gen_0[:, 6:] = "AA"
	if pre_mut > 0:
		quanti = int(size*pre_mut)
		mutated = np.arange(size)
		mutated = np.random.choice(mutated, quanti, False)
		proportion = np.random.uniform(0, prop_mut, quanti)
		proportion = np.round(proportion*ngens*2, 0).astype(int)
		mask = np.arange(2*ngens)*np.ones(quanti)[:, np.newaxis]
		new_gens = np.full((quanti, 2*ngens), 'A')
		new_gens[mask <= proportion[:, np.newaxis]] = 'a'
		new_gens = np.random.permutation(new_gens.T).astype(object)
		new_gens = (new_gens.T[:, ::2]+new_gens.T[:, 1::2]).astype('<U9')
		pop_gen_0[mutated, 6:] = new_gens

	pop_gen_0[:, 2] = -1
	pop_gen_0[:, 3] = -1
	pop_gen_0[:, 4] = 0
	pop_gen_0[:, 5] = 0.0
	return pop_gen_0

def matcher_couple(arr_of_pop, ppolygam=0):
	"""
	Function that creat the couple that will give (or not) childs.

	Parameters
	----------
	arr_of_pop : numpy.ndarray
		2 dimensions array, containing the people of the current generation.
	ppolygam : float, optional
		Probability of infedility. If equal to 0 then all couples will be
		strictly monogamous. If equal to 1 then all couples will be fully
		polygamous. Note tha the the rate of polygamous is the same for the
		two sex.

	Returns
	-------
	couple_array : numpy.ndarray
		3 dimensions array, containing the couples made by the people of the
		current generation.

	Exemple
	-------
	In[0] : pop = individus_init(10, 0.6, 2, exactprop=False)
	In[1] : matcher_couple(pop, 0.2)
	Out[1]: np.array([[['4', 'f', '-1', '-1', '0', '0.0', 'AA', 'AA'],
					   ['3', 'm', '-1', '-1', '0', '0.0', 'AA', 'AA']],
					  [['0', 'f', '-1', '-1', '0', '0.0', 'AA', 'AA'],
					   ['2', 'm', '-1', '-1', '0', '0.0', 'AA', 'AA']],
					  [['7', 'f', '-1', '-1', '0', '0.0', 'AA', 'AA'],
					   ['1', 'm', '-1', '-1', '0', '0.0', 'AA', 'AA']],
					  [['8', 'f', '-1', '-1', '0', '0.0', 'AA', 'AA'],
					   ['9', 'm', '-1', '-1', '0', '0.0', 'AA', 'AA']],
					  [['6', 'f', '-1', '-1', '0', '0.0', 'AA', 'AA'],
					   ['5', 'm', '-1', '-1', '0', '0.0', 'AA', 'AA']],
					  [['6', 'f', '-1', '-1', '0', '0.0', 'AA', 'AA'],
					   ['3', 'm', '-1', '-1', '0', '0.0', 'AA', 'AA']],
					  [['4', 'f', '-1', '-1', '0', '0.0', 'AA', 'AA'],
					   ['2', 'm', '-1', '-1', '0', '0.0', 'AA', 'AA']]],
					dtype='<U9')

	"""
	women = arr_of_pop[arr_of_pop[:, 1] == 'f']
	men = arr_of_pop[arr_of_pop[:, 1] == 'm']
	couple = []
	if ppolygam < 1:
		fliste, mliste = list(women), list(men)
		stop = False
		while stop != True:
			if (len(fliste) > 0)&(len(mliste) > 0):
				nbf = np.random.randint(0, len(fliste))
				nbm = np.random.randint(0, len(mliste))
				couple.append([fliste[nbf], mliste[nbm]])
				fliste.pop(nbf)
				mliste.pop(nbm)
			else:
				stop = True

			if (len(fliste) == 0)|(len(mliste) == 0):
				stop = True

		if (ppolygam > 0)&(len(arr_of_pop) > 2):
			ninfi = int(ppolygam*len(arr_of_pop))
			c = 0
			idw = women[:, 0]
			idm = men[:, 0]
			stop = False
			alread = np.array(couple)[:, :, 0].tolist()
			while stop != True:
				infiw = np.random.randint(0, len(idw))
				infim = np.random.randint(0, len(idm))
				if [str(idw[infiw]), str(idm[infim])] not in alread:
					couple.append([women[infiw], men[infim]])
					alread = np.array(couple)[:, :, 0].tolist()
					c += 1

				if c >= ninfi:
					stop = True

	elif ppolygam >= 1:
		for i in range(len(women)):
			for j in range(len(men)):
				couple.append([women[i], men[j]])

	couple_array = np.array(couple)
	return couple_array

def origines(mother, father, arr_of_all_pop, maxdeep=10):
	"""
	Reconstructs the parent tree of the target individual.

	Parameters
	----------
	mother : numpy.ndarray
		A 1 dimensions array, it is the mother's informations, whose origins
		we want to reconstruct.
	father : numpy.ndarray
		A 1 dimensions array, it is the father's informations, whose origins
		we want to reconstruct.
	maxdeep : int, optional
		Made the function stop at the maxdeep-th generation.
	arr_of_all_pop : numpy.ndarray
		A 2 dimensions array array of individuals generated during simulation.
		Corresponds to the array of all the people ever created, represented
		by np.concatenate(all_pop) in the function Evoluteur.

	Returns
	-------
	parents : list
		List of the identity of the targeted individual, and parents of
		parents of parents...ect. It size is : (Number of generation,
		2*U[n-1]) ; U0 = 1.
		Structure: [[mother, father],
					[[mother, father],
					 [mother, father]],
					[[mother, father, mother, father],
					 [mother, father, mother, father]],
					...
					[[mother, father, ..., mother, father],
					 [mother, father, ..., mother, father]]]

	Exemple
	-------
	In[0] : pop = individus_init(10, 0.6, 2, exactprop=False)
	In[1] : couple = matcher_couple(pop, 0.2)
	In[2] : origines(couple[0, 0], couple[0, 1], pop, maxdeep=10)
	Out[2]: [np.array([[[4]], [[3]]])]

	"""
	parents = []
	g = int(mother[4])
	if g > maxdeep:
		g = maxdeep

	# (self, mom, dad) number
	related = np.copy(arr_of_all_pop)[:, np.array([0, 2, 3])].astype(int)
	parents.append(np.array([[int(mother[0])], [int(father[0])]]))
	parents.append(np.array([[mother[2], mother[3]],
							[father[2], father[3]]], dtype=int))

	if g > 0:
		for i in range(1, g):
			cmfrm = related[parents[i], 1:].reshape((2,
										   int(2*len(parents[i][0]))))

			parents.append(cmfrm)

		return parents

	else:
		return [np.array([[parents[0][0]], [parents[0][1]]])]

def kinship(mother, father, arr_of_all_pop, maxdeep=10):
	"""
	Function that compute the genetic kinship between the mother and the
	father.

	Parameters
	----------
	mother : numpy.ndarray
		A 1 dimensions array, it is the mother's informations.
	father : numpy.ndarray
		A 1 dimensions array, it is the father's informations.
	arr_of_all_pop : numpy.ndarray
		A 2 dimensions array of all peoples generated during the simulation.
		It corresponds to 'np.concatenate(all_pop)' into the function
		Evoluteur.
	maxdeep : int, optional
		Made the origines function stop at the maxdeep-th generation.

	Returns
	-------
	comm : float
		It is an array that show how much the mother and father are geneticaly
		close to each other.

	Exemple
	-------
	In[0] : pop = individus_init(10, 0.6, 2, exactprop=False)
	In[1] : couple = matcher_couple(pop, 0.2)
	In[2] : kinship(couple[0, 0], couple[0, 1], pop, maxdeep=10)
	Out[2]: 0.0

	"""
	parents = origines(mother, father, arr_of_all_pop, maxdeep)
	len_mom = len(parents)
	comm = []
	for i in range(len_mom):
		max_n = np.max([parents[i][0], parents[i][1]])
		c1 = np.zeros(max_n+1)
		c2 = np.zeros(max_n+1)
		vm, cm = np.unique(parents[i][0], return_counts=True)
		vf, cf = np.unique(parents[i][1], return_counts=True)
		c1[vm] = cm
		c2[vf] = cf
		corr = c1[(c1 > 0)&(c2 > 0)]+c2[(c1 > 0)&(c2 > 0)]
		comm.append(corr.sum()/2/len(parents[i][0]))

	if len(comm) <= 2:
		comm = np.mean(np.array(comm, dtype=float))
	else:
		comm = np.median(np.array(comm, dtype=float)[1:])

	return comm

def gene_transmis(mother, father, p_mutation, sc_consang):
	"""
	Function managing the transmission of genes from parents to their child.

	Parameters
	----------
	mother : numpy.ndarray
		A 1 dimensions array, it is the mother's informations.
	father : numpy.ndarray
		A 1 dimensions array, it is the father's informations.
	p_mutation : float
		The base mutate probabimity.
	sc_consang : numpy.ndarray
		A 1 dimensions array the consanguinity score.

	Returns
	-------
	gens : numpy.ndarray
		A 1 dimensions array, it is the gens of the child.

	Exemple
	-------
	In[0] : pop = individus_init(10, 0.6, 2, exactprop=False)
	In[1] : couple = matcher_couple(pop, 0.2)
	In[2] : sc_consang = kinship(couple[0, 0], couple[0, 1], pop, maxdeep=10)
	In[3] : gene_transmis(couple[0, 0], couple[0, 1], 0.0001, sc_consang)
	Out[2]: np.array(['AA', 'AA'], dtype=object)

	"""
	p_mut = p_mutation+sc_consang
	mother_gens = np.sum(mother[6:].astype('O'))
	father_gens = np.sum(father[6:].astype('O'))
	ln = len(mother_gens)
	mother_gens = np.array(list(mother_gens))
	father_gens = np.array(list(father_gens))
	mother_gens = mother_gens.reshape((2, int(ln/2)))
	father_gens = father_gens.reshape((2, int(ln/2)))
	k1, k2 = np.random.randint(0, 2, (2, int(ln/2)))
	par1 = mother_gens[k1, np.arange(ln//2)]
	par2 = father_gens[k2, np.arange(ln//2)]
	mut1, mut2 = np.random.rand(2, ln//2) < p_mut
	par1[mut1] = 'a'
	par2[mut2] = 'a' 
	gens = np.sum(np.array([par1, par2], dtype='O').T, axis=1)
	return gens

def descendant(arr_of_couple, p_woman, p_mutation, id_pop_bef,
			   arr_of_all_pop, dp=None, coresvals=None, inf_consg=1,
			   maxdeep=10):
	"""
	Generate the childs created by the couples.

	Parameters
	----------
	arr_of_couple : numpy.ndarray
		Array that countain the couple formed by the matcher_couple function.
	p_woman : float
		Probability to drawn a woman. I must be in the range of 0 to 1..
	p_mutation : float
		Mutation probability (whitout taking in count the parenty between
		people).
	id_pop_bef : int
		Number of individuals that were generated from the beginning.
	arr_of_all_pop : numpy.ndarray
		A 2 dimensions array of all individuals generated during simulation.
	dp : numpy.ndarray, optional
		A 2 dimensions array. It is the probability density of having n
		children (n coming from coresvals). The default is None.
	coresvals : numpy.ndarray, optional
		A 1 dimension array. It is the boundaries between the different
		density probability function about the number of child. The default is
		None.
	inf_consg : float, optional
		To modulate the inluence of the consanguinity score. The default is 1.
	maxdeep : int, optional
		Made the origines function stop at the maxdeep-th generation.

	Raises
	------
	ValueError
		An error were found either in dp or coresvals.

	Returns
	-------
	childs_list : numpy.ndarray
		A 2 dimension array. This is the child array.

	Exemple
	-------
	In[0] : pop = individus_init(10, 0.6, 2, exactprop=False)
	In[1] : couple = matcher_couple(pop, 0.2)
	In[2] : descendant(couple, 0.6, 0.0001, 10,pop, dp=None, coresvals=None,
					   inf_consg=1, maxdeep=10)
	Out[2]: array([['10', 'm', '4', '3', '1', '0.0', 'AA', 'AA'],
				   ['11', 'm', '4', '3', '1', '0.0', 'AA', 'AA'],
				   ['12', 'f', '4', '3', '1', '0.0', 'AA', 'AA'],
				   ['13', 'm', '4', '3', '1', '0.0', 'AA', 'AA'],
				   ['14', 'm', '4', '3', '1', '0.0', 'AA', 'AA'],
				   ['15', 'f', '0', '2', '1', '0.0', 'AA', 'AA'],
				   ['16', 'm', '0', '2', '1', '0.0', 'AA', 'AA'],
				   ['17', 'f', '0', '2', '1', '0.0', 'AA', 'AA'],
				   ['18', 'f', '7', '1', '1', '0.0', 'AA', 'AA'],
				   ['19', 'm', '7', '1', '1', '0.0', 'AA', 'AA'],
				   ['20', 'f', '7', '1', '1', '0.0', 'AA', 'AA'],
				   ['21', 'm', '8', '9', '1', '0.0', 'AA', 'AA'],
				   ['22', 'f', '8', '9', '1', '0.0', 'AA', 'AA'],
				   ['23', 'f', '8', '9', '1', '0.0', 'AA', 'AA'],
				   ['24', 'm', '8', '9', '1', '0.0', 'AA', 'AA'],
				   ['25', 'f', '6', '5', '1', '0.0', 'AA', 'AA'],
				   ['26', 'm', '6', '5', '1', '0.0', 'AA', 'AA'],
				   ['27', 'f', '6', '5', '1', '0.0', 'AA', 'AA'],
				   ['28', 'f', '6', '5', '1', '0.0', 'AA', 'AA'],
				   ['29', 'f', '6', '3', '1', '0.0', 'AA', 'AA'],
				   ['30', 'f', '6', '3', '1', '0.0', 'AA', 'AA'],
				   ['31', 'f', '6', '3', '1', '0.0', 'AA', 'AA'],
				   ['32', 'm', '6', '3', '1', '0.0', 'AA', 'AA'],
				   ['33', 'f', '6', '3', '1', '0.0', 'AA', 'AA'],
				   ['34', 'f', '6', '3', '1', '0.0', 'AA', 'AA'],
				   ['35', 'm', '4', '2', '1', '0.0', 'AA', 'AA'],
				   ['36', 'f', '4', '2', '1', '0.0', 'AA', 'AA'],
				   ['37', 'm', '4', '2', '1', '0.0', 'AA', 'AA']],
			   dtype='<U9')

	"""
	leng = len(arr_of_couple)
	shp = arr_of_all_pop.shape
	# First: compute the number of child(s) per couple(s) probabilites
	if type(coresvals) != type(None):
		if leng < coresvals[-1]:
			udp = dp[np.where(coresvals > leng)[0][0]-1]
		else:
			udp = dp[-1]

	else:
		if leng <= 10:
			udp = np.array([0.  , 0.01, 0.08, 0.31, 0.28, 0.17, 0.15])
		elif leng <= 30:
			udp = np.array([0.01, 0.05, 0.18, 0.28, 0.2 , 0.16, 0.12])
		else:
			udp = np.array([0.01, 0.07, 0.45, 0.35, 0.08, 0.03, 0.01])

	n_chs = np.random.choice(np.arange(len(udp)), leng, True, p=udp)
	cumul = np.append(0, np.cumsum(n_chs))
	childs_list = np.zeros((cumul[-1], shp[1]), dtype = '<U9')
	# id number
	childs_list[:, 0] = id_pop_bef + np.arange(cumul[-1])
	# sex
	childs_list[:, 1] = sexe_individus(p_woman, cumul[-1])
	# generation number
	childs_list[:, 4] = str(int(arr_of_couple[0, 0, 4])+1)
	for i in range(1, leng+1):
		# mother number
		childs_list[cumul[i-1]:cumul[i] , 2] = arr_of_couple[i-1, 0, 0]
		# father number
		childs_list[cumul[i-1]:cumul[i] , 3] = arr_of_couple[i-1, 1, 0]
		sc_consg = kinship(arr_of_couple[i-1, 0], arr_of_couple[i-1, 1],
						   arr_of_all_pop, maxdeep)

		# consanguinity factor
		childs_list[cumul[i-1]:cumul[i], 5] = str(sc_consg)[:9]
		for n in range(cumul[i-1], cumul[i]):
			# genes
			childs_list[n, 6:] = gene_transmis(arr_of_couple[i-1, 0],
											   arr_of_couple[i-1, 1],
											   p_mutation, sc_consg*inf_consg)

	return childs_list

def count_muts(arr_of_pop, ngens):
	"""
	Function to count the number of people who have n mutations.

	Parameters
	----------
	arr_of_pop : numpy.ndarray
		A 2 dimensions array of the people of the generation.
	ngens : int
		Number of gene pairs.

	Returns
	-------
	mutgs : numpy.ndarray
		A 1 dimension array of the number of people who have n mutated gens.

	Exemple
	-------
	In[0] : pop = individus_init(10, 0.6, 2, exactprop=False)
	In[1] : count_muts(pop, 2)
	Out[2]: np.array([10.,  0.,  0.,  0.,  0.])

	"""
	mutgs = np.zeros(int(2*ngens+1))
	for j in range(len(arr_of_pop)):
		temp = np.array(list(np.sum(arr_of_pop[j, 6:].astype('O'))))
		mutgs[len(temp[temp == 'a'])] += 1

	return mutgs

def evoluteur(ngeneration, length_ini, p_woman, ngens, p_mutation, ppolygam=0,
			  perf_equi_ini=False, dp=None, coresvals=None, inf_consg=1,
			  maxdeep=10, pre_mut_ini=0, prop_mut_ini=0):
	"""
	Function that gather the other function to simulate the evolution of the
	initial population.

	Parameters
	----------
	ngeneration : int
		The number of generation you want to generate, must be >=1.
	length_ini : int
		Number of people in the initial population, avoid descending below
		10 peoples.
	p_woman : float
		Probability to drawn a woman. It must be in the range of 0 to 1.
	ngens : int
		Number of gene pairs.
	p_mutation : float
		Mutation probability (whitout taking in count the parenty between
		people).
	ppolygam : float, optional
		Probability of infedility. If equal to 0 then all couples will be
		strictly monogamous. If equal to 1 then all couples will be fully
		polygamous. Note tha the the rate of polygamous is the same for the
		two sex.
	perf_equi_ini : bool, optional
		If True, the initial population will have the best equilibrium between
		the number of men and women. The default is False.
	dp : numpy.ndarray, optional
		A 2 dimensions array. It is the probability density of having n
		children (n coming from coresvals). The default is None.
	coresvals : numpy.ndarray, optional
		A 1 dimension array. It is the boundaries between the different
		density probability function about the number of child. The default is
		None.
	inf_consg : float, optional
		To modulate the inluence of the consanguinity score. The default is 1.
	maxdeep : int, optional
		Made the origines function stop at the maxdeep-th generation.

	Returns
	-------
	muta_n : numpy.ndarray
		A 1 dimension array, number of idividus with n mutation.
	evo_len_pop : numpy.ndarray
		A 1 dimension array, Number of peoples by generation.
	evo_len_pop_cum : numpy.ndarray
		A 1 dimension array, Cumulative sum of the number of people by
		generation.
	all_pop : numpy.ndarray
		A 2 dimensions array, countain all the people generated during the
		simulation.

	Exemple
	-------
	In[0] : evoluteur(ngeneration=5, length_ini=2, p_woman=0.6, ngens=2,
					  p_mutation=1e-3, ppolygam=0.1, perf_equi_ini=True,
					  dp=None, coresvals=None, inf_consg=1e-3, maxdeep=10,
					  pre_mut_ini=0, prop_mut_ini=0)
	Out[0]: (np.array([[ 2., 0., 0., 0., 0.], [ 3., 0., 0., 0., 0.],
					   [ 8., 0., 0., 0., 0.], [19., 0., 0., 0., 0.],
					   [33., 0., 0., 0., 0.], [67., 3., 0., 0., 0.]]),
			 np.array([ 2.,  3.,  8., 19., 33., 70.]),
			 np.array([  2.,   5.,  13.,  32.,  65., 135.]),
			 np.array([['0', 'f', '-1', ..., '0.0', 'AA', 'AA'],
					   ['1', 'm', '-1', ..., '0.0', 'AA', 'AA'],
					   ['2', 'f', '0', ..., '0.0', 'AA', 'AA'],
					   ...,
					   ['132', 'm', '61', ..., '0.5', 'AA', 'AA'],
					   ['133', 'm', '61', ..., '0.5', 'AA', 'AA'],
					   ['134', 'f', '61', ..., '0.5', 'AA', 'AA']],
					 dtype='<U9')
			 )

	"""
	if type(coresvals) != type(None):
		#All this part is only to check if there are error in variable dp
		# and/or coresvals.
		if (type(coresvals) != list)&(type(coresvals) != np.ndarray):
			coresvals = np.ravel(coresvals)

		if len(coresvals) < 1:
			raise ValueError(
	"Corresponding values with the given probability density must be > 0.")

		if len(coresvals) != len(dp):
			raise ValueError("""There must be the same number of density
			probability and boundaries on corresponding population length.""")

		if coresvals.dtype != int:
			coresvals = np.array(coresvals, dtype=int)

		if type(dp) != type(None):
			if (type(dp) != list)&(type(dp) != np.ndarray):
				dp = np.array(dp)

			if (dp.dtype != float)&(dp.dtype != object):
				dp = np.array(dp, dtype=object)

			if len(np.shape(dp)) != 2:
				raise ValueError(
		"Probability density must be 2 dimension, not ", len(np.shape(dp)))

			for i in range(np.shape(dp)[0]):
				if len(dp[i]) == 0:
					raise ValueError("""Given probability density must have,
					   at least two values, indice:""", i)

				if np.round(np.sum(dp[i]), 8) != 1:
					raise ValueError("Probability sum must be equal to one:",
					   i)

	muta_n = []
	evo_len_pop = []
	evo_len_pop_cum = []
	current_generation = individus_init(length_ini, p_woman, ngens,
										perf_equi_ini, pre_mut_ini,
										prop_mut_ini)

	couples = matcher_couple(current_generation, ppolygam)
	all_pop = []
	all_pop.append(current_generation)
	print("Tere are ", length_ini, " peoples:", 
		  len(current_generation[current_generation[:, 1] == 'f']),
		  "women and",
		  len(current_generation[current_generation[:, 1] == 'm']), "men.")

	muta_n.append(count_muts(current_generation, ngens))
	evo_len_pop.append(len(current_generation))
	evo_len_pop_cum.append(np.cumsum(evo_len_pop))
	for i in range(ngeneration):
		current_generation = descendant(couples, p_woman, p_mutation,
										evo_len_pop_cum[i][-1],
										np.concatenate(all_pop), dp,
										coresvals, inf_consg, maxdeep)

		num_w = len(current_generation[current_generation[:, 1] == 'f'])
		num_m = len(current_generation[current_generation[:, 1] == 'm'])
		if (num_w == 0)&(num_m == 0):
			print('The population get extinct at the '+str(i)
				 +'-th generation. With '+str(num_w)+' woman(en) and '
				 +str(num_m)+' man(en).')

			break

		else:
			muta_n.append(count_muts(current_generation, ngens))
			all_pop.append(current_generation)
			evo_len_pop.append(len(current_generation))
			evo_len_pop_cum.append(np.cumsum(evo_len_pop))
			if  (num_w == 0)|(num_m == 0):
				print('The population get extinct at the '+str(i)
					 +'-th generation. With '+str(num_w)+' woman(en) and '
					 +str(num_m)+' man(en).')

				break

			else:
				couples = matcher_couple(current_generation, ppolygam)
				print("Generation", str(i+1)+"/"+str(ngeneration))
				print("\t", len(current_generation), 'individus')

	muta_n = np.array(muta_n)
	evo_len_pop = np.array(evo_len_pop, dtype=float)
	evo_len_pop_cum = np.cumsum(evo_len_pop)
	all_pop = np.concatenate(all_pop)
	return muta_n, evo_len_pop, evo_len_pop_cum, all_pop
