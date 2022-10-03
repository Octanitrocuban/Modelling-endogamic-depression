# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 15:42:06 2022

@author: Matthieu Nougaret
"""
import numpy as np
#=============================================================================
def SexeIndividus(pWoman):
	"""
	Sex of the people in function of the probability to drawn a woman.

	Parameters
	----------
	pWoman : float
		Probability to drawn a woman. I must be in the range of 0 to 1.

	Returns
	-------
	str
		The sex of thr people : 'f' or 'm'. Closer this number will be to 1,
		higger the the probility to draw a woman will be.

	Exemple
	-------
	In[0] : SexeIndividus(0.9)
	Out[0] : 'f'
	"""
	S = np.random.random(1)
	if S < pWoman: 
		return "f"
	else:
		return "m"

def IndividusIni(nIndividus, pWoman, Ngens, PerfectEquil=False):
	"""
	Create the initial population with nIndividus. With proportion of women is
	pWomen.

	Parameters
	----------
	nIndividus : int
		Number of people in the initial population. If the number is to low,
		there wil be risk that the pouplation does not survive.
	pWoman : float
		Probability to drawn a woman. It must be in the range of 0 to 1.
	Ngens : int
		Number of gene pairs.
	PerfectEquil : bool, optional
		If True, the initial population will have the best equilibrium between
		the number of men and women. The default is False.

	Returns
	-------
	Pop_gen_0 : numpy.ndarray
		DESCRIPTION.

	Exemple
	-------
	In[0] : IndividusIni(10, 0.6, 2, PerfectEquil=False)
	Out[0] : np.array([['0', 'm', '-1', '-1', 'AA', 'AA'],
				       ['1', 'm', '-1', '-1', 'AA', 'AA'],
				       ['2', 'f', '-1', '-1', 'AA', 'AA'],
				       ['3', 'f', '-1', '-1', 'AA', 'AA'],
				       ['4', 'f', '-1', '-1', 'AA', 'AA'],
				       ['5', 'f', '-1', '-1', 'AA', 'AA'],
				       ['6', 'f', '-1', '-1', 'AA', 'AA'],
				       ['7', 'm', '-1', '-1', 'AA', 'AA'],
				       ['8', 'f', '-1', '-1', 'AA', 'AA'],
				       ['9', 'f', '-1', '-1', 'AA', 'AA']],
				      dtype='<U5')
	"""
	Pop_gen_0 = np.full([nIndividus, int(4+Ngens)], '', dtype='<U9')
	if PerfectEquil == True:
		for i in range(nIndividus):
			Pop_gen_0[i, 0] = str(i)#identity of the people
			if i%2 == 0:
				Pop_gen_0[i, 1] = 'f'
			else:
				Pop_gen_0[i, 1] = 'm'
			Pop_gen_0[i, 2] = -1#id of the mother
			Pop_gen_0[i, 3] = -1#id of the father
			Pop_gen_0[i, 4:] = "AA"#2 healthy genes
	else:
		for i in range(nIndividus):
			Pop_gen_0[i, 0] = str(i)
			Pop_gen_0[i, 1] = SexeIndividus(pWoman)
			Pop_gen_0[i, 2] = -1
			Pop_gen_0[i, 3] = -1
			Pop_gen_0[i, 4:] = "AA"
	if len(Pop_gen_0[Pop_gen_0[:, 1] == 'f']) == 0:
		Pop_gen_0[np.random.randint(0, nIndividus), 1] = 'f'
	if len(Pop_gen_0[Pop_gen_0[:, 1] == 'm']) == 0:
		Pop_gen_0[np.random.randint(0, nIndividus), 1] = 'm'
	return Pop_gen_0

def MatcherCouple(ArrOfPop, ppolygam=0):
	"""
	Function that creat the couple that will give (or not) childs.

	Parameters
	----------
	ArrOfPop : numpy.ndarray
		2 dimensions array, containing the people of the current generation.
	ppolygam : float, optional
		Probability of infedility. If equal to 0 then all couples will be
		strictly monogamous. If equal to 1 then all couples will be fully
		polygamous. Note tha the the rate of polygamous is the same for the
		two sex.

	Returns
	-------
	CoupleArray : numpy.ndarray
		3 dimensions array, containing the couples made by the people of the
		current generation.

	"""
	Women = ArrOfPop[ArrOfPop[:, 1] == 'f']
	Men = ArrOfPop[ArrOfPop[:, 1] == 'm']
	Couple = []
	if ppolygam < 1:
		Fliste, Mliste = list(Women), list(Men) ; Stop = False
		while Stop != True:
			if (len(Fliste) > 0)&(len(Mliste) > 0):
				nbf = np.random.randint(0, len(Fliste))
				nbm = np.random.randint(0, len(Mliste))
				Couple.append([Fliste[nbf], Mliste[nbm]])
				Fliste.pop(nbf)
				Mliste.pop(nbm)
			else:
				Stop = True
			if (len(Fliste) == 0)|(len(Mliste) == 0):
				Stop = True
		if ppolygam > 0:
			Ninfi = int(ppolygam*len(ArrOfPop)) ; c = 0
			idw = Women[:, 0] ; idm = Men[:, 0] ; Stop = False
			alread = np.array(Couple)[:, :, 0].tolist()
			while Stop != True:
				infiw = np.random.randint(0, len(idw))
				infim = np.random.randint(0, len(idm))
				if [str(idw[infiw]), str(idm[infim])] not in alread:
					Couple.append([Women[infiw], Men[infim]])
					alread = np.array(Couple)[:, :, 0].tolist()
					c += 1
				if c >= Ninfi:
					Stop = True
	elif ppolygam >= 1:
		for i in range(len(Women)):
			for j in range(len(Men)):
				Couple.append([Women[i], Men[j]])
	CoupleArray = np.array(Couple)
	return CoupleArray

def Origines(Mother, Father, ArrOfAllPop):
	"""
	Reconstructs the parent tree of the target individual.

	Parameters
	----------
	Mother : numpy.ndarray
		A 1 dimensions array, it is the mother's informations, whose origins
		we want to reconstruct.
	Father : numpy.ndarray
		A 1 dimensions array, it is the father's informations, whose origins
		we want to reconstruct.

	ArrOfAllPop : numpy.ndarray
		A 2 dimensions array array of individuals generated during simulation.
		Corresponds to the array of all the people ever created, represented
		by np.concatenate(Evolution) in the function Evoluteur.

	Returns
	-------
	Parents : list
		List of the identity of the targeted individual, and parents of
		parents of parents...ect. It size is : (Number of generation,
	  2*U[n-1]) U0 = 1. Organization = [mother, father, ..., mother, father]

	"""
	Parent1, Parent2, c = [], [], 1
	related = np.copy(ArrOfAllPop)[:, np.array([0, 2, 3])]
	related = related.astype(int)# (self, mom, dad) number
	Parent1.append([int(Mother[0])])
	Parent2.append([int(Father[0])])
	Parent1.append(np.array([Mother[2], Mother[3]], dtype=int))
	Parent2.append(np.array([Father[2], Father[3]], dtype=int))
	if -1 not in Parent1[1]:
		Stop = False
		while Stop != True:
			comfrom1 = np.ravel(related[Parent1[c], 1:])
			comfrom2 = np.ravel(related[Parent2[c], 1:])
			Parent1[c] = Parent1[c].tolist()
			Parent2[c] = Parent2[c].tolist()
			if -1 in comfrom1:
				break
			Parent1.append(comfrom1)
			Parent2.append(comfrom2)
			c += 1
		return Parent1, Parent2
	else:
		return [Parent1[0]], [Parent2[0]]

def Kinship(Mother, Father, ArrOfAllPop):
	"""
	Function that compute the genetic kinship between the mother and the
	father.

	Parameters
	----------
	Mother : numpy.ndarray
		A 1 dimensions array, it is the mother's informations.
	Father : numpy.ndarray
		A 1 dimensions array, it is the father's informations.
	ArrOfAllPop : numpy.ndarray
		A 2 dimensions array of all peoples generated during the simulation.
		It corresponds to 'np.concatenate(Evolution)' into the function
		Evoluteur.

	Returns
	-------
	comm : float
		It is an array that show how much the mother and father are geneticaly
		close to each other.

	"""
	MotBr, FatBr = Origines(Mother, Father, ArrOfAllPop)
	if len(MotBr) != len(FatBr):
		print("MotBr :\n", MotBr, "\n")
		print("FatBr :\n", FatBr, "\n\n")
		raise
	comm = [] ; L = len(ArrOfAllPop)
	for i in range(len(MotBr)):
		val1, val2 = 0, 0 ; Len = len(MotBr[i])
		for j in range(Len):
			if MotBr[i][j] in FatBr[i]:
				val1 += 1/Len
			if FatBr[i][j] in MotBr[i]:
				val2 += 1/Len
		comm.append((val1+val2)/2)
	comm = np.array(comm, dtype=float)/((np.arange(len(comm))+1)*2).sum()
	return comm

def gene_transmis(Mother, Father, pMutation, ArrOfAllPop, InfConsg=1):
	"""
	Function managing the transmission of genes from parents to their child.

	Parameters
	----------
	Mother : numpy.ndarray
		A 1 dimensions array, it is the mother's informations.
	Father : numpy.ndarray
		A 1 dimensions array, it is the father's informations.
	pMutation : float
		The base mutate probabimity.
	ArrOfPop : numpy.ndarray
		A 2 dimensions array of all peoples generated during the simulation.
		It corresponds to 'np.concatenate(Evolution)' into the function
		Evoluteur.
	InfConsg : float, optional
		To modulate the inluence of the consanguinity score. The default is 1.

	Returns
	-------
	gens : numpy.ndarray
		A 1 dimensions array, it is the gens of the child.

	"""
	ScConsang = Kinship(Mother, Father, ArrOfAllPop)
	pMut = pMutation+np.sum(ScConsang)*InfConsg
	GM, GP = np.sum(Mother[4:].astype('O')), np.sum(Father[4:].astype('O'))
	ln = len(GM) ; GM = np.array(list(GM)) ; GP = np.array(list(GP))
	GM = GM.reshape((2, int(ln/2))) ; GP = GP.reshape((2, int(ln/2)))
	k1, k2 = np.random.randint(0, 2, (2, int(ln/2)))
	P1 = GM[k1, np.arange(ln//2)] ; P2 = GP[k2, np.arange(ln//2)]
	mut1, mut2 = np.random.rand(2, ln//2) < pMut
	P1[mut1] = 'a' ; P2[mut2] = 'a' 
	gens = np.sum(np.array([P1, P2], dtype='O').T, axis=1)
	return gens

def Descendant(ArrayOfTheCouple, pWoman, pMutation, nIdPop_nMoins1, ArrOfPop, 
			   dp=None, CoresVals=None):
	"""
	Generate the childs created by the couples.

	Parameters
	----------
	ArrayOfTheCouple : numpy.ndarray
		Array that countain the couple formed by the MatcherCouple function.
	pWoman : float
		Probability to drawn a woman. I must be in the range of 0 to 1..
	pMutation : float
		Mutation probability (whitout taking in count the parenty between
		people).
	nIdPop_nMoins1 : int
		Number of individuals that were generated from the beginning.
	ArrOfPop : numpy.ndarray
		A 2 dimensions array of all individuals generated during simulation.
	dp : numpy.ndarray, optional
		A 2 dimensions array. It is the cumulative sum of the probability
		density of having n children (n coming from CoresVals). The default
		is None.
	CoresVals : numpy.ndarray, optional
		A 1 dimension array. It is the boundaries between the different
		density probability function about the number of child. The default is
		None.

	Raises
	------
	ValueError
		An error were found either in dp or CoresVals.

	Returns
	-------
	ChildsArray : numpy.ndarray
		A 2 dimension array. This is the child array.

	"""
	ChildsList, Len, shp = [], len(ArrayOfTheCouple), ArrOfPop.shape
	if type(CoresVals) != type(None):
		uCV = 0
		for i in range(len(CoresVals)-1):
			if (CoresVals[i] < Len)&(Len <= CoresVals[i+1]):
				uCV = CoresVals[i]
			elif (CoresVals[i+1] <= Len)&(i == len(CoresVals)-2):
				uCV = CoresVals[i+1]
		udp = np.cumsum(dp[CoresVals == uCV][0], dtype=float)
		udp = np.append(0, udp)
	else:
		if len(ArrayOfTheCouple) <= 10:
			udp = np.array([0., 0.00, 0.01, 0.09, 0.40, 0.68, 0.85, 1])
		elif len(ArrayOfTheCouple) <= 30:
			udp = np.array([0., 0.01, 0.06, 0.24, 0.52, 0.72, 0.88, 1])
		else:
			udp = np.array([0., 0.01, 0.08, 0.53, 0.88, 0.96, 0.99, 1])
	npschs = np.arange(len(udp))
	for i in range(len(ArrayOfTheCouple)):
		k = np.random.random(1)
		j = npschs[udp-k >= 0][0]-1
		for n in range(j):
			Indiv = np.full([shp[1]], '', dtype = '<U9')
			Indiv[0] = str(nIdPop_nMoins1 + len(ChildsList))
			Indiv[1] = SexeIndividus(pWoman)
			Indiv[2] = ArrayOfTheCouple[i, 0, 0]
			Indiv[3] = ArrayOfTheCouple[i, 1, 0]
			Indiv[4:] = gene_transmis(ArrayOfTheCouple[i ,0],
							 ArrayOfTheCouple[i, 1], pMutation, ArrOfPop)
			ChildsList.append(Indiv)
	ChildsArray = np.array(ChildsList, dtype='<U9')
	return ChildsArray

def CountMuts(ArrOfPop, Ngens):
	"""
	Function to count the number of people who have n mutations.

	Parameters
	----------
	ArrOfPop : numpy.ndarray
		A 2 dimensions array of the people of the generation.
	Ngens : int
		Number of gene pairs.

	Returns
	-------
	mutgs : numpy.ndarray
		A 1 dimension array of the number of people who have n mutated gens.

	"""
	mutgs = np.zeros(int(2*Ngens+1))
	for j in range(len(ArrOfPop)):
		temp = np.array(list(np.sum(ArrOfPop[j, 4:].astype('O'))))
		mutgs[len(temp[temp == 'a'])] += 1
	return mutgs

def Evoluteur(NGeneration, LenPopini, pWoman, Ngens, pMutation, ppolygam=0,
			  PerfEquiIni=False, dp=None, CoresVals=None):
	"""
	Function that gather the other function to simulate the evolution of the
	initial population.

	Parameters
	----------
	NGeneration : int
		The number of generation you want to generate, must be >=1.
	LenPopini : int
		Number of people in the initial population, avoid descending below
		10 peoples.
	pWoman : float
		Probability to drawn a woman. It must be in the range of 0 to 1.
	Ngens : int
		Number of gene pairs.
	pMutation : float
		Mutation probability (whitout taking in count the parenty between
		people).
	ppolygam : float, optional
		Probability of infedility. If equal to 0 then all couples will be
		strictly monogamous. If equal to 1 then all couples will be fully
		polygamous. Note tha the the rate of polygamous is the same for the
		two sex.
	PerfEquiIni : bool, optional
		If True, the initial population will have the best equilibrium between
		the number of men and women. The default is False.
	dp : numpy.ndarray, optional
		A 2 dimensions array. It is the cumulative sum of the probability
		density of having n children (n coming from CoresVals). The default
		is None.
	CoresVals : numpy.ndarray, optional
		A 1 dimension array. It is the boundaries between the different
		density probability function about the number of child. The default is
		None.

	Returns
	-------
	Mut_N : numpy.ndarray
		A 1 dimension array, number of idividus with n mutation.
	EvoLenPop : numpy.ndarray
		A 1 dimension array, Number of peoples by generation.
	EvoLenPopCum : numpy.ndarray
		A 1 dimension array, Cumulative sum of the number of people by
		generation.
	AllPop : numpy.ndarray
		A 2 dimensions array, countain all the people generated during the
		simulation.

	"""
	if type(CoresVals) != type(None):
		#All this part is only to check if there are error in variable dp
		# and/or CoresVals.
		if (type(CoresVals) != list)&(type(CoresVals) != np.ndarray):
			CoresVals = np.ravel(CoresVals)
		if len(CoresVals) < 1:
			raise ValueError(
	"Corresponding values with the given probability density must be > 0.")
		if len(CoresVals) != len(dp):
			raise ValueError("""There must be the same number of density
			probability and boundaries on corresponding population length.""")
		if CoresVals.dtype != int:
			CoresVals = np.array(CoresVals, dtype=int)
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
	Mut_N = []
	EvoLenPop, EvoLenPopCum = [], []
	Gnn = IndividusIni(LenPopini, pWoman, Ngens, PerfEquiIni)
	Couple = MatcherCouple(Gnn, ppolygam)
	Evolution = []
	Evolution.append(Gnn)
	print("Tere are ",LenPopini," peopl :", len(Gnn[Gnn[:, 1] == 'f']),
	   "women and", len(Gnn[Gnn[:, 1] == 'm']), "men.")
	Mut_N.append(CountMuts(Gnn, Ngens))
	EvoLenPop.append(len(Gnn))
	EvoLenPopCum.append(np.cumsum(EvoLenPop))
	for i in range(NGeneration):
		Gnn = Descendant(Couple, pWoman, pMutation, EvoLenPopCum[i][-1],
				   np.concatenate(Evolution), dp, CoresVals)
		if len(Gnn) == 0:
			print("The population get extinct at the", i, "generation.")
			break
		else:
			Couple = MatcherCouple(Gnn, ppolygam)
			Mut_N.append(CountMuts(Gnn, Ngens))
			Evolution.append(Gnn)
			EvoLenPop.append(len(Gnn))
			EvoLenPopCum.append(np.cumsum(EvoLenPop))
			print("Generation", str(i+1)+"/"+str(NGeneration))
			print("\t", len(Gnn), 'individus')
	Mut_N = np.array(Mut_N)
	EvoLenPop = np.array(EvoLenPop, dtype=float)
	EvoLenPopCum = np.cumsum(EvoLenPop)
	AllPop = np.concatenate(Evolution)
	return Mut_N, EvoLenPop, EvoLenPopCum, AllPop
