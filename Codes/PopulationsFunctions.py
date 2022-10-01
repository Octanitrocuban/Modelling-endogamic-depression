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
		Probability to drawn a woman. I must be in the range of 0 to 1.
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
	return Pop_gen_0




