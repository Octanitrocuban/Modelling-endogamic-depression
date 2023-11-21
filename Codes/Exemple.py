# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import PopulationsFunctions as pf
import ShowFunctions as sf
#=============================================================================
# Number of generation
Gener = 4
# Number of pair of gens
ngens = 4
# Probability that a children be a woman
pW = 0.5
# Bases probabilities for a mutation
pMutList = 1*10**np.arange(-5, 0, dtype=float)
# If the initial population will be the best equilibrate as possible
PerfEq = True
# Probability density function for couple to have n childs
dpc = np.array([[0.00, 0.02, 0.07, 0.32, 0.30, 0.19, 0.10],
				[0.03, 0.08, 0.28, 0.28, 0.18, 0.09, 0.06],
				[0.12, 0.19, 0.35, 0.26, 0.06, 0.01, 0.01],
				[0.21, 0.28, 0.38, 0.10, 0.02, 0.01, 0.00],
				[0.50, 0.30, 0.14, 0.03, 0.02, 0.01, 0.00]],
				dtype=float)
# Boundaries that make the population use different pdf of childs
CorVal = np.array([0, 10, 20, 30, 40])
# Strength of the consanguinity score
Consg = 0.005
# Percentage of polygmous couple
pgam = 0.1

# Base probability for a mutation
pM = 0.000001
# initial size
Len0 = 2


MutN, EvoL, EvoLCum, AllPop = pf.Evoluteur(Gener, Len0, pW, ngens, pM, pgam,
										   PerfEq, dpc, CorVal, Consg,
										   maxdeep=10)

"""
for Len0 in range(2, 52, 2):# Number of people into the first generaion
	for EarlS in range(16):#Early stoping of the origines tree
		for i in range(500):
			# Calling the main function
			MutN, EvoL, EvoLCum, AllPop = pf.Evoluteur(Gener, Len0, pW,
												ngens, pM, pgam, PerfEq, dpc,
												CorVal, Consg, EarlS)
"""
# Functions used to represent the data generated
# Evolution of the proportion of mutated gens over generations
sf.plotEvolGen(MutN, FgSz=(16, 6), Leg=True)

sf.plotEvolConsang(AllPop, FgSz=(16, 6), Leg=False, marge=0.01)

# Evolution of the length of the population over generations
sf.EvolLenPopShow(EvoL, EvoLCum, FgSz=(16, 6))

# Genealogic links between generated people over generations
#sf.GenealogicTree(AllPop, Figsize=(17, 12), STitle=20, marge=0.015)

# Genealogic links between generated people over generations plus the mutation
# rate of eache people (number of gens/number of mutated gens)
sf.InformativLinearTree(AllPop, Figsize=(17, 14), STitl=20, Sscat=28,
						Cmap='gnuplot', marge=0.015)

# Genealogic links between generated people over generations plus the level of
# consanguinity between the person of the couples
#sf.ConsangLinearTree(AllPop, Figsize=(17, 14), STitl=20, Sscat=28,
#						Cmap='gnuplot', marge=0.015)
