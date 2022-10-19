# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 17:30:52 2022

@author: Matthieu Nougaret
"""
import numpy as np
import PopulationsFunctions as pf
import ShowFunctions as sf
#=============================================================================
# Number of generation
Gener = 5
# Number of people into the first generaion
Len0 = 6
# Probability that a children be a woman
pW = 0.5
# Number of pair of gens
ngens = 500
# Base probability for a mutation
pM = 0.0000001
# Percentage of polygmous couple
pgam = 0.0
# If the initial population will be the best equilibrate as possible
PerfEq = True
# Probability density function for couple to have n childs
dpc = np.array([[0.00, 0.01, 0.06, 0.32, 0.30, 0.19, 0.12],
				[0.03, 0.08, 0.28, 0.28, 0.18, 0.09, 0.06],
				[0.12, 0.19, 0.35, 0.26, 0.06, 0.01, 0.01],
				[0.21, 0.28, 0.38, 0.10, 0.02, 0.01, 0.00],
				[0.50, 0.30, 0.14, 0.03, 0.02, 0.01, 0.00]],
				dtype=float)
# Boundaries that make the population use different pdf of childs
CorVal = np.array([0, 15, 30, 45, 80])
# Strength of the consanguinity score
Consg = 4
#Early stoping of the origines tree
EarlS = 10

# Calling the main function
MutN, EvoL, EvoLCum, AllPop = pf.Evoluteur(Gener, Len0, pW, ngens, pM, pgam,
										   PerfEq, dpc, CorVal, Consg, EarlS)

# Functions used to represent the data generated
# Evolution of the proportion of mutated gens over generations
sf.plotEvolGen(MutN, FgSz=(16, 6), Leg=False)
# Evolution of the length of the population over generations
sf.EvolLenPopShow(EvoL, EvoLCum, FgSz=(16, 6))
# Genealogic links between generated people over generations
sf.GenealogicTree(AllPop, Figsize=(17, 12), STitle=20)
# Genealogic links between generated people over generations plus the mutation
# rate of eache people (number of gens/number of mutated gens)
sf.InformativLinearTree(AllPop, Figsize=(17, 14), STitl=20, Sscat=28,
						Cmap='turbo')
