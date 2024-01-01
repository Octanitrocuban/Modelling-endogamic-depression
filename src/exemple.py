# -*- coding: utf-8 -*-
"""
This file give an exemple on how to use the functions from
populations_functions and show_functions module.
"""
import numpy as np
import populations_functions as pf
import show_functions as sf
#=============================================================================
# Number of generation
generations = 15

# Number of pair of gens
ngens = 3

# Probability that a children be a woman
pwoman = 0.5

# Bases probabilities for a mutation
pMutList = 1*10**np.arange(-5, 0, dtype=float)

# If the initial population will be the best equilibrate as possible
perf_eq = True

# Probability density function for couple to have n childs
dpc = np.array([[0.00, 0.02, 0.05, 0.33, 0.30, 0.21, 0.09],
				[0.03, 0.08, 0.28, 0.28, 0.19, 0.10, 0.04],
				[0.12, 0.19, 0.35, 0.26, 0.07, 0.01, 0.00],
				[0.21, 0.28, 0.38, 0.10, 0.03, 0.00, 0.00],
				[0.50, 0.30, 0.14, 0.04, 0.02, 0.00, 0.00]],
				dtype=float)

# Boundaries that make the population use different pdf of childs
corval = np.array([0, 10, 20, 30, 40])

# Strength of the consanguinity score
consang_sc = 0.5

# Percentage of polygmous couple
ppgam = 0.1

# Base probability for a mutation
pmuta = 0.000001

# initial size
len0 = 4

# Modelisation
mutn, evo_l, evolcum, allpop = pf.evoluteur(generations, len0, pwoman, ngens,
										   pmuta, ppgam, perf_eq, dpc, corval,
										   consang_sc, maxdeep=15)

# Functions used to represent the data generated
# Evolution of the proportion of mutated gens over generations
sf.plot_evol_gen(mutn, figsize=(16, 6), save='../img/gens_repart.png')

# Evolution of the consanguinity score's in the population through generation
sf.plot_evol_consang(allpop, figsize=(16, 6),
					 marge=0.01, save='../img/consang_avg.png')

# Evolution of the length of the population over generations
sf.evol_len_pop_show(evo_l, evolcum, figsize=(16, 6),
					 save='../img/pop_len_evol.png')

# Genealogic links between generated people over generations
sf.genealogic_tree(allpop, figsize=(17, 12), size_title=20, marge=0.015,
				   save='../img/genealogic_tree.png')

# Genealogic links between generated people over generations plus the mutation
# rate of eache people (number of gens/number of mutated gens)
sf.informativ_linear_tree(allpop, figsize=(17, 14), size_title=20,
						  scat_size=28, cmap='gnuplot', marge=0.015,
						  save='../img/informativ_tree.png')

# Genealogic links between generated people over generations plus the level of
# consanguinity between the person of the couples
sf.consang_linear_tree(allpop, figsize=(17, 14), size_title=20, scat_size=28,
					   cmap='gnuplot', marge=0.015,
					   save='../img/genealogic_consang_tree.png')
