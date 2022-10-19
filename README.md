# Modelling-endogamic-depression
In these repository, I will try to model the endogamic depression by creating a population and making it evoluate.

"Inbreeding depression is the reduced biological fitness which has the potential to result from inbreeding (the breeding of related individuals)." Wikipedia

The script are in the Codes folder.
The python script: "PopulationsFunctions.py" countains the function to make the simulation.

 The function insde are:
  
  - SexeIndividus: Sex of the people in function of the probability to drawn a woman.
    
  - IndividusIni: Create the initial population with nIndividus. With proportion of women is pWomen.

  - MatcherCouple: Function that creat the couple that will give (or not) childs.
    
  - Origines: Reconstructs the parent tree of the target individual.
    
  - Kinship: Function that compute the genetic kinship between the mother and the father.
    
  - gene_transmis: Function managing the transmission of genes from parents to their child.
    
  - Descendant: Generate the childs created by the couples.
    
  - CountMuts: Function to count the number of people who have n mutations.
    
  - Evoluteur: Function that gather the other function to simulate the evolution of the initial population.

The python script: "ShowFunctions.py" countain the function to represent the data generated.

 The function insde are:

  - plotEvolGen: This fuction is used to show the evolution of the proportion of mutation in the population through generation.

  - EvolLenPopShow: Represents changes in population size over generations.

  - GenealogicTree: Represents the family tree from the LUCA ancestor (identifier -1) of all individuals generated during the simulation. Gender and descent/ancestry links are indicated.

  - InformativLinearTree: This fuction used to show the genealogical link between the peoples created during the simulation, and their mutation rate (=number of mutated gens/total number of gens).






