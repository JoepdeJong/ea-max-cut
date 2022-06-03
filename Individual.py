import numpy as np

class Individual:
	def __init__(self, genotype = [], number_of_cliques = None ):
		self.genotype = np.array(genotype)
		self.fitness = 0
		self.cliques = []
		self.partial_fitness = np.zeros(number_of_cliques)

	
	def initialize_uniform_at_random(genotype_length):
		individual = Individual()
		individual.genotype = np.random.choice((0,1), p=(0.5, 0.5), size=genotype_length)
		return individual
