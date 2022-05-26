import numpy as np

from Individual import Individual
from FitnessFunction import FitnessFunction

def uniform_crossover(individual_a: Individual, individual_b: Individual, p = 0.5, clique = [] ):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	l = len(individual_a.genotype)
	offspring_a = Individual(l)
	offspring_b = Individual(l)
    
	m = np.random.choice((0,1), p=(p, 1-p), size=l)
	offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
	offspring_b.genotype = np.where(1 - m, individual_a.genotype, individual_b.genotype)
	
	return [offspring_a, offspring_b]

def one_point_crossover(individual_a: Individual, individual_b: Individual ):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	l = len(individual_a.genotype)
	offspring_a = Individual(l)
	offspring_b = Individual(l)
    
	l = len(individual_a.genotype)
	m = np.arange(l) < np.random.randint(l+1)
	offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
	offspring_b.genotype = np.where(~m, individual_a.genotype, individual_b.genotype)
	
	return [offspring_a, offspring_b]

def two_point_crossover(individual_a: Individual, individual_b: Individual ):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	offspring_a = Individual()
	offspring_b = Individual()
    
	l = len(individual_a.genotype)
	m = (np.arange(l) < np.random.randint(l+1)) ^ (np.arange(l) < np.random.randint(l+1))
	offspring_a.genotype = np.where(m, individual_b.genotype, individual_a.genotype)
	offspring_b.genotype = np.where(~m, individual_b.genotype, individual_a.genotype)
	
	return [offspring_a, offspring_b]

def custom_crossover( fitness: FitnessFunction, individual_a: Individual, individual_b: Individual ):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	l = np.zeros(len(individual_a.genotype))
	offspring_a = Individual(l)
	offspring_b = Individual(l)
	offspring_a.cliques = individual_a.cliques
	offspring_b.cliques = individual_b.cliques

	# Loop over all the cliques:
	for clique in individual_a.cliques:

		# Create parents inside the clique:
		parent_a_c = Individual(l)
		parent_b_c = Individual(l)

		parent_a_c.genotype[clique] = individual_a.genotype[clique]
		parent_b_c.genotype[clique] = individual_b.genotype[clique]

		# Compute fitness of the parents:
		fitness.evaluate(parent_a_c)
		fitness.evaluate(parent_b_c)

		# Perform crossover inside the clique to generate the offsprings
		offspring_a_c, offspring_b_c = uniform_crossover(parent_a_c, parent_b_c)

		# Compute fitness of the offsprings:
		fitness.evaluate(offspring_a_c)
		fitness.evaluate(offspring_b_c)

		# Find the 2 best individuals from the parents and offsprings
		individuals = [
			parent_a_c, parent_b_c,
			offspring_a_c, offspring_b_c
		]

		sorted_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)

		# Use the genotype of the best two individuals
		offspring_a.genotype[clique] = sorted_individuals[0].genotype[clique]
		offspring_b.genotype[clique] = sorted_individuals[1].genotype[clique]

	# TODO: one-points crossover between cliques
	return [offspring_a, offspring_b]

