import numpy as np

from Individual import Individual
from FitnessFunction import FitnessFunction

def uniform_crossover(individual_a: Individual, individual_b: Individual, p = 0.5, number_of_cliques = None):
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
	l = len(individual_a.genotype)
	offspring_a = Individual(l, number_of_cliques)
	offspring_b = Individual(l, number_of_cliques)
    
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
	number_of_cliques = len(individual_a.cliques)
	offspring_a = Individual(l, number_of_cliques)
	offspring_b = Individual(l, number_of_cliques)
	offspring_a.cliques = individual_a.cliques
	offspring_b.cliques = individual_b.cliques

	# Loop over all the cliques:
	# Todo: make crossover more efficient by applying uniform crossover on the whole parent and only evaluating the fitness at the cliques to select the best sub-offsprings.
	for clique_number, clique in enumerate(individual_a.cliques):
		# Perform crossover inside the clique to generate the offsprings
		offspring_a_c, offspring_b_c = uniform_crossover(offspring_a, offspring_b, number_of_cliques=number_of_cliques)

		# Compute fitness of the offsprings:
		fitness.evaluate_partial(offspring_a_c, clique_number)
		fitness.evaluate_partial(offspring_b_c, clique_number)

		# Find the 2 best individuals from the parents and offsprings
		individuals = [
			offspring_a, offspring_b,
			offspring_a_c, offspring_b_c
		]

		sorted_individuals = sorted(individuals, key=lambda x: x.partial_fitness[clique_number], reverse=True)

		# Use the genotype of the best two individuals
		offspring_a.genotype[clique] = sorted_individuals[0].genotype[clique]
		offspring_b.genotype[clique] = sorted_individuals[1].genotype[clique]


	# TODO: one-points crossover between cliques
	# For this, we have to know the order of the cliques in the chain, since we want to crossover between cliques and crossover all the cliques after the crossover point.
	
	# Pick a random clique
	clique_a = np.random.randint(len(individual_a.cliques))
	clique_b = np.random.randint(len(individual_b.cliques))
	offspring_a2 = Individual(l)
	offspring_b2 = Individual(l)
	offspring_a2.genotype = individual_a.genotype
	offspring_b2.genotype = individual_b.genotype
	offspring_a2.cliques = individual_a.cliques
	offspring_b2.cliques = individual_b.cliques

	offspring_a2.genotype[individual_a.cliques[clique_a]] = 1 - individual_a.genotype[individual_a.cliques[clique_a]]
	offspring_b2.genotype[individual_b.cliques[clique_b]] = 1 - individual_b.genotype[individual_b.cliques[clique_b]]

	# Compute fitness of the offsprings:
	fitness.evaluate(offspring_a)
	fitness.evaluate(offspring_b)
	fitness.evaluate(offspring_a2)
	fitness.evaluate(offspring_b2)

	individuals = [
		offspring_a, offspring_b,
		offspring_a2, offspring_b2
	]
	sorted_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
		

	return [sorted_individuals[0], sorted_individuals[1]]

