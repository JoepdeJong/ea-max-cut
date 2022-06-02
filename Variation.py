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

	cliques = individual_a.cliques
	number_of_cliques = len(cliques)

	# Create two offsprings using uniform crossover
	offspring_a, offspring_b = uniform_crossover(individual_a, individual_b, p=0.5, number_of_cliques=number_of_cliques)

	# Evaluate the offsprings on the cliques and compare their fitness
	for clique_number, clique in enumerate(cliques):
		# Perform crossover inside the clique to generate the offsprings

		# Compute fitness of the offsprings:
		fitness.evaluate_partial(offspring_a, clique_number)
		fitness.evaluate_partial(offspring_b, clique_number)

		# Find the 2 best individuals from the parents and offsprings
		individuals = [
			individual_a, individual_b,
			offspring_a, offspring_b
		]

		sorted_individuals = sorted(individuals, key=lambda x: x.partial_fitness[clique_number], reverse=True)

		# Use the genotype of the best two individuals
		individual_a.genotype[clique] = sorted_individuals[0].genotype[clique]
		individual_a.partial_fitness[clique_number] = sorted_individuals[0].partial_fitness[clique_number]
		individual_b.genotype[clique] = sorted_individuals[1].genotype[clique]
		individual_b.partial_fitness[clique_number] = sorted_individuals[1].partial_fitness[clique_number]

	# We want to maximize the cut between the cliques.
	# The cliques are already ordered in a chain, 
	# so we can just make a cut between every clique.
	last_clique_node = None
	for clique_number, clique in enumerate(cliques):
		if clique_number == 0:
			# Set the node in the clique that has an edge to the next clique
			last_clique_node = clique[-1]
			continue
		
		last_clique_node = cliques[clique_number-1][-1]
		
		# Make a cut between the last clique and the current clique if the value of the nodes that connect them is the same
		do_crossover_a = individual_a.genotype[last_clique_node] == individual_a.genotype[clique[0]]
		do_crossover_b = individual_b.genotype[last_clique_node] == individual_b.genotype[clique[0]]

		if do_crossover_a:
			individual_a.genotype[clique] = 1 - individual_a.genotype[clique]
		if do_crossover_b:
			individual_b.genotype[clique] = 1 - individual_b.genotype[clique]
		
		last_clique_node = clique[-1]

	# Validate inter clique cut
	for edge in fitness.inter_clique_edges:
		assert(individual_a.genotype[edge[0]] != individual_a.genotype[edge[1]])
		assert(individual_b.genotype[edge[0]] != individual_b.genotype[edge[1]])

	# Compute fitness of the offsprings:
	fitness.evaluate(individual_a)
	fitness.evaluate(individual_b)

	return [individual_a, individual_b]
