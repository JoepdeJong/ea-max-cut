import copy
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
	## Configure testing
	testing = False
	##
	assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"

	cliques = copy.copy(individual_a.cliques)
	number_of_cliques = len(cliques)

	if testing:
		# Store the sum of the partial fitnesses, to check if it is correct
		computed_fitness = np.zeros((2, number_of_cliques))
	# Create two offsprings using uniform crossover
	offspring_a, offspring_b = uniform_crossover(individual_a, individual_b, p=0.5, number_of_cliques=number_of_cliques)

	# Evaluate the offsprings on the cliques and compare their fitness
	# We also want to maximize the cut between the cliques. The cliques are ordered in a chain so we keep track of the last node to make the cut.
	for clique_number, clique in enumerate(cliques):
		# Compute fitness of the offsprings on the clique:
		fitness.evaluate_partial(offspring_a, clique_number)
		fitness.evaluate_partial(offspring_b, clique_number)
		

		# # Find the 2 best individuals from the parents and offsprings on the current clique
		individuals = copy.deepcopy([
			individual_a, individual_b,
			offspring_a, offspring_b
		])

		# Fitnesses
		fitnesses = np.array([
			individual_a.partial_fitness[clique_number],
			individual_b.partial_fitness[clique_number],
			offspring_a.partial_fitness[clique_number],
			offspring_b.partial_fitness[clique_number]
		])

		# Get the indices of the sorted fitnesses
		sorted_fitness_indices = np.argsort(fitnesses)[::-1]

		# # Use the genotype of the best two individuals to update the genotype of the clique
		individual_a.genotype[clique] = individuals[sorted_fitness_indices[0]].genotype[clique]
		individual_a.partial_fitness[clique_number] = individuals[sorted_fitness_indices[0]].partial_fitness[clique_number]

		individual_b.genotype[clique] = individuals[sorted_fitness_indices[1]].genotype[clique]
		individual_b.partial_fitness[clique_number] = individuals[sorted_fitness_indices[1]].partial_fitness[clique_number]
		
		## Testing
		if testing:
			np.testing.assert_array_equal(individual_a.genotype[clique], individuals[sorted_fitness_indices[0]].genotype[clique])
			np.testing.assert_array_equal(individual_b.genotype[clique], individuals[sorted_fitness_indices[1]].genotype[clique])
			np.testing.assert_array_equal(individual_a.partial_fitness[clique_number], individuals[sorted_fitness_indices[0]].partial_fitness[clique_number])
			np.testing.assert_array_equal(individual_b.partial_fitness[clique_number], individuals[sorted_fitness_indices[1]].partial_fitness[clique_number])

			computed_fitness[0, clique_number] = individual_a.partial_fitness[clique_number]
			computed_fitness[1, clique_number] = individual_b.partial_fitness[clique_number]

			print(individual_a.genotype[clique], individual_a.partial_fitness[clique_number], clique)
			fitness.evaluate_partial(individual_a, clique_number)
			print(individual_a.genotype[clique], individual_a.partial_fitness[clique_number], clique)
			assert(individual_a.partial_fitness[clique_number] == computed_fitness[0, clique_number])

		# Make cuts between cliques
		if clique_number != 0:
			# Make a cut between the last clique and the current clique if the value of the nodes that connect them is the same
			if individual_a.genotype[last_clique_node] == individual_a.genotype[clique[0]]:
				individual_a.genotype[clique] = 1 - individual_a.genotype[clique]

			if individual_b.genotype[last_clique_node] == individual_b.genotype[clique[0]]:
				individual_b.genotype[clique] = 1 - individual_b.genotype[clique]

			if testing:
				fitness.evaluate_partial(individual_a, clique_number)
				assert(individual_a.partial_fitness[clique_number] == computed_fitness[0, clique_number])
				fitness.evaluate_partial(individual_b, clique_number)
				assert(individual_b.partial_fitness[clique_number] == computed_fitness[1, clique_number])
		
		# Set the node in the clique that has an edge to the next clique
		last_clique_node = clique[-1]

	inter_clique_fitness = np.zeros(2)
	
	for edge in fitness.inter_clique_edges:
		inter_clique_fitness[0] += fitness.weights[edge]
		inter_clique_fitness[1] += fitness.weights[edge]
		
		if testing:
			# Validate inter clique cut
			assert(individual_a.genotype[edge[0]] != individual_a.genotype[edge[1]])
			assert(individual_b.genotype[edge[0]] != individual_b.genotype[edge[1]])

	if testing:
		# Compute fitness of the offsprings:
		fitness.evaluate(individual_a)
		fitness.evaluate(individual_b)

		print('A', 'p', individual_a.partial_fitness)
		# print(individual_a.genotype)
		print('Before recomputing', np.sum(individual_a.partial_fitness), inter_clique_fitness[0])
		for clique_number, clique in enumerate(cliques):
			fitness.evaluate_partial(individual_a, clique_number)
			fitness.evaluate_partial(individual_b, clique_number)
			if clique_number == 7:
				print('A', clique_number, individual_a.partial_fitness, individual_a.genotype[clique])
		# print(individual_a.genotype)
		print('A', 'p', individual_a.partial_fitness)
		print('After recomputing', np.sum(individual_a.partial_fitness), inter_clique_fitness[0])


		print("Computed fitness:", np.sum(computed_fitness, 1), inter_clique_fitness)
		print("Fitness:", individual_a.fitness, individual_b.fitness)
		assert(individual_a.fitness == np.sum(computed_fitness[0,:]) + inter_clique_fitness[0])
		assert(individual_b.fitness == np.sum(computed_fitness[1,:]) + inter_clique_fitness[1])


		assert(individual_a.fitness == np.sum(individual_a.partial_fitness) + inter_clique_fitness[0])
		assert(individual_b.fitness == np.sum(individual_b.partial_fitness) + inter_clique_fitness[1])
	
	individual_a.fitness = np.sum(individual_a.partial_fitness) + inter_clique_fitness[0]
	individual_b.fitness = np.sum(individual_a.partial_fitness) + inter_clique_fitness[1]

	return [individual_a, individual_b]
