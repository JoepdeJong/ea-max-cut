import numpy as np
import copy


import Individual
from Utils import ValueToReachFoundException

class FitnessFunction:
	def __init__( self ):
		self.dimensionality = 1 
		self.number_of_evaluations = 0
		self.value_to_reach = np.inf
		self.partial_fitness_to_reach = np.inf

	def evaluate( self, individual: Individual ):
		self.number_of_evaluations += 1
		if individual.fitness >= self.value_to_reach:
			raise ValueToReachFoundException(individual)

	def evaluate_partial(self, individual: Individual, clique_number: int):
		#TODO define partial evaluation weight
		#TODO find proper value for partial fitness to reach
		partial_evaluation_weight = len(self.cliques[clique_number])/len(individual.genotype)
		self.number_of_evaluations += partial_evaluation_weight
		if individual.partial_fitness[clique_number] >= self.partial_fitness_to_reach:
			raise ValueToReachFoundException(individual)
		pass

class OneMax(FitnessFunction):
	def __init__( self, dimensionality ):
		super().__init__()
		self.dimensionality = dimensionality
		self.value_to_reach = dimensionality

	def evaluate( self, individual: Individual ):
		individual.fitness = np.sum(individual.genotype)
		super().evaluate(individual)

class DeceptiveTrap(FitnessFunction):
	def __init__( self, dimensionality ):
		super().__init__()
		self.dimensionality = dimensionality
		self.trap_size = 5
		assert dimensionality % self.trap_size == 0, "Dimensionality should be a multiple of trap size"
		self.value_to_reach = dimensionality

	def trap_function( self, genotype ):
		assert len(genotype) == self.trap_size
		k = self.trap_size
		bit_sum = np.sum(genotype)
		if bit_sum == k:
			return k
		else:
			return k-1-bit_sum

	def evaluate( self, individual: Individual ):
		num_subfunctions = self.dimensionality // self.trap_size
		result = 0
		for i in range(num_subfunctions):
			result += self.trap_function(individual.genotype[i*self.trap_size:(i+1)*self.trap_size])
		individual.fitness = result
		super().evaluate(individual)

class MaxCut(FitnessFunction):
	def __init__( self, instance_file, clique_size = 0 ):
		super().__init__()
		self.edge_list = []
		self.weights = {}
		self.adjacency_list = {}
		self.clique_size = clique_size
		self.read_problem_instance(instance_file)
		self.read_value_to_reach(instance_file)
		self.preprocess()

	def preprocess( self ):
		pass

	def read_problem_instance( self, instance_file ):
		with open( instance_file, "r" ) as f_in:
			lines = f_in.readlines()
			first_line = lines[0].split()
			self.dimensionality = int(first_line[0])
			number_of_edges = int(first_line[1])
			for line in lines[1:]:
				splt = line.split()
				v0 = int(splt[0])-1
				v1 = int(splt[1])-1
				assert( v0 >= 0 and v0 < self.dimensionality )
				assert( v1 >= 0 and v1 < self.dimensionality )
				w = float(splt[2])
				self.edge_list.append((v0,v1))
				self.weights[(v0,v1)] = w
				self.weights[(v1,v0)] = w
				if( v0 not in self.adjacency_list ):
					self.adjacency_list[v0] = []
				if( v1 not in self.adjacency_list ):
					self.adjacency_list[v1] = []
				self.adjacency_list[v0].append(v1)
				self.adjacency_list[v1].append(v0)

			if(self.clique_size > 0):
				self.cliques, self.clique_edges, self.inter_clique_edges = self.get_cliques(self.adjacency_list, self.clique_size)

			assert( len(self.edge_list) == number_of_edges )
	
	def read_value_to_reach( self, instance_file ):
		bkv_file = instance_file.replace(".txt",".bkv")
		with open( bkv_file, "r" ) as f_in:
			lines = f_in.readlines()
			first_line = lines[0].split()
			self.value_to_reach = float(first_line[0])

	def get_weight( self, v0, v1 ):
		if( not (v0,v1) in self.weights ):
			return 0
		return self.weights[(v0,v1)]

	def get_degree( self, v ):
		return len(adjacency_list(v))

	def evaluate( self, individual: Individual ):
		result = 0
		for e in self.edge_list:
			result += self.evaluate_edge(individual, e)

		individual.fitness = result
		super().evaluate(individual)

	def evaluate_partial(self, individual: Individual, clique_number: int):
		result = 0
		for e in self.clique_edges[clique_number]:
			result += self.evaluate_edge(individual, e)

		individual.partial_fitness[clique_number] = result
		super().evaluate_partial(individual, clique_number)

	def evaluate_edge(self, individual: Individual, e):
		v0, v1 = e
		w = self.weights[e]
		if( individual.genotype[v0] != individual.genotype[v1] ):
			return w
		return 0

	def get_cliques(self, adjacency_list, clique_size):
		cliques = []
		cliques_edges = []
		inter_clique_edges = []
		# copy adjacency_list here cause otherwise the items wil be removed from original adjacency_list
		adjacency_list = copy.copy(adjacency_list)
		
		for v in adjacency_list:
			# Find nodes with out_degree the same as clique size. (without counting the node itself, this means that this node has an edge to a node outside the clique)
			if len(adjacency_list[v]) == clique_size:
				# Find the adjacent node that is also an 'inter clique' node.
				for v_adj in adjacency_list[v]:
					if len(adjacency_list[v_adj]) == clique_size:
						inter_clique_edges.append((v,v_adj))

		while( len(adjacency_list) > 0 ):
			clique = []
			w = None
			for v in adjacency_list:
				if( len(adjacency_list[v]) == clique_size - 1 ):
					w = v
					break

			if w is None:
				print("Error: clique not found")
				break
			
			clique.append(w)
			clique.extend(adjacency_list[w])
			cliques.append(clique)

			clique_edge = []
			for v1 in clique:
				for v2 in clique:
					if v1 in adjacency_list and v2 in adjacency_list[v1] and (v2, v1) not in clique_edge:
						clique_edge.append((v1,v2))

			for v in adjacency_list[w]:
				del adjacency_list[v]
			del adjacency_list[w]

			cliques_edges.append(clique_edge)

		return cliques, cliques_edges, inter_clique_edges


