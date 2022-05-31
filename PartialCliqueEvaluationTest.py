import unittest
import numpy as np


from Individual import Individual
from FitnessFunction import MaxCut


class partial_evaluation_test(unittest.TestCase):
    inst = "maxcut-instances/setE/n0000020i00.txt"
    fitness = MaxCut(inst, clique_size=5)


    def test_inter_clique_edges(self):
        fitness = self.fitness
        #individual = Individual.initialize_uniform_at_random(10)
        for i in range(len(fitness.cliques)):
            clique = fitness.cliques[i]
            clique_edges = fitness.clique_edges[i]
            print(clique)
            print(clique_edges)
            self.assertEqual(len(set(clique_edges)), len(clique_edges))
            self.assertEqual( clique.sort() , ([a for a,_ in clique_edges]+[b for _,b in clique_edges]).sort() )


    def test_partial_evaluation(self):
        fitness = self.fitness
        l = fitness.dimensionality
        clique_number = 0
        individual = Individual.initialize_uniform_at_random(l)
        clique = fitness.cliques[clique_number]
        print(clique)
        individual_c = Individual(np.zeros(l))
        individual_c.genotype[clique] =  individual.genotype[clique]

        fitness.evaluate_partial(individual_c)
        fitness_global = individual_c.fitness
        fitness.evaluate_partial(individual_c, clique_number)
        fitness_partial = individual_c.fitness

        # out_going_nodes = []
        # for v in clique:
        #     if v in fitness.adjacency_list and len(fitness.adjacency_list[v]) == len(clique):
        #         out_going_nodes.append(v)

        self.assertEqual(fitness_global, fitness_partial)


if __name__ == "__main__":
    unittest.main()