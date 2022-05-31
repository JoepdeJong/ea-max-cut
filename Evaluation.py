import numpy as np

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

def evaluateInstances(allInstances,crossoverOperators,population_sizes = [500], num_runs = 15):
    for population_size in population_sizes:
        for inst in allInstances:
            inst_name = inst.split('/')[-1].replace('.txt','')
            for cx in crossoverOperators:
                with open(f"results/output.txt", "a") as f:
                    num_evaluations_list = []
                    num_success = 0
                    for i in range(num_runs):
                        fitness = FitnessFunction.MaxCut(inst)
                        print(f'fitness.value_to_reach: {fitness.value_to_reach}')
                        genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=cx, evaluation_budget=100000,
                                                             verbose=False)
                        best_fitness, num_evaluations = genetic_algorithm.run()
                        if best_fitness == fitness.value_to_reach:
                            num_success += 1
                        num_evaluations_list.append(num_evaluations)
                    print("{}/{} runs successful".format(num_success, num_runs))
                    print("{} evaluations (median)".format(np.median(num_evaluations_list)))
                    percentile_list = [10,50,90]
                    percentiles = np.percentile(num_evaluations_list, percentile_list)
                    f.write("{}, {}, {}, {}, {}, {}, {}\n".format(inst_name,cx,population_size, num_success / num_runs, percentiles[0], percentiles[1],
                                                      percentiles[2]))

if __name__ == "__main__":
    instances= ["maxcut-instances/setE/n0000020i00.txt", "maxcut-instances/setE/n0000010i00.txt",
                             "maxcut-instances/setE/n0000040i00.txt", "maxcut-instances/setE/n0000080i00.txt",
                             "maxcut-instances/setE/n0000160i00.txt"]
    population_sizes = [10, 20, 40, 80, 160, 320, 640, 1280, 2560]
    crossovers = ["UniformCrossover", "OnePointCrossover"]
    evaluateInstances(instances, crossovers, population_sizes, num_runs=15)