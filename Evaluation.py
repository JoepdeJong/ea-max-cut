import numpy as np

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction
# Read all .txt files from folder dataset
import os

def findAllTextFilesInFolder(folder):
    files = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            files.append(f'{folder}/{file}')
    return files


def globalEvaluation(instanceFolder,crossoverOperators,population_size = 500, num_runs = 30):
    allInstances = findAllTextFilesInFolder(instanceFolder)
    print(allInstances)
    for inst in allInstances:
        print(inst)
        for cx in crossoverOperators:
            with open(f"results/{instanceFolder.split('/')[-1]}/output-{inst.split('/')[-1].replace('.txt','')}-{cx}.txt", "w") as f:
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
                f.write(f"population_size, num_succes/num_runs, {percentile_list[0]}th percentile, {percentile_list[1]}th percentile, {percentile_list[2]}th percentile")
                f.write("{}, {}, {}, {}, {}\n".format(population_size, num_success / num_runs, percentiles[0], percentiles[1],
                                                  percentiles[2]))




if __name__ == "__main__":
    dataset = 'maxcut-instances/setE'
    crossovers = ["UniformCrossover", "OnePointCrossover"]
    globalEvaluation(dataset,crossovers)
    # for cx in crossovers:
    #     inst = "maxcut-instances/setE/n0000020i00.txt"
    #     with open("output-{}.txt".format(cx), "w") as f:
    #         population_size = 500
    #         num_evaluations_list = []
    #         num_runs = 30
    #         num_success = 0
    #         for i in range(num_runs):
    #             fitness = FitnessFunction.MaxCut(inst)
    #             print(f'fitness.value_to_reach: {fitness.value_to_reach}')
    #             genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=cx, evaluation_budget=100000,
    #                                                  verbose=False)
    #             best_fitness, num_evaluations = genetic_algorithm.run()
    #             if best_fitness == fitness.value_to_reach:
    #                 num_success += 1
    #             num_evaluations_list.append(num_evaluations)
    #         print("{}/{} runs successful".format(num_success, num_runs))
    #         print("{} evaluations (median)".format(np.median(num_evaluations_list)))
    #         percentiles = np.percentile(num_evaluations_list, [10, 50, 90])
    #         f.write("{} {} {} {} {}\n".format(population_size, num_success / num_runs, percentiles[0], percentiles[1],
    #                                           percentiles[2]))
