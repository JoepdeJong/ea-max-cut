import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

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

def getDataFrame(folderpath):
    all_results = []
    results_dict = findAllTextFilesInFolder(folderpath)
    for file in results_dict:
        instance = file.split('-')[1]
        crossover = file.split('-')[2].replace('.txt','')
        with open(f'{file}',"r") as f:
            result_line = f.readlines()[-1]
            values = [float(el) for el in result_line.split(", ")]
            # if we already have instance in all results, we append the values for that operator
            all_results.append([instance,crossover,*values])
    df = pd.DataFrame(all_results,columns=['Instance','CrossoverOperator','pop_size','perc_suc_runs','10th percentile','50th percentile','90th percentile'])
    df.set_index(['Instance','CrossoverOperator'])
    return df

def plotBarChartResult(folderpath):
    df = getDataFrame(folderpath)[1:15]
    seaborn.set(style='ticks')
    seaborn.set(rc={'figure.figsize': (11.7, 8.27)})
    fg = seaborn.catplot(x='Instance',y='perc_suc_runs',hue='CrossoverOperator',
                            data=df,kind='bar',height=8.27,aspect=11.7/8.27)
    fg.set_xlabels('')
    plt.xticks(rotation=90)
    plt.show()

if __name__ == "__main__":
    # dataset = 'maxcut-instances/setE'
    # crossovers = ["UniformCrossover", "OnePointCrossover"]
    # globalEvaluation(dataset,crossovers)
    # x = getDataFrame("results/setE")
    # print(x)
    plotBarChartResult("results/setE")