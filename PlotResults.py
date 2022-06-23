import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

sns.set_theme()
sns.set_palette(sns.color_palette("Set2"))

def make_suc_perc_plot(df):
    res = sns.relplot(data= df, x='pop-size', y='succes-percentage',
                col='instance', col_order = ['n0000010i00', 'n0000020i00', 'n0000040i00', 'n0000080i00', 'n0000160i00'],hue='crossover-operator',
                style='crossover-operator',kind='line',col_wrap=3)
    res.set(xscale="log")
    res.fig.subplots_adjust(top=.9)
    res.set(xlabel="Population size")
    res.set(ylabel="Percentage of successful runs")
    for ax in res.axes.flat:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    res.fig.suptitle('Percentage of succes for different population sizes for different crossover-operators for different instances')
    plt.savefig("suc-perc-plot.png")

def make_eval_plot(df):
    instances = ['n0000010i00', 'n0000020i00', 'n0000040i00', 'n0000080i00', 'n0000160i00']
    crossovers = df['crossover-operator'].unique()
    fig, axs = plt.subplots(2, 3,figsize=(16.92,10),sharey=True)
    fig.subplots_adjust(right=.85,top=.9)
    fig.suptitle('Median number of evaluations (and error band for the 10th and 90th percentile) needed until the optimal solution was found\nfor different population sizes for different crossover-operators for different instances')
    for idx, instance in enumerate(instances):
        col_idx = idx % 3
        row_idx = math.trunc(idx/3)
        ax = axs[row_idx,col_idx]
        axs[-1, -1].axis('off')
        ax.set_title(f'instance = {instance}')
        marker_dict = {' CustomCrossover':':', ' UniformCrossover':'-',' OnePointCrossover': '--'}
        for crossover in crossovers:
            inst_df = df[df['instance'] == instance]
            inst_co_df = inst_df[inst_df['crossover-operator'] == crossover]
            ax.plot(inst_co_df['pop-size'],inst_co_df['50-perc'],marker_dict[crossover],label=crossover)
            ax.fill_between(inst_co_df['pop-size'], inst_co_df['10-perc'], inst_co_df['90-perc'], alpha=0.2)
            ax.set_yscale('log')
            ax.set_xscale('log')
        if idx < 2:
            plt.setp(ax.get_xticklabels(), visible=False)
        if idx == 4:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles,labels,loc='right',frameon=False,title="crossover-operator")
    for ax in axs.flat:
        ax.set(xlabel='Population Size', ylabel='Number of evaluations needed')
    for i,ax in enumerate(axs.flat):
        if i != 2:
            ax.label_outer()
    axs.flat[2].set(xlabel='Population Size',ylabel="")
    plt.savefig("evaluations_plot.png")

def make_plot_evals_crossover_growing_size(df,pop_size):
    fig, ax = plt.subplots()
    custom = df[df['crossover-operator'] == ' CustomCrossover']
    custom = custom[custom["pop-size"] == pop_size ]
    custom['instance-size'] = custom.apply(lambda row: int(row.instance[5:8]),axis=1)

    ax.plot(np.sort(custom['instance-size']), np.sort(custom['50-perc']))
    ax.fill_between(np.sort(custom['instance-size']), np.sort(custom['10-perc']), np.sort(custom['90-perc']), alpha=0.2)
    ax.set(xlabel='Size of graph in nodes', ylabel='Number of evaluations needed')
    plt.show()

def make_plots_evals_crossover_growing_size(df,pop_sizes):
    custom = df[df['crossover-operator'] == ' CustomCrossover']
    custom = custom[custom["pop-size"].isin(pop_sizes)]
    custom['instance-size'] = custom.apply(lambda row: int(row.instance[5:8]),axis=1)
    fig, axs = plt.subplots(2, 3,figsize=(16.92,10))
    # axs[-1, -1].axis('off')
    # fig.subplots_adjust(right=.85, top=.9)
    fig.suptitle('Number of evaluations needed for larger graphs using the CustomCrossover operator.')
    for idx, pop_size in enumerate(pop_sizes):
        this_custom = custom[custom["pop-size"] == pop_size]
        col_idx = idx % 3
        row_idx = math.trunc(idx / 3)
        ax = axs[row_idx, col_idx]
        ax.set_title(f'Population Size = {pop_size}')
        ax.plot(np.sort(this_custom['instance-size']), np.sort(this_custom['50-perc']))
        ax.fill_between(np.sort(this_custom['instance-size']), np.sort(this_custom['10-perc']), np.sort(this_custom['90-perc']), alpha=0.2)
        if idx <= 2:
            pass
        else:
            ax.set(xlabel='Size of graph in nodes')

        if idx in [0, 3]:
            ax.set(ylabel='Number of evaluations')
        else:
            pass
    plt.savefig("custom_crossover_growing_number_of_evals.png")


if __name__ == '__main__':
    col_names = ['instance', 'crossover-operator', 'pop-size', 'succes-percentage', '10-perc',
                 '50-perc', '90-perc']
    df = pd.read_csv("results/output.txt", header=None)
    df.set_axis(col_names, axis=1, inplace=True)
    df = df[df['pop-size'].isin([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])]
    make_suc_perc_plot(df)
    # make_eval_plot(df)
    # make_plots_evals_crossover_growing_size(df,[16, 32, 64, 128,256,512])