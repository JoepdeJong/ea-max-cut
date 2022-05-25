## Evolutionary Algorithm - Max Cut

# Week 1

- Present initial ideas for selection of MaxCut instances to be considered.

Idea for chains of cliques: If we can solve the cliques optimally, then a one-point cross-over between the chain edges of the clique would yield us an optimal solution. Note: We can always create an optimal solution by the 'polarity' argument.

Challenge : How can we solve such a clique optimally, given two parents?
Ideas: Brute Forcing? Weighted uniform crossovers?


- Present initial ideas for GBO improvements:
Idea: Changing the color of a single nodes only affects the adjacent nodes, so you can find local optima in which only a subset of possible solution needs to be considered. 

Can we use these kind of 'quick wins' that are indepedent of the parents to better the algorithm? 


Q's:
- To what extent can we change the code?
- Can we access the weights of edges?
