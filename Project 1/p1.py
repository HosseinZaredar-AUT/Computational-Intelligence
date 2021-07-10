import math
import random
import numpy as np
import matplotlib.pyplot as plt
from graph import Graph, Edge


# ------------------------------------------------------------------------------------------------
# PARAMETERS
mio = 1000      # Mio
lmbd = mio      # Lambda
num_iter = 400  # number of iterations before termination
q = 100         # q parameter for Q-Tournamament selection
pi = 0.8        # probability of initializing a gene to 1 
pc = 0.5        # probability of crossover on 1 bit 
pm = 0.05       # probability of mutation on 1 bit


# ------------------------------------------------------------------------------------------------
# a function to find the Minimum Spanning Tree using Kruskal Algorithm
def get_mst(nodes, edges, chromosome):
    mapping = dict()
    count = 0
    for i in range(len(nodes)):
        if i < len(chromosome):
            if chromosome[i]:
                mapping[i] = count
                count += 1
        else:
            mapping[i] = count
            count += 1

    all_edges = []
    for edge in edges:
        if edge[0] in mapping and edge[1] in mapping:
            all_edges.append(Edge(
                mapping[edge[0]],
                mapping[edge[1]],
                edge[2]
            ))

    g = Graph(len(mapping), all_edges)
    reverse_mapping = {v: k for k, v in mapping.items()}
    found, weight, all_edges = g.KruskalMST()
    return found, weight, all_edges, reverse_mapping


# ------------------------------------------------------------------------------------------------
# a function to calculate the fitness value of a chromosome
def fitness_calc(nodes, edges, chromosome):
    found, weight, _, _ = get_mst(nodes, edges, chromosome)
    if found:
        return 1 / weight
    else:
        return 0


# ------------------------------------------------------------------------------------------------
# Fitness Selection function
def fitness_selection(population, n):

    # calculating probabilities
    fitness_sum = 0
    for i in population:
        fitness_sum += i[1]
    pis = [(i[1] / fitness_sum) for i in population]
        
    # selecting individuals and saving their indices
    selected_indices = np.random.choice(len(population), n, True, pis)
    selected = [population[i] for i in selected_indices]
    return selected


# ------------------------------------------------------------------------------------------------
# Q-Tournament Selection function
def qt_selection(population, n, q):

    selected = []
    for _ in range(n):
        max_fitness = -1
        max_fitness_idx = -1
        for _ in range(q):
            idx = np.random.randint(0, len(population))
            if population[idx][1] > max_fitness:
                max_fitness = population[idx][1]
                max_fitness_idx = idx
        selected.append(population[max_fitness_idx])

    return selected


# ------------------------------------------------------------------------------------------------
# a function to do Bitwise Crossover on 2 parents and generate 2 children
def bitwise_crossover(p1_chrom, p2_chrom):

    c1_chrom = []
    c2_chrom = []

    for i in range(len(p1_chrom)):
        coin = random.random()
        if coin < pc:
            c1_chrom.append(p2_chrom[i])
            c2_chrom.append(p1_chrom[i])
        else:
            c1_chrom.append(p1_chrom[i])
            c2_chrom.append(p2_chrom[i])

    return c1_chrom, c2_chrom


# ------------------------------------------------------------------------------------------------
# a function to do Bitwise Mutation on a chrmosome
def bitwise_mutation(graph, chromosome):
    for i in range(len(chromosome)):
        coin = random.random()
        if coin < pm:
            chromosome[i] = 1 - chromosome[i]


# ------------------------------------------------------------------------------------------------
# THE MAIN FUNCTION
def main():

    # getting the graph from input and storing it as 2D array
    print('Please provide the input:')
    n_steiner, n_terminal, n_edge = map(int, input().split())

    # getting the nodes
    nodes = []
    for _ in range(n_steiner + n_terminal):
        nx, ny = map(int, input().split())
        nodes.append((nx, ny))

    # getting the edges
    edges = []
    graph = [[0 for j in range(n_steiner + n_terminal)] for i in range(n_steiner + n_terminal)]
    for _ in range(n_edge):
        a, b = map(int, input().split())
        length = math.sqrt(((nodes[a][0] - nodes[b][0]) ** 2) + ((nodes[a][1] - nodes[b][1]) ** 2))
        edges.append((a, b, length))
        if b < a:
            graph[a][b] = length
        else:
            graph[b][a] = length


    # ------------------------------------------------------------------------------------------------
    # generating the initial population and evaluating them
    print('Generating the initial population...')
    population = []
    for _ in range(mio):
        chromosome = []
        for i in range(n_steiner):
            coin = random.random()
            if coin < pi:
                chromosome.append(1)
            else:
                chromosome.append(0)
                    
        population.append((chromosome, fitness_calc(nodes, edges, chromosome)))


    # ------------------------------------------------------------------------------------------------
    # evolution loop
    print('Evolution begins...')

    # the lists storing best, average and worst fitness in each generation
    bests = []
    worsts = []
    avgs = []

    for gen_num in range(num_iter):

        # reproduction
        offsprings = []

        for _ in range(lmbd // 2):

            # selecting parents
            parents = fitness_selection(population, 2)

            # bitwise crossover
            c1_chrom, c2_chrom = bitwise_crossover(parents[0][0], parents[1][0])

            # bitwise mutation
            bitwise_mutation(graph, c1_chrom)
            bitwise_mutation(graph, c2_chrom)
            
            # evaluating the children
            child1 = (c1_chrom, fitness_calc(nodes, edges, c1_chrom))
            child2 = (c2_chrom, fitness_calc(nodes, edges, c2_chrom))

            # adding childern to the list
            offsprings.append(child1)
            offsprings.append(child2)


        # next generation population selection
        population = qt_selection(population + offsprings, mio, q)
        
        # finding best, average and worst fitness
        best = max(population, key=lambda x: x[1])[1]
        average = sum([i[1] for i in population]) / len(population)
        worst = min(population, key=lambda x: x[1])[1]

        print('Gen', gen_num, 'Best:', best, 'Average:', average, 'Worst:', worst)

        # adding them to the list
        bests.append(best)
        avgs.append(average)
        worsts.append(worst)


    # ------------------------------------------------------------------------------------------------
    # finding the best solution
    best = None
    max_fitness = 0
    for i in population:
        if i[1] > max_fitness:
            max_fitness = i[1]
            best = i
    chrom = best[0]
    _, weight, all_edges, reverse_mapping = get_mst(nodes, edges, chrom)
    print()
    print('Weight(The Best Steiner Tree):', weight)


    # ------------------------------------------------------------------------------------------------
    # writing the solution in file
    file = open('‫‪steiner_out.txt‬‬', 'w')
    edge_numbers = []
    for e in all_edges:
        edge_numbers.append(edges.index((reverse_mapping[e[0]], reverse_mapping[e[1]], e[2])))
    edge_numbers.sort()
    for en in edge_numbers:
        file.write(str(en))
        file.write("\n")


    # ------------------------------------------------------------------------------------------------
    # plotting best/average/worst fitness over generations
    ran = range(len(bests))
    plt.figure()
    plt.xlabel('Generation Number')
    plt.ylabel('Fitness')
    l1, = plt.plot(ran, bests, label='Best', linewidth=2, color='green')
    l2, = plt.plot(ran, avgs, label='Average', linewidth=2, color='tab:blue')
    l3, = plt.plot(ran, worsts, label='Worst', linewidth=2, color='red')

    plt.legend(handles=[l1, l2, l3], loc='lower right')


    # ------------------------------------------------------------------------------------------------
    # plotting the best steiner tree
    plt.figure()

    # drawing the terminal nodes
    for i in range(n_steiner, n_steiner + n_terminal):
        node = nodes[i]
        plt.scatter(node[0], node[1], color='red')

    # drawing the edges
    for e in all_edges:
        plt.plot(
            [nodes[reverse_mapping[e[0]]][0], nodes[reverse_mapping[e[1]]][0]],
            [nodes[reverse_mapping[e[0]]][1], nodes[reverse_mapping[e[1]]][1]],
            color='tab:blue'
        )
                
    plt.show()


    

if __name__ == "__main__":
    main()