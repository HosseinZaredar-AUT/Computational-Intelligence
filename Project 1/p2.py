import math
import random
import numpy as np
import sys
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------
# PARAMETERS
mio = 2000      # Mio
lmbd = 4 * mio  # Lambda
num_iter = 100  # number of iterations before termination
tau = 0.5       # learning rate
q = 100         # q parameter for Q-Tournamament selection
x_min = -512    # minimum x for eggholder input
x_max = 512     # maximum x for eggholder input
infinity = sys.float_info.max  # +infinity


# ------------------------------------------------------------------------------------------------
# ‫‪Eggholder‬‬ function
def eggholder(x1, x2):
    return -(x2 + 47) * math.sin(math.sqrt(abs(x2 + x1/2 + 47))) - x1 * math.sin(math.sqrt(abs(x1 - (x2 + 47))))


# ------------------------------------------------------------------------------------------------
# a function to calculate the fitness value of a chromosome
def fitness_calc(x1, x2):
    dist = abs(-959.6407 - eggholder(x1, x2))
    if dist != 0:
        if x_min <= x1 <= x_max and x_min <= x2 <= x_max:
            f = 1 / dist
        else:
            f = 0
    else:
        f = infinity
    return f


# ------------------------------------------------------------------------------------------------
# Q-Tournament Selection function
def qt_selection(population, n, q):
    selected = []
    for _ in range(n):
        max_fitness = -1
        max_fitness_idx = -1
        for _ in range(q):
            idx = np.random.randint(0, len(population))
            if population[idx][4] > max_fitness:
                max_fitness = population[idx][4]
                max_fitness_idx = idx
        selected.append(population[max_fitness_idx])
    return selected


# ------------------------------------------------------------------------------------------------
# THE MAIN FUNCTION
def main() :

    # ------------------------------------------------------------------------------------------------
    # generating the initial population and evaluating them
    print('Generating the initial population...')
    population = []
    for _ in range(mio):
        x1 = (random.random() - 0.5) * 1024
        x2 = (random.random() - 0.5) * 1024
        s1 = 12
        s2 = 12
        population.append((x1, x2, s1, s2,fitness_calc(x1, x2)))


    # ------------------------------------------------------------------------------------------------
    # evolution loop
    print('Evolution begins...')

    # the lists storing best, average and worst fitness in each generation
    bests = []
    worsts = []
    avgs = []

    for gen_num in range(num_iter):

        # reproduction and offspring evaluation
        offsprings = []
        for _ in range(lmbd):

            # parent selection (random uniform)
            p1 = population[random.randint(0, mio - 1)]
            p2 = population[random.randint(0, mio - 1)]

            # crossover (weighted average)
            fitness_sum = p1[4] + p2[4]
            x1_avg = (p1[4] * p1[0] + p2[4] * p2[0]) / fitness_sum
            x2_avg = (p1[4] * p1[1] + p2[4] * p2[1]) / fitness_sum
            s1_avg = (p1[4] * p1[2] + p2[4] * p2[2]) / fitness_sum
            s2_avg = (p1[4] * p1[3] + p2[4] * p2[3]) / fitness_sum
                
            # mutating sigma
            new_s1 = s1_avg * math.e ** (-tau * np.random.normal())
            new_s2 = s2_avg * math.e ** (-tau * np.random.normal())

            # mutating x
            x1 = x1_avg + new_s1 * np.random.normal()
            x2 = x2_avg + new_s2 * np.random.normal()

            f = fitness_calc(x1, x2)

            offsprings.append((x1, x2, new_s1, new_s2, f))


        # next population selection (with truncation selection method on (mio +) lambda)
        candidates = population + offsprings
        population = qt_selection(candidates, mio, q)

        # finding best, average and worst fitness
        best = max(population, key=lambda x: x[4])[4]
        average = sum([i[4] for i in population]) / mio
        worst = min(population, key=lambda x: x[4])[4]

        print('Gen', gen_num, 'Best:', best, 'Average:', average, 'Worst:', worst)

        # adding them to the list
        bests.append(best)
        avgs.append(average)
        worsts.append(worst)


    # ------------------------------------------------------------------------------------------------
    # finding the best solution
    best = max(population, key=lambda x: x[4])
    print(
        "x1=", best[0],
        'x2=', best[1],
        'f(x1, x2)=', eggholder(best[0], best[1])
    )


    # ------------------------------------------------------------------------------------------------
    # plotting best/average/worst fitness over generations
    ran = range(len(bests))
    plt.figure()
    plt.xlabel('Iteration Number')
    plt.ylabel('Fitness')
    l1, = plt.plot(ran, bests, label='Best', linewidth=2, color='green')
    l2, = plt.plot(ran, avgs, label='Average', linewidth=2, color='tab:blue')
    l3, = plt.plot(ran, worsts, label='Worst', linewidth=2, color='red')

    plt.legend(handles=[l1, l2, l3], loc='lower right')

    plt.show()


if __name__ == "__main__":
    main()