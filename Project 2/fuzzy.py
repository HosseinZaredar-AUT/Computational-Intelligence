import pandas
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import heapq
import copy

# PARAMETERS
MAX_ITER = 50   # maximum iterations of FCM before it stops
THRESH = 0.001  # the threshold for continuing iterations in FCM
m = 2           # m in FCM
K = 20          # K in FKNN
NUM_X = 50      # number of points in x-axis of the grid of points in FKNN
NUM_Y = 50      # number of points in y-axis of the grid of points in FKNN

# COLORS
colors = ['tab:purple', 'green', 'gray', 'tab:olive', 'r']


def main():

    # reading data
    data_set = pandas.read_csv('sample5.csv', header=None)
    
    # extracting a list of all the points
    X = data_set.iloc[:, :].values

    # entropy over ln(c) for c in range(2, max_c)
    entropy_over_ln(X, max_c=10)

    # cost
    # cost_for_c(X, max_c=10)

    # FCM
    c = 3
    V, U = fcm(X, c, animate=False, plot=True)
    print('Cost =', cost(X, V, U, len(X), c))

    # FKNN
    Y = fknn(X, U, c, plot=False)
    draw_borders(Y, c, new_figure=False)

    plt.show()


def fcm(X, c, animate=False, plot=False):

    if plot or animate:
        plt.figure()

    # number of data points
    N = len(X)

    # data points dimention
    dimention = len(X[0])

    # finding minimum and maximum on each direction
    mins = [min(X[:, n]) for n in range(dimention)]
    maxs = [max(X[:, n]) for n in range(dimention)]

    # plotting data points
    if animate:
        plt.scatter(X[:, 0], X[:, 1], color='tab:blue')

    # randomly intializing centroids
    V = np.zeros((c, dimention))
    for i in range(c):
        for n in range(dimention):
            V[i, n] = (maxs[n] - mins[n]) * random.random() + mins[n]

    # drawing intial centroids
    if animate:
        scatter = plt.scatter(V[:, 0], V[:, 1], marker='x', color='black', s=200)

    # iteration number
    if animate:
        plt.title('0', {'fontsize': 30})

    # prevous iteration centroids
    prev_V = None


    # FCM loop
    for it in range(MAX_ITER):

        # calculating distances
        dist = np.zeros((c, N))
        for i in range(c):
            for k in range(N):
                dist[i, k] = math.sqrt(sum([(V[i, n] - X[k, n])**2 for n in range(dimention)]))

        # calculating u (membership)
        U = np.zeros((c, N))
        for i in range(c):
            for k in range(N):
                U[i, k] = 1 / sum([(dist[i, k]/dist[j, k]) ** (2/(m-1)) for j in range(c)])
            
        # updating centroids
        for i in range(c):
            num = np.zeros((1, dimention))
            denom = 0
            for k in range(N):
                num += U[i, k]**m * X[k]
                denom += U[i, k]**m
            V[i] = num / denom

        # updating new centroids in the plot
        if animate:
            plt.pause(0.5)
            plt.title(str(it + 1), {'fontsize': 30})
            scatter.set_offsets(V)

        # checking if the sum difference from previous centroids is less that THRESH
        if prev_V is not None:
            sum_diff = 0
            for j in range(c):
                sum_diff = math.sqrt(sum([(V[j, n] - prev_V[j, n])**2 for n in range(dimention)]))

            if sum_diff < THRESH:
                if animate:
                    plt.title(f'final iteration: {it + 1}', {'fontsize': 30})
                break

        prev_V = copy.deepcopy(V)


    if not animate and plot:
        plt.scatter(X[:, 0], X[:, 1], color='tab:blue')
        plt.scatter(V[:, 0], V[:, 1], marker='x', color='r', s=200)


    return V, U


def entropy(U, N, c):
    ent = 0
    for i in range(c):
        for k in range(N):
            u_ik = U[i, k]
            ent -= u_ik * math.log(u_ik)
    return ent


def entropy_over_ln(X, max_c):
    cs = range(2, max_c)
    ents = []
    for c in cs:
        _, U = fcm(X, c)
        ents.append(entropy(U, len(X), c) / math.log(c))

    plt.scatter(cs, ents)
    plt.plot(cs, ents)


def cost(X, V, U, N, c):
    cost = 0
    for k in range(N):
        for i in range(c):
            cost += (U[i, k] ** m) * math.sqrt(sum([(V[i, n] - X[k, n])**2 for n in range(len(X[0]))]))
    return cost


def cost_for_c(X, max_c):
    cs = range(1, max_c)
    costs = []
    for c in cs:
        V, U = fcm(X, c)
        costs.append(cost(X, V, U, len(X), c))

    plt.scatter(cs, costs)
    plt.plot(cs, costs)


def fknn(X, U, c, plot=False):

    if plot:
        plt.figure()
    
    # finding minimum and maximum x and y
    min_x = min(X[:, 0])
    max_x = max(X[:, 0])
    min_y = min(X[:, 1])
    max_y = max(X[:, 1])

    delta_x = max_x - min_x
    delta_y = max_y - min_y

    # generating a grid of evenly spaces points
    Y = []
    for i in np.linspace(min_y - 0.05 * delta_y, max_y + 0.05 * delta_y, NUM_Y):
        row = []
        for j in np.linspace(min_x - 0.05 * delta_x, max_x + 0.05 * delta_x, NUM_X):
            row.append([j, i, -1])
        Y.append(row)


    # Algorithm    
    for row in Y:
        for y in row:

            # finding k nearest neighbors
            neighbors = [(math.sqrt((y[0] - x[0])**2 + (y[1] - x[1])**2), i) for i, x in enumerate(X)]
            heapq.heapify(neighbors)
            knn = [heapq.heappop(neighbors) for _ in range(K)]

            max_u = -1
            nearest_cluster = -1

            for i in range(c):
                num = 0
                denom = 0            
                for neighbor in knn:
                    dist_expr = (math.sqrt((y[0] - X[neighbor[1]][0])**2 + (y[1] - X[neighbor[1]][1])**2)) ** ((m-1)/2)
                    num += U[i, neighbor[1]] * dist_expr
                    denom += dist_expr
                
                u_i = num / denom
                if u_i > max_u:
                    max_u = u_i
                    nearest_cluster = i

            y[2] = nearest_cluster

            if plot:
                plt.scatter(y[0], y[1], color=colors[nearest_cluster])
    
    return Y
        

def draw_borders(Y, c, new_figure=False):

    if new_figure:
        plt.figure()

    starts = [[] for _ in range(c)]
    ends = [[] for _ in range(c)]

    for i in range(len(Y) - 1, -1, -1):

        start = [None for _ in range(c)]
        end = [None for _ in range(c)]

        for j in range(0, len(Y[0])):

            prev_c = Y[i][j - 1][2]
            curr_c = Y[i][j][2]

            if j == 0:
                start[curr_c] = Y[i][j][:2]
            if j == len(Y[0]) - 1:
                end[curr_c] = Y[i][j][:2]

            if curr_c != prev_c:
                if start[curr_c] is None:
                    start[curr_c] = Y[i][j][:2]
                end[prev_c] = Y[i][j - 1][:2]

        for k in range(c):
            if start[k] is not None:
                starts[k].append(start[k])
            if end[k] is not None:
                ends[k].append(end[k])


    for i in range(c):    
        plt.plot([p[0] for p in starts[i]], [p[1] for p in starts[i]], color=colors[i])
        plt.plot([p[0] for p in ends[i]], [p[1] for p in ends[i]], color=colors[i])
        if starts[i] != [] and ends[i] != []:
            plt.plot([starts[i][0][0], ends[i][0][0]], [starts[i][0][1], ends[i][0][1]], color=colors[i])
            plt.plot([starts[i][-1][0], ends[i][-1][0]], [starts[i][-1][1], ends[i][-1][1]], color=colors[i])


if __name__ == "__main__":
    main()