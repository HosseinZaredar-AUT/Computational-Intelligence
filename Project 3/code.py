import pandas
import math
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def main():

    # reading data
    dataset = pandas.read_csv('dataset.csv').iloc[:, :].values

    # train and test seperation
    index = math.floor(0.75 * len(dataset))
    train_set = dataset[0: index]
    test_set =  dataset[index:]

    # plotting the test set
    plot_set(test_set, title='Test Set (Actual Values)')

    # training the 1st model
    W, b = train_nn_1(train_set)

    # evaluating the 1st model on the test set
    test_nn_1(W, b, test_set)

    # training the 2nd model
    # W1, W2, b1, b2 = train_nn_2(train_set)

    # evaluating the 2nd model on the test set
    # test_nn_2(W1, W2, b1, b2, test_set)

    plt.show()
    

def plot_set(dataset, title):

    for sample in dataset:
        if sample[2] == 0:
            plt.scatter(sample[0], sample[1], color='r')
        else:
            plt.scatter(sample[0], sample[1], color='tab:blue')
    
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    red_patch = mpatches.Patch(color='r', label='Class 0')
    blue_patch = mpatches.Patch(color='tab:blue', label='Class 1')
    plt.legend(handles=[red_patch, blue_patch], loc='lower right')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)


def cost(actual_output, predicted_output):
    return (actual_output - predicted_output)**2


def train_nn_1(train_set):
    
    # HYPER PARAMETERS
    num_steps = 150
    alpha = 0.4

    # weights and biases initialization
    W = np.zeros((1, 2))
    W[0, 0] = np.random.normal()
    W[0, 1] = np.random.normal()
    b1 = np.zeros((1, 1))

    avg_costs = []

    for step in range(num_steps):
    
        # print(step, end=' ')
        
        # gradients 0 initialization
        grad_W = np.zeros((1, 2))
        grad_b = np.zeros((1, 1))


        sum_cost = 0

        for sample_idx in range(len(train_set)):
            
            # input neurons
            a0 = np.array([
                [train_set[sample_idx][0]],
                [train_set[sample_idx][1]]
            ])
        
            # forward phase
            z1 = W @ a0 + b1
            a1 = sigmoid(z1)
            
            # output labels
            y = train_set[sample_idx][2]

            sum_cost += cost(a1[0, 0], y)

            # backpropagation
            for j in range(2):
                grad_W[0, j] += a0[j, 0] * sigmoid_deriv(z1[0, 0]) * (2 * a1[0, 0] - 2 * y)

            grad_b[0, 0] += sigmoid_deriv(z1[0, 0]) * (2 * a1[0, 0] - 2 * y)
            

        # updating weights and biases
        W -= alpha * (grad_W / len(train_set))
        b1 -= alpha * (grad_b / len(train_set))

        # recording average cost in the current step
        avg_costs.append(sum_cost / len(train_set))
        

    # plotting average cost over epoch
    plt.figure()
    plt.plot(list(range(1, len(avg_costs) + 1)), avg_costs)
    plt.ylabel('Average Cost')
    plt.xlabel('Epoch')


    return W, b1


def test_nn_1(W, b, test_set):

    true_predictions = 0
    predicted = np.zeros((len(test_set), 1))

    for i in range(len(test_set)):

        # input neurons
        a0 = np.array([
            [test_set[i][0]],
            [test_set[i][1]]
        ])

        z = W @ a0 + b
        a1 = sigmoid(z)
        
        actual_value = test_set[i][2]
                
        predicted_value = -1
        if a1[0, 0] > 0.5:
            predicted_value = 1
        else:
            predicted_value = 0
        
        predicted[i, 0] = predicted_value

        if predicted_value == actual_value:
            true_predictions += 1
        
    succes_rate = true_predictions / len(test_set)
    print('Success Rate =', succes_rate)

    # plotting the predicted values
    plt.figure()
    plot_set(np.hstack((test_set[:, (0, 1)], predicted)), 'Test Set (Predicted)')
    

def train_nn_2(train_set):

    # HYPER PARAMETERS
    num_steps = 800
    alpha = 0.8


    # weights and biases initialization

    # layer 1
    W1 = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            W1[i, j] = np.random.normal()

    b1 = np.zeros((2, 1))
    for i in range(2):
        b1[i, 0] = 0

    # layer 2 
    W2 = np.zeros((1, 2))
    for i in range(1):
        for j in range(2):
            W2[i, j] = np.random.normal()

    b2 = np.zeros((1, 1))
    for i in range(1):
        b2[i, 0] = 0

    
    avg_costs = []

    for step in range(num_steps):
        
        # print(step, end=' ')
        
        # gradients 0 initialization
        grad_W2 = np.zeros((1, 2))
        grad_b2 = np.zeros((1, 1))
        
        grad_W1 = np.zeros((2, 2))
        grad_b1 = np.zeros((2, 1))
        
        sum_cost = 0
        
        for sample_idx in range(len(train_set)):
            
            # neurons
            a0 = np.array([
                [train_set[sample_idx][0]],
                [train_set[sample_idx][1]]
            ])
            a1 = np.zeros((2, 1))
            a2 = np.zeros((1, 1))

            # output labels
            y = train_set[sample_idx][2]
        
            # forward phase
            z1 = W1 @ a0 + b1
            a1 = sigmoid(z1)
            z2 = W2 @ a1 + b2
            a2 = sigmoid(z2)


            sum_cost += cost(a2[0, 0], y)
            
        
            # backpropagation
        
            # layer 2
            for j in range(1):
                for k in range(2):
                    grad_W2[j, k] += a1[k, 0] * sigmoid_deriv(z2[j, 0]) * (2 * a2[j, 0] - 2 * y)

            for j in range(1):
                grad_b2[j, 0] += sigmoid_deriv(z2[j, 0]) * (2 * a2[j, 0] - 2 * y)

            grad_a1 = np.zeros((2, 1))
            for k in range(2):
                for j in range(1):
                    grad_a1[k, 0] += W2[j, k] * sigmoid_deriv(z2[j, 0]) * (2 * a2[j, 0] - 2 * y)


            # layer 1
            for j in range(2):
                for k in range(2):
                    grad_W1[j, k] += a0[k, 0] * sigmoid_deriv(z1[j, 0]) * grad_a1[j, 0]

            for j in range(2):
                grad_b1[j, 0] += sigmoid_deriv(z1[j, 0]) * grad_a1[j, 0]
            
        
        # updating weights and biases
        W2 = W2 - alpha * (grad_W2 / len(train_set))
        b2 = b2 - alpha * (grad_b2 / len(train_set))
        W1 = W1 - alpha * (grad_W1 / len(train_set))
        b1 = b1 - alpha * (grad_b1 / len(train_set))
        
        # recording average cost in the current step
        avg_costs.append(sum_cost / len(train_set))

    
    # plotting average cost over epoch
    plt.figure()
    plt.plot(list(range(1, len(avg_costs) + 1)), avg_costs)
    plt.ylabel('Average Cost')
    plt.xlabel('Epoch')

    return W1, W2, b1, b2


def test_nn_2(W1, W2, b1, b2, test_set):

    true_predictions = 0
    predicted = np.zeros((len(test_set), 1))

    for i in range(len(test_set)):

        # input neurons
        a0 = np.array([
            [test_set[i][0]],
            [test_set[i][1]]
        ])

        z1 = W1 @ a0 + b1
        a1 = sigmoid(z1)
        z2 = W2 @ a1 + b2
        a2 = sigmoid(z2)
        
        actual_value = test_set[i][2]
                
        predicted_value = -1
        if a2[0, 0] > 0.5:
            predicted_value = 1
        else:
            predicted_value = 0

        predicted[i, 0] = predicted_value
        
        if predicted_value == actual_value:
            true_predictions += 1
        
    succes_rate = true_predictions / len(test_set)
    print('Success Rate =', succes_rate)

    # plotting the predicted values
    plt.figure()
    plot_set(np.hstack((test_set[:, (0, 1)], predicted)), 'Test Set (Predicted)')


if __name__ == "__main__":
    main()