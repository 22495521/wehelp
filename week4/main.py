import numpy as np

def linear(x):
    return x

def relu(x):
    return np.maximum(0, x)

def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initNetwork1():
    return [
        {
            "w": np.array([
                [0.5, 0.6],
                [0.2, -0.6]
            ]),
            "b": np.array([0.3, 0.25])
        },
        {
            "w": np.array([[0.8, 0.4], [-0.5, 0.5]]),
            "b": np.array([0.6, -0.25])
        }
    ]

def initNetwork2():
    return [
        {
            "w": np.array([
                [0.5, 0.6],
                [0.2, -0.6]
            ]),
            "b": np.array([0.3, 0.25])
        },
        {
            "w": np.array([0.8, 0.4]),
            "b": np.array([-0.5])
        }
    ]


class Network:

    def forward(self, input, network, acFunction,lastAcFunction):
        y = input

        for index, layer in enumerate(network):
            y = np.dot(y, layer["w"]) + layer["b"]

            if index == 0:
                y = acFunction(y)

        y = lastAcFunction(y)   
        return y


nn = Network()

print("----- Model 1 -----")
input1 = np.array([1.5, 0.5])
outputs1 = nn.forward(input1, initNetwork1(), relu, linear)
print("Outputs", outputs1)
expects1 = np.array([0.8, 1])
print("Total Loss", mse(expects1, outputs1))

input2 = np.array([0, 1])
outputs2 = nn.forward(input2, initNetwork1(), relu, linear)
print("Outputs", outputs2)
expects2 = np.array([0.5, 0.5])
print("Total Loss", mse(expects2, outputs2))

print("----- Model 2 -----")


input3 = np.array([0.75,1.25])
outputs3 = nn.forward(input3, initNetwork2(), relu, sigmoid)
print("Outputs", outputs3)
expects3 = np.array([1])
print("Total Loss", binary_cross_entropy(expects3, outputs3))

input4 = np.array([-1, 0.5])
outputs4 = nn.forward(input4, initNetwork2(), relu, sigmoid)
print("Outputs", outputs4)
expects4 = np.array([0])
print("Total Loss", binary_cross_entropy(expects4, outputs4))

  
print("----- Model 3 -----")