import numpy as np

def relu(x):
    return np.maximum(0, x)

def mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

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


class Network:

    def forward(self, input, network, acFunction):
        y = input

        for index, layer in enumerate(network):
            y = np.dot(y, layer["w"]) + layer["b"]

            if index == 0:
                y = acFunction(y)

        return y


nn = Network()

print("----- Model 1 -----")
input1 = np.array([1.5, 0.5])
outputs1 = nn.forward(input1, initNetwork1(), relu)
print("Outputs", outputs1)
expects1 = np.array([0.8, 1])
print("Total Loss", mse(expects1, outputs1))

input2 = np.array([0, 1])
outputs2 = nn.forward(input2, initNetwork1(), relu)
print("Outputs", outputs2)
expects2 = np.array([0.5, 0.5])
print("Total Loss", mse(expects2, outputs2))

print("----- Model 2 -----")