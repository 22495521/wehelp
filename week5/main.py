import numpy as np

def linear(x):
    return x

def relu(x):
    return np.maximum(0, x)

def der_linear(x):
    return 1

def der_relu(x):
    return np.where(x > 0, 1, 0)

def der_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def der_mse(outputs, expects):
    outputs = np.array(outputs, dtype=float)
    expects = np.array(expects, dtype=float)
    return 2 * (outputs - expects) / outputs.size

def der_binary_cross_entropy(outputs, expects):
    outputs = np.array(outputs, dtype=float)
    expects = np.array(expects, dtype=float)
    eps = 1e-12
    outputs = np.clip(outputs, eps, 1 - eps)
    return (outputs - expects) / (outputs * (1 - outputs))
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


def initNetwork1():
    return [
        {
            "w": np.array([
                [0.5, 0.6],
                [0.2, -0.6]
            ]),
            "b": np.array([0.3, 0.25]),
            "activation": relu
        },
        {
            "w": np.array([[0.8],[-0.5]]),
            "b": np.array([0.6]),
            "activation": linear
        },
        {
            "w": np.array([[0.6, -0.3]]),
            "b": np.array([0.4,0.75]),
            "activation": linear
        }
    ]


class Network:

    def forward(self, input, network):
        y = input
        self.cache = []

        for index, layer in enumerate(network):
            layer_input = y
            z = np.dot(y, layer["w"]) + layer["b"]
            y = layer["activation"](z)

            self.cache.append({
                "input": layer_input,
                "z": z,
                "a": y
            })

        return y


class Loss:

    def mse(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean((y_true - y_pred) ** 2)
    
    def binary_cross_entropy(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

nn = Network()

print("----- Model 1 -----")
input1 = np.array([1.5, 0.5])
expects1 = np.array([0.8, 1])
loss_fn = Loss()
learning_rate = 0.01


outputs=nn.forward(input1, initNetwork1())
loss = loss_fn.mse(outputs, expects1)


print(nn.cache)
a = der_mse(outputs, expects1)

b = (1.095 * a[0])

d = 0.6 - learning_rate  * b


print(d)