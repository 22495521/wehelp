import numpy as np


class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        s = self(x)
        return s * (1 - s)


class Linear:
    def __call__(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class Layer:
    def __init__(self, w, b, activation):
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float)
        self.activation = activation


class MSELoss:
    def get_loss(self, expects, outputs):
        expects = np.array(expects, dtype=float)
        outputs = np.array(outputs, dtype=float)
        return np.mean((expects - outputs) ** 2)

    def get_output_gradients(self, outputs, expects):
        outputs = np.array(outputs, dtype=float)
        expects = np.array(expects, dtype=float)
        return 2 * (outputs - expects) / outputs.size


class BCELoss:
    def get_loss(self, expects, outputs):
        expects = np.array(expects, dtype=float)
        outputs = np.array(outputs, dtype=float)
        return -np.sum(expects * np.log(outputs) + (1 - expects) * np.log(1 - outputs))

    def get_output_gradients(self, outputs, expects):
        outputs = np.array(outputs, dtype=float)
        expects = np.array(expects, dtype=float)
        eps = 1e-12
        outputs = np.clip(outputs, eps, 1 - eps)
        return (outputs - expects) / (outputs * (1 - outputs))


class Network:

    def __init__(self, layers):
        self.layers = layers
        self.grads = []

    def forward(self, input):
        y = input
        self.cache = []

        for layer in self.layers:
            layer_input = y
            z = np.dot(y, layer.w) + layer.b
            y = layer.activation(z)

            self.cache.append({
                "input": layer_input,
                "z": z,
            })

        return y

    def backward(self, output_gradients):
        self.grads = []
        delta = output_gradients

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            z = self.cache[i]["z"]
            a_prev = self.cache[i]["input"]

            dL_dz = delta * layer.activation.derivative(z)
            dL_dw = np.outer(a_prev, dL_dz)
            dL_db = dL_dz
            delta = np.dot(layer.w, dL_dz)

            self.grads.insert(0, {"dw": dL_dw, "db": dL_db})

    def zero_grad(self, learning_rate):
        for i, layer in enumerate(self.layers):
            layer.w -= learning_rate * self.grads[i]["dw"]
            layer.b -= learning_rate * self.grads[i]["db"]
        self.grads = []


def initNetwork1():
    return [
        Layer(w=[[0.5, 0.6], [0.2, -0.6]], b=[0.3, 0.25], activation=ReLU()),
        Layer(w=[[0.8], [-0.5]],            b=[0.6],        activation=Linear()),
        Layer(w=[[0.6, -0.3]],              b=[0.4, 0.75],  activation=Linear()),
    ]


def initNetwork2():
    return [
        Layer(w=[[0.5, 0.6], [0.2, -0.6]], b=[0.3, 0.25], activation=ReLU()),
        Layer(w=[[0.8], [0.4]],             b=[-0.5],       activation=Sigmoid()),
    ]


# ----- Model 1 -----
print("----- Model 1 -----")
nn1 = Network(initNetwork1())
expects1 = np.array([0.8, 1])
loss_fn1 = MSELoss()
learning_rate1 = 0.01

outputs = nn1.forward(np.array([1.5, 0.5]))
loss = loss_fn1.get_loss(expects1, outputs)
output_gradients = loss_fn1.get_output_gradients(outputs, expects1)
nn1.backward(output_gradients)
nn1.zero_grad(learning_rate1)

print(f"Loss: {loss:.6f} | Output: {np.round(outputs, 4)}")
for i, layer in enumerate(nn1.layers):
    print(f"  Layer {i} w:\n{np.round(layer.w, 4)}")
    print(f"  Layer {i} b: {np.round(layer.b, 4)}")


# ----- Model 2 -----
print("----- Model 2 -----")
nn2 = Network(initNetwork2())
expects2 = np.array([1.0])
loss_fn2 = BCELoss()
learning_rate2 = 0.01

for epoch in range(1000):
    outputs2 = nn2.forward(np.array([1.5, 0.5]))
    loss2 = loss_fn2.get_loss(expects2, outputs2)
    output_gradients2 = loss_fn2.get_output_gradients(outputs2, expects2)
    nn2.backward(output_gradients2)
    nn2.zero_grad(learning_rate2)

print(f"Loss: {loss2:.6f} | Output: {np.round(outputs2, 4)}")
for i, layer in enumerate(nn2.layers):
    print(f"  Layer {i} w:\n{np.round(layer.w, 4)}")
    print(f"  Layer {i} b: {np.round(layer.b, 4)}")
