import numpy as np


class ReLU:

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, dout):
        return dout * self.mask


class Sigmoid:

    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dout):
        return dout * self.y * (1 - self.y)


class Linear:

    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = x
        return self.y

    def backward(self, dout):
        return dout


class Layer:
    def __init__(self, w, b, activation):
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float)
        self.activation = activation
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        z = np.dot(x, self.w) + self.b
        return self.activation.forward(z)

    def backward(self, dout):
        dL_dz = self.activation.backward(dout)
        # 對W梯度
        self.dw = np.outer(self.x, dL_dz)
        # 對B的梯度
        self.db = dL_dz
        # 對X的梯度往外傳
        return dL_dz @ self.w.T


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

    def forward(self, input):
        y = input
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def backward(self, output_gradients):
        delta = output_gradients
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def zero_grad(self, learning_rate):
        for layer in self.layers:
            layer.w -= learning_rate * layer.dw
            layer.b -= learning_rate * layer.db


def initNetwork1():
    return [
        Layer(w=[[0.5, 0.6], [0.2, -0.6]], b=[0.3, 0.25], activation=ReLU()),
        Layer(w=[[0.8], [-0.5]],            b=[0.6],        activation=Linear()),
        Layer(w=[[0.6, -0.3]],              b=[0.4, 0.75],  activation=Linear()),
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
