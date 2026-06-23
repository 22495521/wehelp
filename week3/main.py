import numpy as np



def initNetwork1():
    return [
        {
            "w" : np.array([
                    [0.5,0.6],
                    [0.2,-0.6]
                ]),
            "b" : np.array([0.3,0.25])
        },
        {
            "w" : np.array([0.8,0.4]),
            "b" : np.array(-0.5)
        }
    ]

def initNetwork2():
    return [
        {
            "w" : np.array([
                    [0.5,0.6],
                    [1.5,-0.8]
                ]),
            "b" : np.array([0.3,1.25])
        },
        {
            "w" : np.array([0.6,-0.8]),
            "b" : np.array(0.3)
        },
                {
            "w" : np.array([0.5,-0.4]),
            "b" : np.array([0.2,0.5])
        }
    ]





class Network:
    
    def __init__(self):
        self 

    def forward(self, input,network):
        n = network()

    
        y = None
        for i in range(len(n)):

            if i == 0:
                y = np.dot(input, n[i]["w"]) + n[i]["b"]
            else:
                y = np.dot(y, n[i]["w"]) + n[i]["b"]
        return y
 


input = np.array([1.5,0.5])
input2 = np.array([0,1])

nn=Network()
outputs1 = nn.forward(input,initNetwork1)
outputs2 = nn.forward(input2,initNetwork1)
print(outputs1)
print(outputs2)

input3 = np.array([0.75,1.25])
input4 = np.array([-1,0.5])
outputs3 = nn.forward(input3,initNetwork2)
outputs4 = nn.forward(input4,initNetwork2)
print(outputs3)
print(outputs4)

