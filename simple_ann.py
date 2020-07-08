import numpy as np


X = np.array([[1,0,1,0], [1,0,1,1], [0,1,0,1]])
y = np.array([[1],[1],[0]])

print(X)
print(y)

def sigmoid(x):
    return 1 / (1+ np.exp(-x))

def derivativeSigmoid(x):
    return x * (1 - x)

epochs = 10000
alpha = 0.1
inputLayer = X.shape[1]
hiddenLayer = 3
outputLayer = 1


wh = np.random.uniform(size = (inputLayer, hiddenLayer))
bh = np.random.uniform(size = (1, hiddenLayer))
wout = np.random.uniform(size = (hiddenLayer, outputLayer))
bout = np.random.uniform(size = (1, outputLayer))

for i in range(epochs):
    # feed forward
    a1 = np.dot(X, wh) + bh
    z1 = sigmoid(a1)
    a2 = np.dot(z1, wout) + bout
    output = sigmoid(a2)
    
    # Backpropagation
    # error for output layer is simple difference of predicted
    # output and actual output
    error_output_layer = y - output
    # slope is calculated using derivative of activation
    # function used at that layer
    slope_output_layer = derivativeSigmoid(output)
    # to calculate delta of any layer just multiply slope and
    # error of that layer
    delta_output_layer = error_output_layer * slope_output_layer
    
    # to calculate error at hidden layer find dot product of
    # previoud layer delta and weights of that layer
    error_hidden_layer = delta_output_layer.dot(wout.T)
    slope_hidden_layer = derivativeSigmoid(z1)
    delta_hidden_layer = error_hidden_layer * slope_hidden_layer
    
    # updating coefficients
    # to update weights calculate dot product of delta of that
    # layer with hidden layer calculated in feed forward 
    # and multiply with learning rate
    wout += z1.T.dot(delta_output_layer) * alpha
    # to update bias just do the sum of delta of that layer 
    # and multiply with learning rate
    bout += np.sum(delta_output_layer) * alpha
    wh += X.T.dot(delta_hidden_layer) * alpha
    bh += np.sum(delta_hidden_layer) * alpha
    
print(output)    
    
    
    
    
    
    
    
    
    
    
    
    









