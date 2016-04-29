# Note: The perceptron requires linearly seperable data.
# This will be my perceptron algorithm

# h(x) = sign(transpose(w)*x)
# input weights and vector

### Algorithm
# Randomly assign w
# pick misclassified point (how to efficiently find?)
# update weight vecotr w = w + yn*Xn
# repeat until all are classified
# return weights?