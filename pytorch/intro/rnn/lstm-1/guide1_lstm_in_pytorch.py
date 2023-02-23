"""
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

"""
Define an LSTM w/ 
    input dimension = 3  => feature vector size is 1*3, e.g., torch.randn(1, 3)
                            Feature vector (element) size is fixed. 
                            A sequence is formed by concatenating feature vectors.
    output dimension = 3
"""

feature_size = 3 # or input_size (H_{in}): The number of expected features in the input `x`
output_size = 3  # or hidden_size (H_{cell}): The number of features in the hidden state `h`
# Note: it is possible to have an output size which differ from hidden_size. => proj_size > 0
num_layers = 1   # >1 creates a stacked LSTM
lstm = nn.LSTM(feature_size, output_size, num_layers)

# The output size will determine the size of hidden vector.
# E.g., output size = 2 ([1,1,2]) ==> hidden vector will also be [1,1,2]
# Recall that h_t = o_t * tanh(c_t), where * indicates element-wise multiplication.
#                                    c_t is the cell state at time t.
# It seems that at t=0, h_0 = o_0.

# The sequence of hidden state has a LENGTH equal to 2. (why?)

# Let's define an input sequence using 1*3 vectors as its elements (or feature vectors).
sequence_length = 5
inputs = [torch.randn(1, feature_size) for _ in range(sequence_length)]  # make a sequence of length 5

# initialize the hidden state.
hidden0 = (torch.randn(1, 1, output_size),
          torch.randn(1, 1, output_size))

"""
Compute the output by processing the input sequence one by one => 
    Process one feature vector at a time
"""
print("Compute the LSTM's output one step at a time:")
hidden_ = hidden0
for i, element in enumerate(inputs):
    # After each step, hidden contains the hidden state.
    input_vec = element.view(1, 1, -1)
    out, hidden = lstm(input_vec, hidden_)
    hidden_ = hidden

    print("step = %d" % i)
    print("input[%d]:" % i)
    print(input_vec)
    print("hidden[%d]:"%i)
    print(hidden)
    print("output[%d]="%i, out)
    print("")


"""
We want to use the entire sequence as input to the LSTM

LSTM's input shape in PyTorch:
    The first axis is the sequence itself
    The second axis indexes instances in the mini-batch
    The axis third indexes elements of the input.
    => can be summarized as "LNF" / "TNF" 
    L: Length; T: number of time steps
    F: feature vector size or H_{in}.
"""
print("Compute the LSTM's output at once:")
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
print("Input tensor shape: ", inputs.shape)
# The first value returned by LSTM is all of the hidden states throughout the sequence.
# The second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension

out, last_hidden = lstm(inputs, hidden0)

print("Last hidden state:\n", last_hidden)
print("The output:\n", out)


"""
stacked LSTM
"""
print("Define a stacked LSTM:")
feature_size = 10
output_size = 20
num_layers = 2

rnn = nn.LSTM(feature_size, output_size, num_layers)

sequence_length = 5
num_samples = 5
input = torch.randn(sequence_length, num_samples, feature_size)

h0 = torch.randn(num_layers, num_samples, output_size)
c0 = torch.randn(num_layers, num_samples, output_size)
output, (hn, cn) = rnn(input, (h0, c0))
print(output)