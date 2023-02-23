"""
https://github.com/udacity/deep-learning-v2-pytorch/blob/master/sentiment-rnn/Sentiment_RNN_Solution.ipynb

Command line outputs were copied into a .TXT file with the same name.
"""

import numpy as np
from string import punctuation
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

"""
Initial configs
"""
seq_length = 200

train_frac = 0.8
valid_frac = 0.5

batch_size = 50
lr=0.001
epochs = 4  # For this example, in 3 to 4 the validation loss stops decreasing.

print_every = 100
clip=5  # gradient clipping


"""
Load data
"""
# read data from text files
with open('movie_reviews/reviews.txt', 'r') as f:
    reviews = f.read()
with open('movie_reviews/labels.txt', 'r') as f:
    labels = f.read()


"""
Data pre-processing
    convert to lowercase letters
    remove punctuations
    Reviews are separated by newline character ("\n") => store them in a list (reviews_split)
    After removing newline characters, concatenate all reviews for extracting a list of all words. 

Outputs:
    reviews_split: list of reviews
    words: list of words   
"""
reviews = reviews.lower()

all_text = ''.join([c for c in reviews if c not in punctuation])  # punctuation='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

reviews_split = all_text.split('\n')  # "all_text" is "str". => we convert it to a list
print("Number of reviews in the dataset: ", len(reviews_split))

# concatenate all reviews - without punctuations and "\n" characters
all_text = ' '.join(reviews_split)

# create a list of words
words = all_text.split()

print("Some words are:\n", words[:10])


"""
Assign indices to words
    To embed words, we need to pass an integer corresponding to each word to a network. 
    We can define a dictionary that maps words in the vocabulary to integers. Then, we
    can represent each review as a list of integers.

Tokenization
    Tokenization is breaking the raw text into small chunks. 
    Tokenization breaks the raw text into words, sentences called tokens.

Output:
    vocab_to_int
    reviews_ints
    encoded_labels
"""
print("Assign integer indices to words and labels ...")

# Build a dictionary that maps words to integers
counts = Counter(words)  # Count the number of times each word is repeated.
vocab = sorted(counts, key=counts.get, reverse=True)
print(f"The last word is \"{vocab[0]}\" with {counts.get(vocab[0])} repetitions.")
print(f"The last word is \"{vocab[-1]}\" with {counts.get(vocab[-1])} repetitions.")
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
print(f"There are {len(vocab_to_int)} (= vocabulary size) unique words in the vocabulary.")

# Tokenizing each review using the "vocab_to_int"
#   Map each review to a numeric representation based on its words.
reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])

print('Numeric representation of the first review: \n', reviews_ints[0])

# Convert labels to numbers
labels_split = labels.split('\n')
encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])


"""
Outlier removal
    remove reviews with zero length and their corresponding lables

Output: 
    reviews_ints
    encoded_labels
"""
print("Outlier removal ... ")

print('Number of reviews before removing outliers: ', len(reviews_ints))

review_lens = Counter([len(x) for x in reviews_ints])

print(f"# of zero-length reviews = {review_lens[0]}")

max_length = max(review_lens.keys())
print(f"# of reviews with maximum length ({max_length}) = {review_lens[max_length]}")

# get outlier indices
del_idx = [i for i, review in enumerate(reviews_ints) if len(review) == 0]
del_idx.extend([i for i, review in enumerate(reviews_ints) if len(review) == max(review_lens)])
print("Indices of outliers (they will be removed): \n", del_idx)

# remove outliers
reviews_ints = [reviews_ints[i] for i in range(len(reviews_ints)) if i not in del_idx]
encoded_labels = np.array([encoded_labels[i] for i in range(len(encoded_labels)) if i not in del_idx])

print('Number of reviews after removing outliers: ', len(reviews_ints))


"""
Pad sequences
    There are short and long reviews.
    We will PAD (w/ zeros) or TRUNCATE reviews to a specific length.
    Example:
        Given a review like ['best', 'movie', 'ever'] with numeric representation of
        [117, 18, 128], the padded version will be [0, 0, 0, 0, 0, 0, 0, 117, 18, 128]
        if seq_length = 10.

Output:
    features: A 2D array with "len(reviews_ints)" rows and "seq_length" columns.
"""
print("Building feature vectors ... ")


def pad_features(reviews_ints, seq_length):
    """
    Left pad reviews with zeros or truncate them.
    Return a matrix
    """
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    for i, row in enumerate(reviews_ints):
        truncated_row = np.array(row)[:seq_length]
        features[i, -len(truncated_row):] = truncated_row

    return features


features = pad_features(reviews_ints, seq_length=seq_length)

print("First values of padded / truncated feature vectors for the first three reviews: \n", features[:3,:30])


"""
Split data into train, validation, and test sets
"""

split_idx = int(len(features)*train_frac)

train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

split_idx = int(len(remaining_x)*valid_frac)

val_x, test_x = remaining_x[:split_idx], remaining_x[split_idx:]
val_y, test_y = remaining_y[:split_idx], remaining_y[split_idx:]

print("Feature Shapes:")
print(f"Train set: {train_x.shape}\n",
      f"Validation set: {val_x.shape}\n",
      f"Test set: {test_x.shape}"
      )


"""
Create Datasets and DataLoaders
"""
print("Datasets and DataLoaders ...")
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last = True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last = True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last = True)

# DEBUG
# dataiter = iter(train_loader)
# sample_x, sample_y = next(dataiter)
#
# print('Sample input size: ', sample_x.size()) # batch_size, seq_length
# print('Sample input: \n', sample_x)
# print()
# print('Sample label size: ', sample_y.size()) # batch_size
# print('Sample label: \n', sample_y)


"""
Define Network Architecture

    See,
    <img src="./assets/lstm_sentiment_diagram.png>
    
    1. Embedding layer converts tokens (numeric representation of each word) to vectors w/
      a specific size. 
      Note that there are around 74K words, so that one-hot encoding is not even possible.
      
    2. An LSTM is defined by specifying its hidden_state size and number of layers.
    
    3. A fully connected layer is used to map the LSTM's output to the desired output size
    
    4. Sigmoid activation turns all outputs into the range [0, 1].
    
    This network has a many to one architecture.
"""

train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('GPU is available.')
else:
    print('GPU is NOT available.')


class sentimentRNN(nn.Module):
    def __init__(self, vocab_size, output_size,
                 embedding_dim, hidden_dim, n_layers,
                 drop_prob=0.5):
        super(sentimentRNN,self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):

        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out[:,-1,:] # The last timestep's output is wanted.

        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)

        return sig_out, hidden

    def init_hidden(self, batch_size):
        """
        Two tensors with size
            n_layers * batch_size * hidden_dim
        are created and initialized with zeros. These tensors will be
        used for initializing HIDDEN STATE and CELL STATE of the LSTM.

        Output:
            two tensors in a tuple
        """

        # "self.parameters()" is a generator that iterates over the parameters
        # of the model. So, we create a variable to hold parameter values.
        weight = next(self.parameters()).data

        nl = self.n_layers
        bs = batch_size
        hd = self.hidden_dim

        if (train_on_gpu):
            hidden = (weight.new(nl, bs, hd).zero_().cuda(),
                      weight.new(nl, bs, hd).zero_().cuda())
            # "weight.new()" creates a tensor that has the same data type and
            # same device as the parameter.
        else:
            hidden = (weight.new(nl, bs, hd).zero_(),
                      weight.new(nl, bs, hd).zero_())
        return hidden


"""
Instantiate the network.
    Hyper-parameters are as follows:
    
        1. vocab_size: Size of our vocabulary or the range of values for 
          our input, word tokens.
          
        2. output_size: Size of our desired output; the number of class 
          scores we want to output (pos/neg).
          
        3. embedding_dim: Number of columns in the embedding lookup 
          table; size of our embeddings.
          
        4. hidden_dim: Number of units in the hidden layers of our LSTM
         cells. Usually larger is better performance wise. Common 
         values are 128, 256, 512, etc.
         
        5. n_layers: Number of LSTM layers in the network. 
          Typically between 1-3
"""

vocab_size = len(vocab_to_int) + 1  # +1 for the 0 used for padding
output_size = 1  # positive or negative
embedding_dim = 400
hidden_dim = 256
n_layers = 2

net = sentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

if train_on_gpu:
    net.cuda()


"""
Train the model
"""
print("Training ...")

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

net.train()

start_time = time.time()

for e in range(epochs):

    # Way one: (uncomment the corresponding piece of code in the following)
    # The default way of initializing hidden states in the original code.
    h = net.init_hidden(batch_size)

    # Way two: (uncomment the corresponding piece of code in the following)
    # => I think this is correct for this problem, beacuse samples are iid.
    # hinit = net.init_hidden(batch_size)

    # See a discussion here about the stateful and stateless LSTMs
    # [https://stackoverflow.com/questions/39681046/keras-stateful-vs-stateless-lstms]
    #   "In stateless cases, LSTM updates parameters on batch1 and then,
    #   initiate hidden states and cell states (usually all zeros) for batch2"
    #   But, in stateful, LSTM uses previous batch's hidden and cell states as
    #   initializer for the next batch.
    #   If you are training one a dataset with some relation between batche, e.g.,
    #   prices of a stock, then, it makes sense to train in a stateful manner.
    #

    counter = 0

    for inputs, labels in train_loader:
        counter += 1

        if train_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        # Way one:
        # We do not want to backprop through the entire training history (all epochs)
        # We want to backpropagate through all computations done in one epoch.
        h = tuple([each.data for each in h])
        # Note:
        #   Here, we want to make a copy of the "h". If we simply set h2 = h, then
        #   "h2 is h" returns True.
        #   But, if we set, h2 = tuple([each for each in h). Then, they will be
        #   completely different instances.
        #   Another way, is to use "from copy import deepcopy". Then,
        #   h2 = deepcopy(h).

        # Way two
        # h = hinit
        # During training, each batch is independent of other batches. Having
        # a stateful LSTM means that you will need to reset the hidden state
        # in between batches yourself if you do want independent batches.
        # The default initial hidden state in Tensorflow is all zeros.
        # [https://adgefficiency.com/tf2-lstm-hidden/]
        #

        net.zero_grad()

        output, h = net(inputs, h)

        loss = criterion(output.squeeze(), labels.float())

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem
        # in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)

        optimizer.step()

        if counter % print_every == 0:
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:
                val_h = tuple([each.data for each in val_h])
                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

print("Elapsed time for TRAINING = %.1f"%(time.time() - start_time))

"""
The Performance on the test set
"""
print("Inference on test set ... ")

test_losses = []
num_correct = 0

h = net.init_hidden(batch_size)

net.eval()

start_time = time.time()

for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    if train_on_gpu:
        inputs, labels = inputs.cuda(), labels.cuda()
    output, h = net(inputs, h)

    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Elapsed time for TEST = %.1f"%(time.time() - start_time))

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))


"""
Inference on a arbitrary inputs
"""
print("Inference on arbitrary inputs ... ")

test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. ' \
                  'This movie had bad acting and the dialogue was slow.'
test_review_pos = 'This movie had the best acting and the dialogue was so good. I loved it.'

def tokenize_review(test_review):
    test_review = test_review.lower() # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int.get(word, 0) for word in test_words])

    return test_ints


def predict(net, test_review, sequence_length=200):
    net.eval()

    # tokenize review
    test_ints = tokenize_review(test_review)

    # pad tokenized sequence
    seq_length = sequence_length
    features = pad_features(test_ints, seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)

    if (train_on_gpu):
        feature_tensor = feature_tensor.cuda()

    # get the output from the model
    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    # print custom response
    if (pred.item() == 1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")


start_time = time.time()

test_ints = tokenize_review(test_review_neg)
features = pad_features(test_ints, seq_length)
feature_tensor = torch.from_numpy(features)
print("input size:", feature_tensor.size())
predict(net, test_review_neg, seq_length)

print("Elapsed time for one test example = %.1f"%(time.time() - start_time))