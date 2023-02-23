"""
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


"""
Problem Statement

Part-of-speech (POS) tagging (or grammatical tagging) is a popular NLP 
process which refers to categorizing words in a text (corpus) based on 
both its definition and its context. 

In NLP, POS tagging helps algorithms understand the grammatical structure
and meaning of the text. 

Common POS tagging methods are as follows:
Viterbi algorithm, Brill tagger, Constraint Grammar, and the Baum-Welch algorithm 
(also known as the forward-backward algorithm). Hidden Markov model and visible 
Markov model taggers can both be implemented using the Viterbi algorithm.
"""


"""
A simplified version of the problem for this tutorial

Consider that we have a small vocabulary set (V). 
The words of the input sentence are in V ($w_{i} \in V$).

T is the tag set containing the followings:
    DET: for determiners
    NN: for nouns
    V: for verbs
We will assign a unique index to each tag. 
    0   DET
    1   NN
    2   V
    
    
Now, we want to run an LSTM over the sentence to tag parts of it.

Let's define the training set:
"""

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
    ("Dog tear that book".split(), ["NN", "V", "DET", "NN"])
]


"""
Assign indices to words and tags
"""
# Assign an index to each UNIQUE word
word_to_ix = {}  # It provides a mapping from each word to its corresponding index.
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:              # if a word is not assigned an index,
            word_to_ix[word] = len(word_to_ix)  # assign a number to it.

print(word_to_ix)

# Assign each tag a unique index.
tag_to_ix = {"DET": 0,
             "NN": 1,
             "V": 2}


"""
Word Embedding

Goal: 
To map words from a vocabulary set into a space which is capable of 
capturing words meaning and relation to other words.

It will provide a vector representation for each word.

Word2Vec is one of the most popular technique to learn word embeddings
using shallow neural network. It was developed by Tomas Mikolov in 2013 
at Google.

"""
# The dimensionality used for embedding is 32 or 64.
# Here, to keep things simple, we will use a very small dimension.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


"""
LSTM Model w/ 
    input dimension = embedding dimension for each word.
    output dimension = hidden dimension
"""


class LSTMTagger(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 vocab_size,
                 tagset_size):

        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim

        # Word embedding model
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # self.lstm = nn.RNN(embedding_dim, hidden_dim)  # Elman RNN

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):

        embeds = self.word_embeddings(sentence)

        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1)) # LNF

        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM,
                   vocab_size=len(word_to_ix),
                   tagset_size=len(tag_to_ix))


"""
Let's compute the output of the UNTRAINED model for an input
"""
print("Output of the untrained model for a given input:")


def prepare_sequence(words, word_to_ix):
    representation = []
    for w in words:
        ix = word_to_ix[w]
        representation.append(ix)
    return torch.tensor(representation, dtype=torch.long)


# input sentence
sent = training_data[0][0]
inputs = prepare_sequence(sent, word_to_ix)
print("inputs:\n", sent, " => ", inputs)

# corresponding tags for its parts
tags = training_data[0][1]
targets = prepare_sequence(tags, tag_to_ix)
print("target:\n", tags, " => ", targets)

print("Set of all tags:\n", tag_to_ix)

# compute model's output, given a tensor contains indices of the words
with torch.no_grad():
    tag_scores = model(inputs)
    print("output:\n",tag_scores)

print("The score vector for the 1st word (%s):"%(sent[0]),  tag_scores[0], " ==argmax==> ", tag_scores[0].argmax())
print("The score vector for the 5th word (%s):"%(sent[4]),  tag_scores[4], " ==argmax==> ", tag_scores[4].argmax())
print("Model's output BEFORE training:\n", tag_scores.argmax(axis=1))

"""
Train the model
"""
print("Training ...")

criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr = 0.1)

for epoch in range(300+1):
    for sent, tags in training_data:
        model.zero_grad()

        inputs = prepare_sequence(sent, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        tag_scores = model(inputs)

        loss = criterion(tag_scores, targets)

        loss.backward()
        optimizer.step()

    if epoch%100 == 0:
        print("epoch = %d, loss = %.3f"%(epoch, loss))


with torch.no_grad():
    tag_scores = model(prepare_sequence(training_data[0][0], word_to_ix))
    print("Model's output AFTER training:\n", tag_scores.argmax(axis=1))