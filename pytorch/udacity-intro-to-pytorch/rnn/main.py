"""
These are my practices

line = data
line_to_tensor ==> data_to_tensor
"""

from utils import unicode_to_ascii, N_LETTERS
from utils import letter_to_tensor, line_to_tensor
from utils import load_data, random_training_example


import torch
import torch.nn as nn

import random
import matplotlib.pyplot as plt

"""
random seeds
"""
def set_seed(seed):
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

"""
remove accents from names
"""
print(unicode_to_ascii('Núñez'))   # Spanish
print(unicode_to_ascii('Böhler'))  # German

"""
load data
"""
data_dict, all_categories = load_data()
# data_dict = {"Arabic": list of all Arabic names, "Chinese": list of all Chinese names, ...}
# For each name, `unicode_to_ascii` is applied.
# all_categories: a list containing names of languages
print("Number of languages: ",len(all_categories))  # <=> len(data_dict.keys())
print(data_dict['Spanish'][:5])

"""
One-hot encoding for each character.
"""
# Allowed characters are stored in ALL_LETTERS which is a constant string
print("one-hot code for one letter: ", letter_to_tensor('J'))  # [1, 57]

# input tensor shape for RNN should be
#   L*N*H_(input)
#   L: sequence length, N: batch size, H_(input): input size
input_seq = 'Jones'
print(line_to_tensor(input_seq).shape)  # [5, 1, 57]

"""
implement RNN from scratch
"""
class MyRNN(nn.Module):
    """
    USAGE:
        n_hidden = 128
        rnn = RNN(N_LETTERS, n_hidden, n_categories)

        input_tensor = line_to_tensor('Jones')
        hidden_tensor = rnn.init_hidden()+

        output, next_hidden = rnn(input_tensor[0], hidden_tensor)
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        input_size: input vector's length.   (I)
            Input vector is a one-hot encoded representation of a character.
        hidden_size: hidden vector's length  (H)
        output_size: number of classes       (O)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        """
        input_tensor:  1 * input size  (I)
        hidden_tensor: 1 * hidden size (H)
        """
        combined_tensor = torch.cat((input_tensor, hidden_tensor), dim=1)  # 1 * (I + H)

        hidden = self.i2h(combined_tensor)  # 1 * (I + H) ==> 1*H
        output = self.i2o(combined_tensor)  # 1 * (I + H) ==> 1*O
        output = self.softmax(output)       # USE criterion = nn.NLLLoss()
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

# Load data
data_dict, all_categories = load_data()
n_categories = len(all_categories)

# create an RNN object
n_hidden = 128
rnn = MyRNN(N_LETTERS, n_hidden, n_categories)

# Prepare inputs
input_seq = 'Jones'
input_tensor = line_to_tensor(input_seq)  # L*1*input size (H_(input))
hidden_tensor = rnn.init_hidden()         # 1*H_(hidden)

# Prediction
print('Process each character ...')
for i in range(input_tensor.shape[0]):
    # apply RNN on each character of the input sequence
    output_tensor, hidden_tensor = rnn(input_tensor[i], hidden_tensor)
    category_idx = torch.argmax(output_tensor).item()
    print('\t', all_categories[category_idx])

category_idx = torch.argmax(output_tensor).item()
print(all_categories[category_idx])


"""
train MyRNN
"""

set_seed(42)

"""
split data
"""
train_frac = 0.8
train_data_dict, test_data_dict = {}, {}
for k in data_dict.keys():
    random.shuffle(data_dict[k])
    num_items = int(len(data_dict[k]))
    train_size = int(num_items*train_frac)
    print(f"In category {k}, {train_size} out of {num_items} are used for training.")
    train_data_dict[k] = data_dict[k][0:train_size]
    test_data_dict[k] = data_dict[k][train_size+1:]


# criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
n_iters = 20000#100000

print("Training ...")


def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

running_loss = 0
all_loss_values = []  # used for plotting
plot_step = 1000
print_step = 5000

for iter in range(n_iters):
    category, data_item, category_tensor, input_tensor = random_training_example(train_data_dict, all_categories)

    # hidden_tensor = rnn.init_hidden()  # 1*H_(hidden)
    #
    # # Apply RNN on each character of the input sequence
    # for i in range(input_tensor.shape[0]):
    #     output_tensor, hidden_tensor = rnn(input_tensor[i], hidden_tensor)
    #
    # loss = criterion(output_tensor, category_tensor)
    #
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    output_tensor, loss = train(input_tensor, category_tensor)

    running_loss += loss

    if (iter+1) % plot_step == 0:
        all_loss_values.append(running_loss / plot_step)
        running_loss = 0

    if (iter+1) % print_step == 0:
        category_idx = torch.argmax(output_tensor).item()
        predicted_category = all_categories[category_idx]
        msg = "CORRECT" if predicted_category == category else f"WRONG ({category})"
        print(f"{iter + 1}/{n_iters} - loss={loss:.4f} - valid. data = {data_item} ({predicted_category}) {msg}")

plt.figure()
plt.plot(all_loss_values)
plt.show()
plt.savefig('main_train_loss.png')


def predict(input_line):
    # print(f"\nYour input is: {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)

        hidden_tensor = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output_tensor, hidden_tensor_ = rnn(line_tensor[i], hidden_tensor)

        category_idx = torch.argmax(output_tensor).item()
        predicted_category = all_categories[category_idx]
        return predicted_category


"""
compute accuracy per category
"""
acc = {}
total_acc = 0
total_num_items = 0
for k in test_data_dict.keys():
    acc[k] = 0
    num_items = 0
    for data_item in test_data_dict[k]:
        num_items += 1
        predicted_category = predict(data_item)
        if predicted_category == k:
            acc[k] += 1
            total_acc +=1
    acc[k] = acc[k]/num_items
    total_num_items += num_items
    print(f"Category {k}; Test accuracy = {acc[k]}")
print(f'===> Overall accuracy = {total_acc/total_num_items}')


while True:
    sentence = input("Enter a name (or quit):")
    if sentence == "quit":
        break

    print(predict(sentence))