import torch
from utils_io import VOC_COLORMAP
from utils_io import VOCSegDataset


def train_loop(trainloader, model, loss, optimizer, device,
               testloader=None, print_every = -1):
    """
    train_loss, train_acc =  train_loop(trainloader, model, loss, optimizer, device, testloader)
    or
    train_loss =  train_loop(trainloader, model, loss, optimizer, device)
    """
    N_train = len(trainloader.sampler) # Number of training samples

    total_loss = 0

    # To compute accuracy, we need to collect the following quantities:
    train_corrects = 0     # Total number of correct predictions
    train_predictions = 0  # Total number of predictions

    # To compute loss every "print_every" steps
    running_loss = 0

    for step, (features, labels) in enumerate(trainloader):
        loss_minibatch, corrects_minibatch, predictions_minibatch = \
            train_minibatch(model, features, labels, loss, optimizer, device)

        total_loss += loss_minibatch
        train_corrects += corrects_minibatch
        train_predictions += predictions_minibatch

        running_loss += loss_minibatch

        # ---------------------------------------------------
        # Additional step to print out the intermediate results
        #  before finishing one epoch.
        if print_every != -1:
            if testloader is not None:
                # we have a validation set.
                if step % print_every == 0:

                    test_loss, test_acc = test_loop(testloader, model, loss, device)

                    print(f"---> Train loss: {running_loss / (print_every * trainloader.batch_size):.6f}.. "
                          f"Test loss: {test_loss:.6f}.. "
                          f"Test accuracy: {test_acc:.3f}")
                    running_loss = 0
            else:
                print(f"---> Train loss: {running_loss / (print_every * trainloader.batch_size):.6f}.. ")
                running_loss = 0
        # ---------------------------------------------------
    train_loss = total_loss / N_train
    train_acc = train_corrects / train_predictions

    return train_loss, train_acc


def test_loop(testloader, model, loss, device):
    """
    Usage:
        test_loss, test_acc = test_loop(testloader, model, criterion, 'cuda')
    """
    N_test = len(testloader.sampler)

    tot_test_loss = 0
    sum_corrects = 0     # Number of correct predictions on the test set
    sum_predictions = 0  # Total number of pixels

    # *** set model to evaluation mode ***
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            log_ps = model(images)
            l = loss(log_ps, labels)
            tot_test_loss += l.sum().sum()

            ps = torch.exp(log_ps)
            sum_corrects += number_of_correct_predictions(ps, labels)
            sum_predictions += labels.numel()

    test_loss = tot_test_loss / N_test
    test_acc = sum_corrects/sum_predictions

    return test_loss.item(), test_acc


def train_minibatch(model, X, y, loss, optimizer, device):
    """
    Train the network with the given minibatch of data

    Usage:
        for step, (features, labels) in enumerate(trainloader):
            loss_minibatch, corrects_minibatch, predictions_minibatch = \
            train_minibatch(model, features, labels, loss, optimizer, device)
        See "train_loop"
    """
    X = X.to(device)
    y = y.to(device)

    # *** set model to train mode ***
    model.train()

    optimizer.zero_grad()

    pred = model(X)
    l = loss(pred, y)   # l.shape returns torch.Size([batch_size])

    l.sum().backward()

    optimizer.step()

    loss_minibatch = l.sum().item()

    corrects_minibatch = number_of_correct_predictions(pred, y)
    predictions_minibatch = float(y.numel())  # Total number of predictions
    # print(train_acc_sum, corrects_minibatch)

    return loss_minibatch, corrects_minibatch, predictions_minibatch


def bilinear_kernel(in_channels, out_channels, kernel_size):
    """
    A convolution layer which converts "in_channels" to "out_channels",
    with a specific "kernel_size", has a weight tensor with shape
    (k*k*out)*in
    In PyTorch's NCHW convention, it should be IN*OUT*K*K.

    Assuming that we have 21 input channels and 21 output channels with
    kernel size = 64, this function returns torch.Size([21, 21, 64, 64]).
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


def number_of_correct_predictions(scores, labels):
    """
    Assuming batch size = 32,
        scores: torch.Size([32, 21, 320, 480])
        labels: torch.Size([32, 320, 480])
    """
    pred = torch.argmax(scores, 1)  # [32, 21, 320, 480] ===> [32, 320, 480]
    num_corrects = (pred == labels).float().sum()
    return num_corrects.item()


def predict(net, img, device):
    """ img: torch tensor with size [3, 320, 480]
    """

    # *** set model to evaluation mode ***
    net.eval()

    X = VOCSegDataset.normalize_input_image(img).unsqueeze(0)  # [3, 320, 480] ==> [1, 3, 320, 480]
    output = net(X.to(device))   # [1, 21, 320, 480]
    pred = output.argmax(dim=1)  # [1, 320, 480]
    # pred = net(X.to(device)).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])  # [1, 320, 480] ==> [320, 480]


def label2image(pred, device):
    # pred: [320, 480]
    colormap = torch.tensor(VOC_COLORMAP, device=device)  # torch.Size([21, 3])
    X = pred.long()  # convert to torch.int64
    return colormap[X, :]  # [320, 480, 3]