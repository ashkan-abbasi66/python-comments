"""
srun -u -p gpu --gres gpu:1 --mem-per-cpu=16GB --time=0-20 python train_fcn.py
"""

from utils_io import *
from utils_model import *
import torch.nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time


# """
# [Optional] Set the priority of the process to high in Windows:
# """
# Found at
# https://stackoverflow.com/questions/1023038/change-process-priority-in-python-cross-platform
# import win32api, win32process, win32con
#
# pid = win32api.GetCurrentProcessId()
# handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
# win32process.SetPriorityClass(handle, win32process.REALTIME_PRIORITY_CLASS)


"""
Get Dataloaders
"""
# data_dir = r'../../../../../PYTHON_CODES/DATASETS'
data_dir = r'../DATASETS'
voc_dir = os.path.join(data_dir, 'VOCdevkit/VOC2012')
print("\nINFO: dataset directory:\n", voc_dir)

batch_size = 32  # 64
crop_size = (320, 480)
trainloader, testloader = get_voc_dataloaders(voc_dir, batch_size, crop_size)
N_train = len(trainloader.sampler)  # Number of training samples


"""
Build model
"""
pretrained_net = torchvision.models.resnet18(pretrained=True)

print("\nINFO: The whole network architecture:\n", pretrained_net)

wanted_layers = list(pretrained_net.children())[:-2]
print("\nINFO: number of layers to keep from the whole network: ", len(wanted_layers))
net = torch.nn.Sequential(*wanted_layers)

num_classes = 21
net.add_module('final_conv', torch.nn.Conv2d(512, num_classes, kernel_size=1))

st = 32
net.add_module('transpose_conv', torch.nn.ConvTranspose2d(num_classes, num_classes,
                                                          stride=st,
                                                          kernel_size=2*st,
                                                          padding=st//2))
W = bilinear_kernel(num_classes, num_classes, 2*st)
net.transpose_conv.weight.data.copy_(W)


"""
Set optimizer and loss function
"""

if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('\nINFO: device: {}'.format(my_device))

def loss(predictions, labels):
    """
    Assume that batch size = 32 and "num_classes = 21". Then,
    predictions = a tensor with shape [32, 21, 320, 480]  (NCHW); One-hot encoded predictions
    targets shape = a tensor with shape [32, 320, 480]     (NHW); Numeric labels
    Then,
    F.cross_entropy(predictions, labels, reduction='none') has shape [32, 320, 480]
    Therefore, we use "mean" function on it two times to reduce it to 32 numbers
    (torch.Size([32])).

    Each of these numbers will represent the prediction error for one input image
    in the batch.
    """
    return F.cross_entropy(predictions, labels, reduction='none').mean(1).mean(1)
    # Alternatively:
    # return torch.nn.CrossEntropyLoss(reduction='none')(predictions, labels).mean(1).mean(1)

lr = 0.001
wd = 1e-3

optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
net.to(my_device)

# To avoid CUDA out of memory error
torch.cuda.empty_cache()


"""
Training
"""
start_time = time.time()

num_epochs = 100#10

# Tp keep track of the statistics at the completion of epoch
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

print_every = len(trainloader)//4   # E.g., with 34 batches, it prints training statistics when 8 batches are presented.
valid_loss_min = 10000.0

for e in range(num_epochs):
    print(f'Epoch: {e + 1}/{num_epochs}')
    train_loss, train_acc = train_loop(trainloader, net, loss, optimizer,
                                       my_device,
                                       testloader,
                                       print_every)

    test_loss, test_acc = test_loop(testloader, net, loss, my_device)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    if test_loss <= valid_loss_min:
        msg = '\nINFO: validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'
        print(msg.format(valid_loss_min, test_loss))
        torch.save(net.state_dict(), 'trained_fcn_at%d_val_acc_%.3f.pt'%(e,test_acc))
        valid_loss_min = test_loss

    print("End of Epoch: {}/{}.. ".format(e + 1, num_epochs),
          "Training Loss: {:.3f}.. ".format(train_loss),
          "Test Loss: {:.3f}.. ".format(test_loss),
          "Training Accuracy: {:.3f}..".format(train_acc),
          "Test Accuracy: {:.3f}".format(test_acc))

training_time = time.time() - start_time
msg = "\nINFO: Elapsed training time: %0.2f\n" % training_time
print(msg)

"""
Plots
"""

plt.figure()
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.savefig('plot_train_validation_losses.png')
plt.close()
# plt.show()

plt.figure()
plt.plot(train_accuracies, label='Training acc')
plt.plot(test_accuracies, label='Validation acc')
plt.legend(frameon=False)
plt.savefig('plot_train_validation_accuracies.png')
plt.close()
# plt.show()


# """
# Predictions on some test images
# """
# test_features, test_labels = read_voc_images(voc_dir, is_train=False)
#
# checkpoint = torch.load(checkpoint_filename)
# net.load_state_dict(checkpoint)
#
# image_lists = []
# n_rows = 4
# for i in range(n_rows):
#     crop_rect = (0, 0, 320, 480)
#
#     input_image = crop_image(test_features[i], crop_rect)   # [3, 320, 480]
#     output = predict(net, input_image, my_device)           # [320, 480]
#     pred = label2image(output, my_device)                   # [320, 480, 3]
#     label_image = crop_image(test_labels[i], crop_rect)     # [3, 320, 480]
#
#     image_lists.append([input_image.permute(1, 2, 0), pred.cpu(), label_image.permute(1, 2, 0)])
#
# plot_images_in_rows(image_lists, len(image_lists), save_path='')
# plot_images_in_rows(image_lists, len(image_lists), save_path='plot_some_test_images')
#
# temp = input('Press Enter to End the Program. ')
