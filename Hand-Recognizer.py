# In[1]:


# Import necessary packages
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time


# In[2]:


### Run this cell

from torchvision import datasets, transforms
from torchvision.datasets import MNIST

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
get_ipython().system('tar -zxvf MNIST.tar.gz')

# Download and load the training data
trainset = MNIST(root ='./', download=False, train=True, transform=transform)
valset = MNIST(root = './', download=False, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)


# In[3]:


dataiter = iter(trainloader)
images, labels = dataiter.next()
print(labels)
print(type(images))
print(images.shape)
print(labels.shape)


# In[4]:


plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')


# In[5]:


figure = plt.figure()
num_of_images = 64
for index in range(1, num_of_images+1):
    plt.subplot(8,8 , index)
    plt.axis('off')
    plt.imshow(images[index-1].numpy().squeeze(), cmap='gray_r')


# In[6]:


from torch import nn

# Layer details for the neural network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)
print('Model 0 weight matrix:\n',model[0].weight,'\n')
print('Model 0 weight matrix shape:\n',model[0].weight.shape,'\n')
print('Model 0 bias matrix:\n',model[0].bias,'\n')
print('Model 0 bias matrix shape: \n',model[0].bias.shape,'\n')


# In[7]:


criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
print(images.shape)
images = images.view(images.shape[0], -1)
print('Shape of Images:',images.shape)
print('Shape of Label:',labels.shape)
logps = model(images)
print(logps.shape)
loss = criterion(logps, labels)


# In[8]:


print('Weights Dimension: ', model[0].weight.shape)


# In[9]:


print(model[0])
print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)
model[0].weight.grad.shape


# In[10]:


from torch import optim

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# In[11]:


model.parameters()


# In[12]:


print('Initial weights - ', model[0].weight)

images, labels = next(iter(trainloader))
images.resize_(64, 784)

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Forward pass, then backward pass, then update weights
output = model(images)
loss = criterion(output, labels)
loss.backward()
print('Gradient -', model[0].weight.grad)


# In[13]:


# Take an update step and few the new weights
optimizer.step()
print('Updated weights - ', model[0].weight)


# In[14]:


optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        #This is where the model learns by backpropagating
        loss.backward()

        #And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)


# In[15]:


def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


# In[40]:


images, labels = next(iter(valloader))

print(images[0].shape)

img = images[0].view(1, 784)

for index in range(1, 64+1):


    plt.subplot(8,8 , index)
    plt.axis('off')
    plt.imshow(images[index-1].numpy().squeeze(), cmap='gray_r')


# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)


probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)


# In[19]:


correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))


# In[ ]:
