import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#first prepare data
x_numpy,y_numpy=datasets.make_regression(n_samples=100,n_features=1,noise=20.0,random_state=1)
'''
Parameters
n_samplesint, default=100
The number of samples.

n_featuresint, default=100
The number of features.

n_informativeint, default=10
The number of informative features, i.e., the number of features used to build the linear model used to generate the output.

n_targetsint, default=1
The number of regression targets, i.e., the dimension of the y output vector associated with a sample. By default, the output is a scalar.

biasfloat, default=0.0
The bias term in the underlying linear model.

effective_rankint, default=None
If not None:
The approximate number of singular vectors required to explain most of the input data by linear combinations. Using this kind of singular spectrum in the input allows the generator to reproduce the correlations often observed in practice.

If None:
The input set is well conditioned, centered and gaussian with unit variance.

tail_strengthfloat, default=0.5
The relative importance of the fat noisy tail of the singular values profile if effective_rank is not None. When a float, it should be between 0 and 1.

noisefloat, default=0.0
The standard deviation of the gaussian noise applied to the output.

shufflebool, default=True
Shuffle the samples and the features.

coefbool, default=False
If True, the coefficients of the underlying linear model are returned.

random_stateint, RandomState instance or None, default=None
Determines random number generation for dataset creation. Pass an int for reproducible output across multiple function calls. See Glossary.

Returns
Xndarray of shape (n_samples, n_features)
The input samples.

yndarray of shape (n_samples,) or (n_samples, n_targets)
The output values.

coefndarray of shape (n_features,) or (n_features, n_targets)
The coefficient of the underlying linear model. It is returned only if coef is True.
'''
'''
step 1)design the model
step 2)find the loss and optimizer
step 3)training loop
'''

x=torch.from_numpy(x_numpy.astype(np.float32))
y=torch.from_numpy(y_numpy.astype(np.float32))
#if we want to print the column vector
y=y.view(y.shape[0],1)
print("test",y)

n_samples,n_features=x.shape

#define the model
input_size=n_features
output_size=n_features #we only want to have one value
model=nn.Linear(input_size,output_size)  #one layer

#loss and optimizer
learning_rate=0.01
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

#training for loop
num_of_epochs=100
for epoch in range(num_of_epochs):
    #forward pass and loss
    y_predicted=model(x)
    loss=criterion(y_predicted,y)

    #backward pass
    loss.backward()
    optimizer.step()  #update weights
    optimizer.zero_grad()  #we need to set the gradients to zero each time because it sums up the gradients

    if(epoch+1)%10==0:
        print(f'epoch:{epoch+1},loss={loss.item():.4f}')

#plot
predicted=model(x).detach()    #detach() generates a new tensor where our gradient calculation attribute is false
plt.plot(x_numpy,y_numpy,'ro')
plt.plot(x_numpy,predicted,'b')
plt.show()

