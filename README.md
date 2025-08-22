# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:Harisha.S
 
### Register Number:212224230087
```python import torch
import torch.nn as nn #neural
import matplotlib.pyplot as plt #plot

# genrate input and output
torch.manual_seed(71) #reproduce
X=torch.linspace(1,50,50).reshape(-1,1)
e=torch.randint(-8,9,size=(50,1),dtype=torch.float)
y= 2*X+1+e

#plot original data
plt.scatter(X,y,color='purple')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Genrated data for linear regression")
plt.show()

#linear modal
class Model(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.linear=nn.Linear(in_features,out_features) #(1,1)input and outputs

    def forward(self,x):
      return self.linear(x)
#Inilaize the model
torch.manual_seed(59)
model= Model(1,1)
#print inital weight and bias
initial_weight=model.linear.weight.item()
initial_bias=model.linear.bias.item()
print("\nName: ")
print("Register No:")
print(f"Initial weight: {initial_weight:.8f},Initial bias: {initial_bias:.8f}\n")

# Loss fun and Optimizer
loss_function=nn.MSELoss()
Optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
#Train model
epochs=100 #how many times
losses=[]
for epoch in range(1,epochs+1): #loop
  Optimizer.zero_grad() #reset to zero before starts new one
  y_pred=model(X) #forward pass,predict Y
  loss=loss_function(y_pred,y) #compute loss btw actual and predict
  losses.append(loss.item()) #store loss value for track#PLOT ORIGINAL DATA AND BEST FIT LINE
plt.scatter(X,y,label="Orginal Data")
plt.plot(x1,y1,'r',label="Best fit line")
plt.xlabel('X')
plt.ylabel('y')
plt.title("""Linear Regression
          Trained Model:Best fit line""")
plt.legend()
plt.show()


  loss.backward()
  Optimizer.step()

  # print loss,weight,bias for every EPOCH
print(f'epoch: {epoch:2} loss: {loss.item():10.8f}'
      f'weight: {model.linear.weight.item():10.8f}'
      f'bias: {model.linear.bias.item():10.8f}')
from typing_extensions import final
#FINAL WEIGHT AND BIAS
final_weight=model.linear.weight.item()
final_bias=model.linear.bias.item()
print('\nName:HARISHA.S')
print("Register No:212224230087")
print(f"Final weight: {final_weight:.8f},Final bias: {final_bias:.8f}\n")

# BEST FIT LINE CALCULATION
x1=torch.tensor([X.min().item(),X.max().item()])
y1= x1 * final_weight + final_bias

#Prediction for X=120
x_new =torch.tensor([[120.0]])
y_new_pred=model(x_new).item()
print("\nName:HARISHA.S")
print("Register No:212224230087")
print(f"\nPrediction for x=120: {y_new_pred:.8f}")
```


```python
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        #Include your code here



# Initialize the Model, Loss Function, and Optimizer

```

### Dataset Information

<img width="720" height="575" alt="image" src="https://github.com/user-attachments/assets/c662f026-442a-49ec-8b3f-6b392ae3a473" />

### OUTPUT
     Training Loss Vs Iteration Plot
<img width="722" height="545" alt="image" src="https://github.com/user-attachments/assets/852c53c1-a602-4767-acb3-feecd994b9bd" />

    Best Fit line plot
<img width="712" height="577" alt="image" src="https://github.com/user-attachments/assets/6b86625b-9c89-4a39-becb-3c6d64789fbb" />


### New Sample Data Prediction
<img width="311" height="83" alt="image" src="https://github.com/user-attachments/assets/ab55b907-0268-479d-958d-d21f5ca6cca6" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
