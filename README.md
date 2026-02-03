# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.
<img width="1237" height="875" alt="Screenshot 2026-02-02 094523" src="https://github.com/user-attachments/assets/b166d330-6d48-4bbc-aac1-97998ad7d700" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Sanchita Sandeep

### Register Number: 212224240142

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('/content/Exp 1 csv - Sheet1.csv')
X = dataset1[['Input values']].values
y = dataset1[['Output values']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Name: Sanchita Sandeep
# Register Number: 212224240142
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8) # Input layer: 1 feature in, 8 features out
        self.fc2 = nn.Linear(8, 10) # Hidden layer: 8 features in, 10 features out
        self.fc3 = nn.Linear(10, 1) # Output layer: 10 features in, 1 feature out
        self.relu = nn.ReLU()
        self.history = {'loss':[]}
  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
# Write your code here
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)

# Name:Sanchita Sandeep
# Register Number:212224240142
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss = criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')
loss_df = pd.DataFrame(ai_brain.history)
import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')


```

### Dataset Information
Include screenshot of the generated data
<img width="267" height="417" alt="Screenshot 2026-02-02 094534" src="https://github.com/user-attachments/assets/e7911cea-9cd6-416b-8a50-93cc350a1fb1" />

### OUTPUT

### Training Loss Vs Iteration Plot
Include your plot here
<img width="860" height="568" alt="Screenshot 2026-02-02 094632" src="https://github.com/user-attachments/assets/cff889c6-ecbf-4229-aa02-09f1a7bc1870" />
<img width="412" height="226" alt="Screenshot 2026-02-02 094645" src="https://github.com/user-attachments/assets/ce72df97-e230-44e6-b5f5-14fcc39c9ce6" />

### New Sample Data Prediction
Include your sample input and output here
<img width="309" height="32" alt="Screenshot 2026-02-02 095058" src="https://github.com/user-attachments/assets/1219eeb1-1ace-488d-91d8-e327425b7c00" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
