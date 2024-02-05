import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

data = pd.read_csv('./archive/mnist_test.csv')
data_array = np.array(data)
m, n = data_array.shape

train_data = data_array.T
label_data = train_data[0]
pixel_data = train_data[1: n]

flattened_data = torch.tensor(pixel_data, dtype=torch.float32)

batch_size = 100
data_loader = DataLoader(TensorDataset(flattened_data), batch_size=batch_size, shuffle=True)

class Numbers(nn.Module) :
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x) :
        logits = self.linear_relu_stack(x)
        return logits

model = Numbers()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

num_epochs = 100  # Adjust as needed
for epoch in range(num_epochs):
    for i, batch in enumerate(data_loader):
        inputs = batch[0]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, label_data[:batch_size])
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')