# Get datasets

import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

train_data = datasets.MNIST(root='MNIST-data',                        
                            transform=transforms.ToTensor(),          
                            train=True,                               
                            download=True                             
                           )

test_data = datasets.MNIST(root='MNIST-data',                        
                            transform=transforms.ToTensor(),          
                            train=False,                               
                            download=True                             
                           )

# Split data into train and validation

train_data, valid_data = torch.utils.data.random_split(train_data, [50000, 10000])

# Create dataloaders
batch_size = 200

train_loader = torch.utils.data.DataLoader( 
    train_data, 
    shuffle=True, 
    batch_size=batch_size
)

valid_loader = torch.utils.data.DataLoader(
    valid_data,
    shuffle=True,
    batch_size=batch_size
)

test_loader = torch.utils.data.DataLoader(
    test_data, 
    shuffle=True, 
    batch_size=batch_size
)
# create CNN

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=3), # 1 x 28 x 28
            torch.nn.ReLU(),                        # 10 x 26 x 26
            torch.nn.Conv2d(10, 20, kernel_size=3), # 20 x 24 x 24
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(11520, 10)
        )

    def forward(self, x):
        x = self.cv_layers(x)
        x = self.fc_layer(x)
        x = torch.nn.functional.softmax(x)
        return x

cnn = CNN()

# Define optimizer function
learning_rate = 0.01

optimizer = torch.optim.Adam(
    params=cnn.parameters(),
    lr=learning_rate    
)
# Define loss function
criterion = torch.nn.CrossEntropyLoss()

writer = SummaryWriter(log_dir="runs")  

# Train the model
def train(model, epochs, k_size):
    losses = []
    for epoch in range(epochs):
        for idx, batch in enumerate(train_loader):
            inputs, labels = batch
            pred = model(inputs)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)
            print("Epochs: ", epoch, "batch number: ", idx, "loss: ", loss)
            writer.add_scalar('Loss/Train', loss, epoch*len(train_loader) + idx) 
    
    return(losses) 

if __name__ == '__main__':
    # Piece our CNN flow here
    train(cnn, 10, 3)

    
