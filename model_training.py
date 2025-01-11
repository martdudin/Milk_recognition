from matplotlib import pyplot as plt
import torch
import torch.utils.data.dataloader
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import lr_scheduler
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(pretrained=True)

# Freeze all params
for params in model.parameters():
    params.requires_grad_ = False

# Add a final layer
nr_filters = model.fc.in_features # number of input features of last layer
model.fc = nn.Linear(nr_filters, 1)

model = model.to(device)

# Loss
loss_fn = BCEWithLogitsLoss() # Binary cross entropy with sigmoid
losses = []
val_losses = []

epoch_train_losses = []
epoch_test_losses = []

n_epochs = 10
early_stopping_tolerance = 3
early_stopping_threshold = 0.03

# Optimizer
optimizer = torch.optim.Adam(model.fc.parameters())

traindir = 'C:/Users/martd/Milk_recognition/data/training'
testdir = "C:/Users/martd/Milk_recognition/data/validate"
# traindir = os.path.join("data", "training")
# testdir = os.path.join("data", "validation")

# Transformations
train_transforms = transforms.Compose([transforms.Resize((32,32)),
                                       transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225],
                                       )
                                       ])

test_transforms = transforms.Compose([transforms.Resize((32,32)),
                                       transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225],
                                       )
                                       ])

# Datasets
train_data = datasets.ImageFolder(traindir, transform=train_transforms)
test_data = datasets.ImageFolder(testdir, transform=test_transforms)

# Dataloader
trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=16)
testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=16)

def make_train_step(model, optimizer, loss_fn):
    def train_step(x, y):
        # Make a prediction
        yhat = model(x)
        # Enter train mode
        model.train()
        # Compute loss
        loss = loss_fn(yhat, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return loss
    return train_step

# Train step
train_step = make_train_step(model, optimizer, loss_fn)

for epoch in range(n_epochs):
    epoch_loss = 0
    for i, data in tqdm(enumerate(trainloader), total = len(trainloader)):
        x_batch, y_batch = data
        x_batch = x_batch.to(device)
        y_batch = y_batch.unsqueeze(1).float()
        y_batch = y_batch.to(device)
        
        loss = train_step(x_batch, y_batch)
        epoch_loss += loss/len(trainloader)
        losses.append(loss)
    
    epoch_train_losses.append(epoch_loss)
    print(f'\nEpoch: {epoch + 1}, train loss: {epoch_loss}')
    
    with torch.no_grad():
        cum_loss = 0
        for x_batch, y_batch in testloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.unsqueeze(1).float()
            y_batch = y_batch.to(device)
            
            model.eval()
            
            yhat = model(x_batch)
            val_loss = loss_fn(yhat, y_batch)
            cum_loss += loss/len(testloader)
            val_losses.append(val_loss.item())
        
        epoch_test_losses.append(cum_loss)
        print(f'Epoch: {epoch + 1}, val loss: {cum_loss}')
        
        best_loss = min(epoch_test_losses)
        
        if cum_loss <= best_loss:
            best_model_wts = model.state_dict()
        
        early_stopping_counter = 0
        if cum_loss > best_loss:
            early_stopping_counter += 1
        
        if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
            print("\nTerminating: early stopping")
            break

model.load_state_dict(best_model_wts)

def inference(test_data):
    idx = torch.randint(1, len(test_data), (1,))
    sample = torch.unsqueeze(test_data[idx][0], dim=0).to(device)
    
    if torch.sigmoid(model(sample)) < 0.5:
        print("Prediction: apple")
    else:
        print("Prediction: bus")
    
    plt.imshow(test_data[idx][0].permute(1, 2, 0))
    plt.show()

inference(test_data)
inference(test_data)
inference(test_data)
inference(test_data)
inference(test_data)