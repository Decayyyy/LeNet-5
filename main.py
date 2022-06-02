import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms      

# -------------- hyper-parameters -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
EPOCHS = 20 
BATCH_SIZE = 32 

# ------------- preprocess: normalization ---------
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.1307, ), (0.3081, ))  
])

# ------------- download and upload MNIST------------------
train_set = datasets.MNIST("data_sets", train = True, download = True, transform = transform)
test_set = datasets.MNIST("data_sets", train = False, download = True, transform = transform)
train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = True)

# ------------- LeNet-5--------------------
class LeNet(nn.Module):
    
    def __init__(self):
        super().__init__()

        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding)
        self.conv1 = nn.Conv2d(1, 6, (5, 5), (1, ), 2) 
        self.conv3 = nn.Conv2d(6, 16, (5, 5)) 
        
        #torch.nn.Linear(in_channels, out_channels)
        self.fc5 = nn.Linear(16*5*5, 120) 
        self.fc6 = nn.Linear(120, 84) 
        self.fc7 = nn.Linear(84, 10) 

    def forward(self, x):
        x = self.conv1(x) 
        x = F.relu(x) 
        x = F.max_pool2d(x, 2, 2) 

        x = self.conv3(x) 
        x = F.relu(x) 
        x = F.max_pool2d(x, 2, 2) 

        x = x.view(x.size(0), -1) 
        x = self.fc5(x) 
        x = F.relu(x) 
        
        x = self.fc6(x) 
        x = F.relu(x) 
        
        x = self.fc7(x) 
        output = F.softmax(x, dim=1) 
        return output


# ---------------- optimizer --------------------------
model = LeNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())


# ---------------- train -----------------------
def train_model(my_model, device, trains_loader, optimizers, epoches):
    
    my_model.train()
    for batch_idx, (data, target) in enumerate(trains_loader):
        data, target = data.to(device), target.to(device)
        optimizers.zero_grad()
        output = my_model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizers.step()
        if batch_idx % 1000 == 0:
            print("Training Epoch: {} \t Loss: {:.5f}".format(epoches, loss.item()))

# ---------------- test ------------------------
def test_model(my_model, device, test_loder):
    
    my_model.eval()  
    correct = 0.0    
    test_loss = 0.0   
    with torch.no_grad():  
        for data, target in test_loder:
            data, target = data.to(device), target.to(device)
            output = my_model(data)
            test_loss += F.cross_entropy(output, target).item()
            predict = output.argmax(dim=1)
            correct += predict.eq(target.view_as(predict)).sum().item()  
        
        avg_loss = test_loss / len(test_loder.dataset)
        correct_ratio = 100 * correct / len(test_loder.dataset)
        print("Average loss: {:.5f}\t Accuracy: {:.5f}\n".format(
            avg_loss, correct_ratio
        ))

# -------------- main ----------------------------
for epoch in range(1, EPOCHS+1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)


