import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



# Define the MLP class
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln=nn.LayerNorm(n_embd)

    def forward(self, x):
        return self.net(self.ln(x))
    
class InputLayer(nn.Module):
    def __init__(self,input_dim,n_embd):
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj=nn.Linear(input_dim, n_embd)

    def forward(self, x):
        out=self.flatten(x)
        return self.proj(out)


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers, n_embd):
        super().__init__()
        self.input_layer=InputLayer(input_dim,n_embd)
        self.layers = nn.ModuleList([FeedFoward(n_embd) for _ in range(num_layers)])

        self.ln_f=nn.LayerNorm(n_embd)
        self.output_layer=nn.Linear(n_embd,output_dim)
        

    def forward(self, x):
        out=self.input_layer(x)
        for layer in self.layers:
            out = out + layer(out)
            out=layer(out)
        out=self.ln_f(out)
        out=self.output_layer(out)
        return out
    

def test_step(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train_step(dataloader,model,loss_fn,optimizer,eval_interval=1000):
    # every once in a while evaluate the loss on train and val sets
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # evaluate the loss
        logits = model(X)
        train_loss = loss_fn(logits, y)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch%eval_interval==0:
        #     train_loss,current=train_loss.item(), (batch+1)*len(X)
        #     print(f"loss: {train_loss:>7f}  [{current:>5d}/{size:>5d}]")


def train(train_loader,test_loader,model,loss_fn,optimizer,device,eval_interval=100,max_iters=100):
    for iter in range(max_iters):
        if iter%2==0:
            print(f"Epoch : {iter}")
        train_step(train_loader, model, loss_fn, optimizer,  eval_interval=eval_interval)

        test_step(test_loader, model, loss_fn)

from matgraphdb.mlcore.metrics import ClassificationMetrics
from matgraphdb.mlcore.models import MajorityClassClassifier, WeightedRandomClassifier
    

if __name__ == '__main__':

    
    if torch.cuda.is_available():
        print("GPU is available")
        device = torch.device("cuda")
    else:
        print("GPU is not available")
        device = torch.device("cpu")

    

    dropout=0.2
    learning_rate=0.0001
    batch_size = 64
    num_layers=1
    layer_size=128

    max_iters=10
    eval_interval=10

    # Data preprocessing and loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data/mnist', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Check dataset size
    # print(f"Train dataset size: {len(train_dataset)}")
    # for i in range(10):
    #     print(f"Data shape: {train_dataset[i][0].shape}")
    #     print(f"Data shape: {train_dataset[i][1]}")

    # define the model
    device=  "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
    # model = MultiLayerPerceptron(input_dim=28*28,output_dim=10,num_layers=num_layers,n_embd=layer_size)
    # model.to(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # loss_fn=nn.CrossEntropyLoss()

    # train(train_loader,test_loader,model,optimizer=optimizer,loss_fn=loss_fn,device=device,eval_interval=eval_interval,max_iters=max_iters)


    # Count the number of occurrences of each class in the training dataset
    class_counts = torch.zeros(10)  # There are 10 classes in MNIST (0-9 digits)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    # Calculate the weights for each class
    total_samples = len(train_dataset)
    class_weights = total_samples / class_counts

    # Optionally, normalize weights so that the minimum weight is 1
    class_weights = class_weights / class_weights.min()

    # Assuming the majority class is class index 1
    # find the majority class
    majority_class_index = class_counts.argmax()
    majority_class = class_weights[majority_class_index]

    # Initialize classifiers
    weighted_random_classifier = WeightedRandomClassifier(class_weights)
    weighted_random_classifier.to(device)
    majority_class_classifier = MajorityClassClassifier(majority_class,num_classes=len(class_weights))
    majority_class_classifier.to(device)
    # Initialize classifiers with computed weights and majority class
    weighted_classifier = WeightedRandomClassifier(class_weights)
    majority_classifier = MajorityClassClassifier(majority_class, num_classes=10)

    # Generate dummy inputs and true labels
    dummy_input = torch.randn(10, 5)
    _, y_true = next(iter(train_loader))

    # Get predictions
    weighted_logits = weighted_classifier(dummy_input)
    majority_logits = majority_classifier(dummy_input)

    weighted_probabilities = torch.softmax(weighted_logits, dim=1)
    majority_probabilities = torch.softmax(majority_logits, dim=1)

    weighted_predictions = weighted_probabilities.argmax(1)
    majority_predictions = majority_probabilities.argmax(1)

    # Calculate accuracy
    accuracy_weighted = (weighted_predictions == y_true).float().mean()
    accuracy_majority = (majority_predictions == y_true).float().mean()

    print("Weighted Random Classifier Accuracy:", accuracy_weighted.item())
    print("Majority Class Classifier Accuracy:", accuracy_majority.item())