# Define the MLP class
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        dropout=0.2
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
        self.input_dim=input_dim
        self.output_dim=output_dim
        
        self.input_layer=InputLayer(input_dim,n_embd)
        self.layers = nn.ModuleList([FeedFoward(n_embd) for _ in range(num_layers)])

        self.ln_f=nn.LayerNorm(n_embd)
        self.output_layer=nn.Linear(n_embd,self.output_dim)
        
    
    def forward(self, x):
        out=self.input_layer(x)
        for layer in self.layers:
            out = out + layer(out)
            out=layer(out)
        out=self.ln_f(out)
        out=self.output_layer(out)
        return out
    
class LinearRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressor, self).__init__()
        # Define the parameters / weights of the model
        self.linear = nn.Linear( input_dim, output_dim)  # Assuming x and y are single-dimensional

    def forward(self, x):
        return self.linear(x)



class WeightedRandomClassifier(nn.Module):
    def __init__(self, class_counts):
        super().__init__()
        self.class_counts = class_counts

    def forward(self, x):
        # Generate random guesses according to the class weights for each example in the batch
        random_guesses = torch.multinomial(self.class_counts, x.size(0), replacement=True)
        # Convert indices to one-hot encoding
        return F.one_hot(random_guesses, num_classes=len(self.class_counts)).to(torch.float32)

    
class MajorityClassClassifier(nn.Module):
    def __init__(self, majority_class, num_classes):
        super().__init__()
        self.majority_class = majority_class
        self.num_classes = num_classes

    def forward(self, x):
        # Return the majority class for each example in the batch
        majority_class_tensor = torch.full((x.size(0),), self.majority_class, dtype=torch.long)
        # Convert indices to one-hot encoding
        return F.one_hot(majority_class_tensor, num_classes=self.num_classes).to(torch.float32)


def test_baseline_classifiers():
    from matgraphdb.mlcore.metrics import ClassificationMetrics
    # Assuming class_weights for 3 classes
    class_counts= torch.tensor([2.0,4.0,6.0])

    # Assuming the majority class is class index 1
    majority_class = torch.argmax(class_counts)

    # Initialize classifiers
    weighted_random_classifier = WeightedRandomClassifier(class_counts)
    majority_class_classifier = MajorityClassClassifier(majority_class,num_classes=len(class_counts))

    # Dummy input (batch size of 10, features size of 5)
    dummy_input = torch.randn(10, 5)

    # Get predictions
    weighted_random_logits = weighted_random_classifier(dummy_input)
    majority_class_logits = majority_class_classifier(dummy_input)


    weighted_random_probailities = torch.sigmoid(weighted_random_logits)
    majority_class_probailities = torch.sigmoid(majority_class_logits)

    weighted_random_predictions = weighted_random_probailities.argmax(1)
    majority_class_predictions = majority_class_probailities.argmax(1)



    print("Random Predictions:", weighted_random_predictions)
    print("Majority Class Predictions:", majority_class_predictions)


    # Calculate the accuracy of the predictions
    # accuracy = ClassificationMetrics.accuracy(y_true=y_test, y_pred=weighted_random_predictions)
    # print("Accuracy:", accuracy)




if __name__ == "__main__":
    
    test_baseline_classifiers()



    