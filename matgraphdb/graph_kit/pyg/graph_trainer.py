import os
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch_geometric.transforms as T 

from matgraphdb.graph_kit.pyg.callbacks import EarlyStopping, MetricsTacker, Checkpointer
from matgraphdb.graph_kit.pyg.models import WeightedRandomClassifier, MajorityClassClassifier
from matgraphdb.utils import ML_DIR,ML_SCRATCH_RUNS_DIR
from matgraphdb.utils import LOGGER

class Trainer:
    def __init__(self,train_dataset, test_dataset, model, loss_fn, optimizer, device, 
                run_path=ML_SCRATCH_RUNS_DIR, 
                run_name='scratch',
                batch_size=64,
                max_iters=100,
                early_stopping_patience=5,
                eval_interval=4
                ):
        self.train_loader=DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader=DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        self.model=model
        self.loss_fn=loss_fn
        self.optimizer=optimizer
        self.device=device
        self.eval_interval=eval_interval
        self.max_iters=max_iters
        self.early_stopping_patience=early_stopping_patience

        self.run_path=run_path
        self.run_dir=os.path.join(self.run_path,run_name)
        os.makedirs(self.run_dir,exist_ok=True)

        self.es=EarlyStopping(patience = early_stopping_patience)
        self.metrics_tacker=MetricsTacker(save_path=self.run_dir,is_regression=self.is_regression())
        self.checkpointer=Checkpointer(save_path=self.run_dir,verbose=1)

        self.meta_data={}
        self.best_loss=None
        self.best_model=None
    

        # self.model.to(device)

    def initialize_meta_data(self):
        self.meta_data['model_name']=self.model.__class__.__name__
        self.meta_data['loss_fn']=self.loss_fn.__class__.__name__
        self.meta_data['optimizer']=self.optimizer.__class__.__name__
        self.meta_data['eval_interval']=self.eval_interval
        self.meta_data['max_iters']=self.max_iters
        self.meta_data['early_stopping_patience']=self.early_stopping_patience
        self.meta_data['eval_iterval']=self.eval_interval
        self.meta_data['device']=self.device

    def get_class_weights(self,number_of_classes):
        class_counts = torch.zeros(number_of_classes)  # Replace `number_of_classes` with your actual number of classes

        for _,labels in self.train_loader:
            labels=labels.to(self.device)
            for label in labels:
                class_counts[label] += 1


        # Prevent division by zero in case some class is not present at all
        class_weights = 1. / (class_counts + 1e-5)  
        # Normalize weights so that the smallest weight is 1.0
        self.class_weights = class_weights / class_weights.min()
        return class_counts,class_weights

    def is_regression(self):
        return isinstance(self.loss_fn,nn.MSELoss)
    
    def get_num_classes(self):
        if self.is_regression():
            return 1
        else:
            return self.model.output_dim
    
    def calculate_loss(self,logits,y_true):
        if self.is_regression():
            logits = logits[:,0]
            train_loss = self.loss_fn(logits, y_true)
        else:
            train_loss = self.loss_fn(logits, y_true)
        return train_loss
    
    def train_step(self,dataloader):
        """
        Trains the model on the given dataloader.

        Args:
            dataloader (DataLoader): The dataloader to train on.

        Returns:
            float: The average loss per batch on the training data.
        """
        num_batches = len(dataloader)
        batch_train_loss = 0.0
        for i_batch, (X,y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            logits = self.model(X)

            train_loss = self.calculate_loss(logits, y)
            batch_train_loss += train_loss.item()

            # Backpropagation
            self.optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
                
        batch_train_loss = batch_train_loss / num_batches
        return batch_train_loss
    
    def test_step(self,dataloader):
        """
        Tests the model on the given dataloader.

        Args:
            dataloader (DataLoader): The dataloader to test on.

        Returns:
            float: The average loss per batch on the test data.
        """
        num_batches = len(dataloader)
        self.model.eval()
        batch_test_loss = 0.0
        with torch.no_grad():
            for i_batch,(X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)

                logits = self.model(X)
                batch_test_loss = self.calculate_loss(logits, y)
                   
        batch_test_loss /= num_batches
        return batch_test_loss
    
    def predict(self, dataloader, return_probabilities=False):
        """
        Predicts the labels for the given dataloader.

        Args:
            dataloader (DataLoader): The dataloader to predict on.
            return_probabilities (bool, optional): Whether to return the probabilities of the predictions. Defaults to False.

        Returns:
            list: A list of predicted labels.
        """
        total_samples = len(dataloader.dataset)
        batch_size = dataloader.batch_size
        num_classes=self.get_num_classes()

        # Determine the size and type of the predictions tensor
        if return_probabilities and not self.is_regression():
            predictions = torch.zeros(total_samples, num_classes, dtype=torch.float, device=self.device)
            actual=torch.zeros(total_samples, num_classes, dtype=torch.float, device=self.device)
        elif not self.is_regression():
            predictions = torch.zeros(total_samples, dtype=torch.long, device=self.device)
            actual=torch.zeros(total_samples, dtype=torch.long, device=self.device)
        else:
            predictions = torch.zeros(total_samples, dtype=torch.float, device=self.device)
            actual=torch.zeros(total_samples, dtype=torch.float, device=self.device)

        self.model.eval()
        with torch.no_grad():
            sample_idx = 0
            for i_batch,(X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)

                logits = self.model(X)

                if self.is_regression():
                    batch_predictions = logits.squeeze()
                else:
                    probailities = torch.sigmoid(logits)
                    if return_probabilities:
                        batch_predictions=probailities
                    else:
                        batch_predictions=probailities.argmax(1)

                # Calculate the number of predictions to store
                batch_size_actual = X.size(0)
                predictions[sample_idx:sample_idx + batch_size_actual] = batch_predictions
                actual[sample_idx:sample_idx + batch_size_actual] = y
                sample_idx += batch_size_actual
        return actual,predictions
    
    def train(self):
        LOGGER.info(f"___Starting training___")

        if self.model.__class__.__name__ in ['WeightedRandomClassifier','MajorityClassClassifier']:
            batch_test_loss=self.test_step(self.test_loader)
            self.max_iters=1
        # Main training loop
        for iter in range(self.max_iters):
            # Train step
            batch_train_loss=self.train_step(self.train_loader)
            # Test step
            batch_test_loss=self.test_step(self.test_loader)


            if iter%self.eval_interval==0:
                
                train_actual,train_predictions=self.predict(self.train_loader,return_probabilities=False)
                test_actual,test_predictions=self.predict(self.test_loader,return_probabilities=False)
                self.metrics_tacker.calculate_metrics(y_pred=train_predictions,y_true=train_actual,
                                                      batch_loss=batch_train_loss,
                                                      epoch=iter,
                                                      n_features=self.model.input_dim,
                                                      num_classes=self.model.output_dim,
                                                      split='train')
                self.metrics_tacker.calculate_metrics(y_pred=test_predictions,y_true=test_actual,
                                                      batch_loss=batch_test_loss,
                                                      epoch=iter,
                                                      n_features=self.model.input_dim,
                                                      num_classes=self.model.output_dim,
                                                      split='test')
                LOGGER.info(f"Train Loss: {batch_train_loss} | Test Loss: {batch_test_loss}")

            if self.es is not None:
                if self.es(model=self.model, val_loss=batch_test_loss):
                    self.best_loss=batch_test_loss
                    self.best_model=copy.deepcopy(self.model)
                    LOGGER.info("Early stopping")
                    LOGGER.info('_______________________')
                    LOGGER.info(f'Stopping : {iter - self.es.counter}')
                    LOGGER.info(f'Best loss: {self.es.best_loss}')

                    self.checkpointer.save_model(model=self.best_model,epoch=iter)
                    break
                elif iter==self.max_iters-1:
                    self.checkpointer.save_model(model=self.model,epoch=iter)
        LOGGER.info(f"___Ending training___")

        LOGGER.info(f"___Starting Saving metrics___")
        self.metrics_tacker.save_metrics()
        LOGGER.info(f"___Ending saving metrics___")
            

if __name__=='__main__':
    from matgraphdb.mlcore.models import MultiLayerPerceptron

    
    from matgraphdb.mlcore.datasets import MaterialGraphDataset

    graph_dataset=MaterialGraphDataset.ec_element_chemenv(
                                        use_weights=False,
                                        use_node_properties=True,
                                        #,properties=['group','atomic_number']
                                        )
    print(graph_dataset.data)


    rev_edge_types=[]
    edge_types=[]
    for edge_type in graph_dataset.data.edge_types:
        rel_type=edge_type[1]
        if 'rev' in rel_type:
            rev_edge_types.append(edge_type)
        else:
            edge_types.append(edge_type)
    print(edge_types)
    print(rev_edge_types)
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        edge_types=edge_types,
        rev_edge_types=rev_edge_types, 
    )
    train_data, val_data, test_data = transform(graph_dataset.data)
    print("Train Data")
    print("-"*200)
    print(train_data)
    print("Val Data")
    print("-"*200)
    print(val_data)
    print("Test Data")
    print("-"*200)
    print(test_data)


    # device=  "cuda:0" if torch.cuda.is_available() else torch.device("cpu")

    # model=MultiLayerPerceptron(input_dim=28*28,output_dim=10,num_layers=1,n_embd=128)
    # loss_fn=nn.CrossEntropyLoss()

    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    # trainer=Trainer(train_dataset,test_dataset,model,loss_fn,optimizer,device,
    #                 eval_interval=2, 
    #                 max_iters=4,
    #                 batch_size=128,
    #                 early_stopping_patience=5)
    # trainer.train()
    # # print(trainer.get_num_classes())
    # # predictions=trainer.predict(test_loader,return_probabilities=False)
    # # print(predictions.shape)
