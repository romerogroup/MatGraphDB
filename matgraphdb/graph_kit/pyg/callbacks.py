import os
import copy
import json

import torch

from matgraphdb.graph_kit.pyg.metrics import RegressionMetrics, ClassificationMetrics



class EarlyStopping():
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        """The early stopping callback

        Parameters
        ----------
        patience : int, optional
            The number of epochs to wait to see improvement on loss, by default 5
        min_delta : float, optional
            The difference theshold for determining if a result is better, by default 0
        restore_best_weights : bool, optional
            Boolean to restore weights, by default True
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.best_mape_loss = None
        self.best_mae_loss = None
        self.counter = 0
        self.status = 0

    def __call__(self, model, test_loss:float, mae_loss):
        """The class calling method

        Parameters
        ----------
        model : torch.nn.Module
            The pytorch model
        test_loss : float
            The validation loss
        mape_val_loss : float
            The map_val_loss

        Returns
        -------
        _type_
            _description_
        """
        if self.best_loss == None:
            self.best_loss = test_loss
            self.best_mae_loss = mae_loss

            self.best_model = copy.deepcopy(model)
        elif self.best_loss - test_loss > self.min_delta:
            self.best_loss = test_loss
            self.best_mae_loss = mae_loss
            
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - test_loss < self.min_delta:
            self.counter +=1
            if self.counter >= self.patience:
                self.status = f'Stopped on {self.counter}'
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                return True
            self.status = f"{self.counter}/{self.patience}"
            # print( self.status )
        return False


class MetricsTacker():
    def __init__(self,save_path, is_regression=False):
        self.metrics_dict={}
        self.n_metrics=0
        self.is_regression=is_regression

        self.metrics_dict['train']={}
        self.metrics_dict['test']={}
        self.save_path=save_path

        if self.is_regression:
            self.get_regression_metrics(split='train')
            self.get_regression_metrics(split='test')
        else:
            self.get_classification_metrics(split='train')
            self.get_classification_metrics(split='test')

    def get_regression_metrics(self,split):
        self.metrics_dict[split]['mse']=[]
        self.metrics_dict[split]['mae']=[]
        self.metrics_dict[split]['rmse']=[]
        self.metrics_dict[split]['msle']=[]
        self.metrics_dict[split]['r2']=[]
        self.metrics_dict[split]['adjusted_r2']=[]
        self.metrics_dict[split]['explained_variance_score']=[]
        self.metrics_dict[split]['mape']=[]
        self.metrics_dict[split]['huber_loss']=[]
        self.metrics_dict[split]['batch_loss']=[]
        self.metrics_dict[split]['epoch']=[]

        self.metric_names=list(self.metrics_dict.keys())

    def get_classification_metrics(self,split):
        self.metrics_dict[split]['accuracy']=[]
        self.metrics_dict[split]['class_weights']=[]
        self.metrics_dict[split]['confusion_matrix']=[]
        self.metrics_dict[split]['batch_loss']=[]
        self.metrics_dict[split]['epoch']=[]

        self.metric_names=list(self.metrics_dict.keys())

    def calculate_metrics(self,y_pred,y_true, batch_loss, epoch, n_features,num_classes, split):
        """
        Calculates the metrics for a given set of predictions and true values.

        Args:
            y_pred (torch.Tensor): The predicted values.
            y_true (torch.Tensor): The true values.
            batch_loss (torch.Tensor): The trained loss.
            epoch (int): The current epoch.
            n_features (int): The number of features.
            split (str): The split for which the metrics are being calculated.

        Returns:
            None
        """
        if self.is_regression:
            self.metrics_dict[split]['mse'].append(RegressionMetrics.mean_squared_error(y_pred, y_true))
            self.metrics_dict[split]['mae'].append(RegressionMetrics.mean_absolute_error(y_pred, y_true))
            self.metrics_dict[split]['rmse'].append(RegressionMetrics.root_mean_squared_error(y_pred, y_true))
            self.metrics_dict[split]['mape'].append(RegressionMetrics.mean_absolute_percentage_error(y_pred, y_true))
            self.metrics_dict[split]['msle'].append(RegressionMetrics.mean_squared_logarithmic_error(y_pred, y_true))
            self.metrics_dict[split]['r2'].append(RegressionMetrics.r_squared(y_pred, y_true))
            self.metrics_dict[split]['adjusted_r2'].append(RegressionMetrics.adjusted_r_squared(y_pred, y_true,n_features))
            self.metrics_dict[split]['explained_variance_score'].append(RegressionMetrics.explained_variance_score(y_pred, y_true))
            self.metrics_dict[split]['huber_loss'].append(RegressionMetrics.huber_loss(y_pred, y_true))
            self.metrics_dict[split]['batch_loss'].append(batch_loss)
        else:
            self.metrics_dict[split]['accuracy'].append(ClassificationMetrics.accuracy(y_pred, y_true))
            confusion_matrix=ClassificationMetrics.confusion_matrix(y_pred, y_true, num_classes)
            self.metrics_dict[split]['confusion_matrix'].append(confusion_matrix)
            self.metrics_dict[split]['batch_loss'].append(batch_loss)
            self.metrics_dict[split]['class_weights'].append(ClassificationMetrics.class_weights(y_true))
        self.metrics_dict[split]['epoch'].append(epoch)
        self.n_metrics+=1

    def get_metrics_dict(self):
        return self.metrics_dict
    
    def get_metric_names(self):
        return self.metric_names
    
    def format_for_json(self):
        """
        Formats all metrics for JSON serialization, converting tensors to lists.
        """
        formatted_dict = {}
        for key, value in self.metrics_dict.items():
            print(key)
            if isinstance(value, dict):
                formatted_dict[key] = {k: self._tensor_to_list(v) for k, v in value.items()}
            else:
                formatted_dict[key] = self._tensor_to_list(value)
        return formatted_dict

    def _tensor_to_list(self, item):
        """
        Converts a tensor to a list or returns the item if it's not a tensor.
        """
        if isinstance(item, torch.Tensor):
            return item.tolist()  # Convert tensors to lists
        elif isinstance(item, list):
            return [self._tensor_to_list(x) for x in item]  # Recursively process lists
        else:
            return item  # Return the item as is if not a tensor or list
    
    def save_metrics(self):
        """
        Saves the metrics to a file.

        Args:
            path (str): The path to save the metrics to.
        """
        formatted_data = self.format_for_json()
        with open(os.path.join(self.save_path,'metrics.json'),'w') as f:
            json.dump(formatted_data, f)


class Checkpointer:
    def __init__(self, save_path, verbose=1):
        """
        Initializes the ModelCheckpoint callback.

        Args:
            save_path (str): Directory where the model checkpoints will be saved.
            verbose (int): Verbosity mode, 0 or 1.

        """
        self.save_path = save_path
        self.verbose = verbose


    def save_model(self, model, epoch, checkpoint_name=None):
        """
        Saves the model to the specified path.
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if checkpoint_name is None:
            filename = f'model_epoch_{epoch:04d}.pth'
        else:
            filename = f'{checkpoint_name}.pth'
        filepath = os.path.join(self.save_path, filename)
        torch.save(model.state_dict(), filepath)
        if self.verbose:
            print(f"Model checkpoint saved: {filepath}")