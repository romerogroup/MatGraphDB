import json
import os
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, Tuple

import mlflow
import torch


def callbacks(callback_fn: Callable):
    """Decorator to ensure callback functions take a Trainer instance as argument"""

    @wraps(callback_fn)
    def wrapper(trainer: "BaseTrainer", *args, **kwargs):
        return callback_fn(trainer, *args, **kwargs)

    return wrapper


class BaseTrainer(ABC):
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_data,
        train_val_data=None,
        test_data=None,
        test_val_data=None,
        scheduler=None,
        num_epochs: int = 100,
        eval_interval: int = 100,
        baseline_models: list[str] = None,
        metrics: dict = None,
        evaluation_callbacks: list[Callable] = None,
        training_dir: str = "./training_runs",
        use_mlflow: bool = True,
        mlflow_experiment_name: str = None,
        mlflow_tracking_uri: str = "http://127.0.0.1:8080",
        mlflow_record_system_metrics: bool = True,
        additional_parameters: dict = None,
    ):
        """

        Args:

            model: The PyTorch model to train.
            num_epochs (int): Number of epochs for training.
            interval (int): How often to record and log the losses.
            train_loader: DataLoader for the training set.
            train_val_loader: DataLoader for the training validation set.
            test_loader: DataLoader for the test set.
            test_val_loader: DataLoader for the test validation set.
            use_mlflow (bool): Whether to use mlflow to log the metrics.
        """

        self.training_dir = training_dir
        n_runs = len(os.listdir(self.training_dir))
        self.run_dir = os.path.join(self.training_dir, f"run_{n_runs}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.eval_interval = eval_interval
        self.evaluation_callbacks = evaluation_callbacks
        self.use_mlflow = use_mlflow
        self.baseline_models = baseline_models

        self.data_splits = {}
        self.data_splits["train"] = train_data
        if train_val_data is not None:
            self.data_splits["train_val"] = train_val_data
        if test_data is not None:
            self.data_splits["test"] = test_data
        if test_val_data is not None:
            self.data_splits["test_val"] = test_val_data
        self.split_names = list(self.data_splits.keys())

        if self.use_mlflow:
            self.mlflow_experiment_name = mlflow_experiment_name
            self.mlflow_tracking_uri = mlflow_tracking_uri
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(self.mlflow_experiment_name)
            if mlflow_record_system_metrics:
                os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
            if additional_parameters is not None:
                mlflow.log_params(additional_parameters)

    @abstractmethod
    def train_step(self, data_batch, **kwargs) -> float:
        """
        Performs a validation step.

        Args:
            data_batch: A batch of data from the validation DataLoader.

        This method should:
            - Iterate over self.train_loader.
            - Compute the model output and loss using self.loss_fn.
            - Perform backpropagation and an optimizer step using self.optimizer.
            - Return an aggregated training loss (e.g., the average loss over the epoch).
        Returns:
            float: The training loss for the epoch.
        """
        pass

    @abstractmethod
    def validation_step(
        self, data_batch, **kwargs
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Performs a validation step.
        Args:
            data_batch: A batch of data from the validation DataLoader.

        This method should:
            - Use the provided data_batch from a validation DataLoader.
            - Compute the model output and loss (without performing backpropagation).
            - Return an aggregated validation loss.
        Returns:
            float: The validation loss.
            torch.Tensor: The validation output.
            torch.Tensor: The validation target.
        """
        pass

    @abstractmethod
    def eval_metrics(self, preds, targets, **kwargs):
        """
        Evaluates the metrics for the given predictions and targets.

        Args:
            data_batch: A batch of data from the validation DataLoader.

        This method should:
            - Iterate over the test DataLoader (e.g., self.test_loader).
            - Compute the model output and loss (in evaluation mode).
            - Return an aggregated test loss.
        Returns:
            Dict
        """
        pass

    def train(
        self,
        metrics_to_record: list[str] = None,
    ):
        """


        Runs the training loop for the specified number of epochs.

        At every `interval` epoch, it computes validation and test losses,
        prints them, and logs the metrics using mlflow.
        """
        if self.use_mlflow:
            mlflow.start_run()

        if metrics_to_record is None:
            metrics_to_record = []

        self.data_splits_metrics = {}
        for split_name in self.split_names:
            self.data_splits_metrics[split_name]["loss"] = []

        epoch_str = f"Epoch: {epoch:03d}"
        epoch_str += "(" + "|".join(self.split_names) + ")"
        print(epoch_str)

        self.epochs_dir = os.path.join(self.run_dir, "epochs")
        os.makedirs(self.epochs_dir, exist_ok=True)

        self.current_preds = None
        self.current_targets = None
        self.current_loss = None
        self.current_train_loss = None
        self.current_epoch = 0
        for epoch in range(1, self.num_epochs + 1):

            self.current_train_loss = self.train_step(self.data_splits["train"])

            if self.scheduler:
                self.scheduler.step()

            if epoch % self.eval_interval == 0:
                self.current_epoch = epoch
                self.epoch_dir = os.path.join(self.epochs_dir, f"epoch_{epoch}")
                os.makedirs(self.epoch_dir, exist_ok=True)

                metrics_str = f"Epoch: {epoch:03d}"

                for split_name, data_batch in self.data_splits.items():
                    loss, self.current_preds, self.current_targets = (
                        self.validation_step(data_batch)
                    )
                    metrics_dict = self.eval_metrics(
                        self.current_preds, self.current_targets
                    )
                    if epoch == 1:
                        self.data_splits_metrics[split_name]["loss"] = []
                        for metric_name in metrics_dict.keys():
                            self.data_splits_metrics[split_name][metric_name] = []

                    self.data_splits_metrics[split_name]["loss"].append(loss)
                    for metric_name, metric_value in metrics_dict.items():
                        self.data_splits_metrics[split_name][metric_name].append(
                            metric_value
                        )

                    metrics_str += f" || {split_name} Loss: {loss:.4f}"
                for metric_name in metrics_to_record:
                    for split_name in self.split_names:
                        metrics_str += f" || {metric_name}: {self.data_splits_metrics[split_name][metric_name][-1]:.4f} "

                for callback in evaluation_callbacks:
                    callback(self)

                with open(os.path.join(self.epoch_dir, "metrics.json"), "w") as f:
                    json.dump(self.data_splits_metrics, f)

                if self.use_mlflow:
                    for split_name in self.split_names:
                        metrics_dict = self.data_splits_metrics[split_name]
                        for metric_name, metric_values in metrics_dict.items():

                            mlflow.log_metric(
                                f"{split_name}_{metric_name}",
                                metric_values[-1],
                                step=epoch,
                            )

                    mlflow.pytorch.log_model(
                        self.model, os.path.join(self.epoch_dir, "model")
                    )

        if self.use_mlflow:
            mlflow.end_run()


if __name__ == "__main__":
    import torch

    optimizer = torch.optim.Adam(torch.nn.Linear(10, 10).parameters(), lr=0.01)

    print(dir(optimizer))
    print(optimizer.__dict__)
