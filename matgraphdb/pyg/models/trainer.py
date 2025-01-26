from abc import ABC, abstractmethod

import mlflow
import torch


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        loss_fn,
        num_epochs,
        interval=1,
        optimizer=None,
        optimizer_kwargs=None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.interval = interval

        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), **optimizer_kwargs
            )

    def train(self, train_step, test_step):
        for epoch in range(1, self.num_epochs + 1):

            if epoch % self.interval == 0:
                record_loss = True
            else:
                record_loss = False

            train_loss = train_step(
                self.model,
                self.train_loader,
                self.optimizer,
                self.loss_fn,
            )
            if record_loss:
                val_loss = test_step(self.model, self.val_loader, self.loss_fn)
                test_loss = test_step(self.model, self.test_loader, self.loss_fn)
                print(
                    f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}"
                )

                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("test_loss", test_loss, step=epoch)
                mlflow.log_metric("train_loss", train_loss, step=epoch)

        mlflow.pytorch.log_model(self.model, "model")
