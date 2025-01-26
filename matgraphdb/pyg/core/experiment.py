import logging
import os
from datetime import datetime

import mlflow

from matgraphdb.pyg.core.logging import (
    log_metrics,
    log_parameters,
    log_task_metrics,
    mlflow_run,
)
from matgraphdb.pyg.utils.data import split_data

logger = logging.getLogger(__name__)

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"


def print_configurations(experiment_config, train_config, model_config, data_config):
    """Print all configurations in a structured format"""
    print("\n" + "=" * 50)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 50)
    for key, value in experiment_config.items():
        print(f"{key:25}: {value}")

    print("\n" + "=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    for key, value in train_config.items():
        print(f"{key:25}: {value}")

    print("\n" + "=" * 50)
    print("MODEL CONFIGURATION")
    print("=" * 50)
    for key, value in model_config.items():
        print(f"{key:25}: {value}")

    print("\n" + "=" * 50)
    print("DATA CONFIGURATION")
    print("=" * 50)
    for key, value in data_config.items():
        print(f"{key:25}: {value}")
    print("\n")


def run_experiment(
    model_cls,
    trainer_cls,
    data_list,
    experiment_config,
    train_config,
    model_config,
    data_config,
):
    @mlflow_run(experiment_config.get("experiment_name", "default"))
    def _run():
        logger.info("Starting experiment run")
        logger.info(f"Using model class: {model_cls.__name__}")
        logger.info(f"Using trainer class: {trainer_cls.__name__}")
        logger.info(f"Dataset size: {len(data_list)} samples")

        print_configurations(experiment_config, train_config, model_config, data_config)

        # Dataset preparation
        logger.info("Preparing datasets...")
        train_loader, val_loader, test_loader = split_data(
            data_list,
            train_size_ratio=train_config["train_size_ratio"],
            batch_size=train_config["batch_size"],
            seed=train_config.get("seed", None),
        )
        logger.info(
            f"Created data loaders with batch size {train_config['batch_size']}"
        )

        # Model initialization
        logger.info("Initializing model...")
        model = model_cls(**model_config)
        logger.info(f"Model architecture:\n{model}")

        # Trainer setup
        logger.info("Setting up trainer...")
        trainer = trainer_cls(
            model=model,
            optimizer=train_config["optimizer"],
            loss_fn=train_config["loss_fn"],
            device=train_config["device"],
            optimizer_kwargs=train_config.get("optimizer_kwargs", {}),
        )
        logger.info(f"Using device: {train_config['device']}")
        logger.info(f"Using optimizer: {train_config['optimizer'].__name__}")

        logger.info("Logging parameters to MLflow...")
        log_parameters(experiment_config)
        log_parameters(train_config)
        log_parameters(model_config)
        log_parameters(data_config)

        # Training loop
        logger.info(f"Starting training for {train_config['num_epochs']} epochs")
        for epoch in range(1, train_config["num_epochs"] + 1):
            train_loss = trainer.train_step(train_loader)

            if epoch % train_config["interval"] == 0:
                val_loss = trainer.eval_step(val_loader)
                test_loss = trainer.eval_step(test_loader)
                logger.info(
                    f"Epoch {epoch}/{train_config['num_epochs']}: "
                    f"Train Loss = {train_loss:.4f}, "
                    f"Val Loss = {val_loss:.4f}, "
                    f"Test Loss = {test_loss:.4f}"
                )
                log_metrics(epoch, train_loss, val_loss, test_loss)

                # Update best model tracking
                trainer._update_best_model(test_loss, epoch)

        logger.info("Training completed. Restoring best model...")
        trainer.restore_best_model()

        # Generate and log predictions
        logger.info("Generating predictions on test set...")
        results_df = trainer.predict(
            test_loader,
            apply_inverse_transform=data_config.get("apply_log_to_target", False),
        )

        task_type = getattr(model, "task_type", "regression")
        logger.info(f"Computing metrics for task type: {task_type}")
        log_task_metrics(results_df, task_type)

        # Save and log predictions with task-aware filename
        logger.info("Saving predictions...")
        results_csv = f"predictions_{task_type}.csv"
        results_df.to_csv(results_csv, index=False)
        mlflow.log_artifact(results_csv)
        # Save and log predictions
        results_csv = "predictions.csv"
        results_df.to_csv(results_csv, index=False)
        mlflow.log_artifact(results_csv)

        logger.info(
            f"Best validation loss: {trainer.best_loss:.4f} at epoch {trainer.best_epoch}"
        )
        mlflow.log_metrics(
            {"best_val_loss": trainer.best_loss, "best_epoch": trainer.best_epoch}
        )

        logger.info("Saving model to MLflow...")
        mlflow.pytorch.log_model(trainer.model, "model")

        logger.info("Experiment completed successfully")
        return model

    return _run()
