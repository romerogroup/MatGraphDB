import inspect
from contextlib import contextmanager

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

# from matgraphdb.pyg.core.metrics import (
#     compute_mae,
#     compute_r2,
#     compute_rmse,
#     get_pearson_r,
# )


def mlflow_run(experiment_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            mlflow.set_tracking_uri("http://127.0.0.1:8080")
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run():
                return func(*args, **kwargs)

        return wrapper

    return decorator


def log_metrics(epoch, train_loss, val_loss, test_loss):
    mlflow.log_metrics(
        {"train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss},
        step=epoch,
    )


def log_parameters(params: dict):
    """Smart parameter logging that handles complex objects"""
    processed = {}
    for k, v in params.items():
        if inspect.isclass(v):
            processed[k] = v.__name__
        elif hasattr(v, "__call__"):
            processed[k] = v.__name__ if hasattr(v, "__name__") else str(v)
        elif isinstance(v, (list, tuple)):
            processed[k] = ",".join(map(str, v))
        elif isinstance(v, dict):

            def process_dict(prefix, d):
                for sub_k, sub_v in d.items():
                    key = f"{prefix}.{sub_k}" if prefix else str(sub_k)
                    if isinstance(sub_v, dict):
                        process_dict(key, sub_v)
                    else:
                        processed[key] = sub_v

            process_dict(k, v)

        else:
            processed[k] = str(v)
    mlflow.log_params(processed)


# def log_table(params:dict):


def log_task_metrics(results_df, task_type):
    """Log metrics based on task type"""
    if task_type == "regression":
        log_correlation_metrics(results_df)
    else:
        log_classification_metrics(results_df, task_type)


def log_correlation_metrics(results_df):
    """Log regression metrics"""
    metrics = {
        "test_pearson_r": results_df[["predicted", "actual"]].corr().iloc[0, 1],
        "test_r2": r2_score(results_df.actual, results_df.predicted),
        "test_mae": mean_absolute_error(results_df.actual, results_df.predicted),
        "test_rmse": mean_squared_error(results_df.actual, results_df.predicted),
    }
    mlflow.log_metrics(metrics)


def log_classification_metrics(results_df, task_type):
    """Log classification metrics"""
    y_true = results_df["actual"]
    y_pred = results_df["predicted"]

    metrics = {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_f1": f1_score(
            y_true, y_pred, average="macro" if task_type == "multiclass" else "binary"
        ),
        "test_precision": precision_score(
            y_true, y_pred, average="macro" if task_type == "multiclass" else "binary"
        ),
        "test_recall": recall_score(
            y_true, y_pred, average="macro" if task_type == "multiclass" else "binary"
        ),
    }

    mlflow.log_metrics(metrics)

    # Log confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm)
    cm_csv = "confusion_matrix.csv"
    cm_df.to_csv(cm_csv, index=False)
    mlflow.log_artifact(cm_csv)


def get_function_parameters(func, *args, **kwargs) -> dict:
    """Capture function arguments as a dictionary"""
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return dict(bound_args.arguments)
