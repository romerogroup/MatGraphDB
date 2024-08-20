import torch
import torcheval.metrics.functional as FEVAL

class RegressionMetrics():
    def mean_absolute_error(y_pred, y_true):
        """
        Calculates the mean absolute error.

        Args:
            y_pred (torch.Tensor): The predicted values.
            y_true (torch.Tensor): The true values.

        Returns:
            torch.Tensor: The mean absolute error.
        """
        return torch.mean(torch.abs(y_pred - y_true))

    def root_mean_squared_error(y_pred, y_true):
        """
        Calculates the root mean squared error.

        Args:
            y_pred (torch.Tensor): The predicted values.
            y_true (torch.Tensor): The true values.

        Returns:
            torch.Tensor: The root mean squared error.
        """
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

    def mean_squared_logarithmic_error(y_pred, y_true):
        """
        Calculates the mean squared logarithmic error.

        Args:
            y_pred (torch.Tensor): The predicted values.
            y_true (torch.Tensor): The true values.

        Returns:
            torch.Tensor: The mean squared logarithmic error.
        """
        return torch.mean((torch.log1p(y_pred) - torch.log1p(y_true)) ** 2)

    def r_squared(y_pred, y_true):
        """
        Calculates the R-squared.

        Args:
            y_pred (torch.Tensor): The predicted values.
            y_true (torch.Tensor): The true values.

        Returns:
            torch.Tensor: The R-squared.
        """
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

    def adjusted_r_squared(y_pred, y_true, n_features):
        """
        Calculates the adjusted R-squared.

        Args:
            y_pred (torch.Tensor): The predicted values.
            y_true (torch.Tensor): The true values.
            n_features (int): The number of features.

        Returns:
            torch.Tensor: The adjusted R-squared.
        """
        n = len(y_true)
        r2 = RegressionMetrics.r_squared(y_pred, y_true)
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    def mean_absolute_percentage_error(y_pred, y_true):
        """
        Calculates the mean absolute percentage error.

        Args:
            y_pred (torch.Tensor): The predicted values.
            y_true (torch.Tensor): The true values.

        Returns:
            torch.Tensor: The mean absolute percentage error.
        """
        return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

    def median_absolute_error(y_pred, y_true):
        """
        Calculates the median absolute error.

        Args:
            y_pred (torch.Tensor): The predicted values.
            y_true (torch.Tensor): The true values.

        Returns:
            torch.Tensor: The median absolute error.
        """
        return torch.median(torch.abs(y_pred - y_true))

    def explained_variance_score(y_pred, y_true):
        """
        Calculates the explained variance score.

        Args:
            y_pred (torch.Tensor): The predicted values.
            y_true (torch.Tensor): The true values.

        Returns:
            torch.Tensor: The explained variance score.
        """
        variance_y_true = torch.var(y_true)
        variance_y_pred = torch.var(y_pred)
        return 1 - (variance_y_true - variance_y_pred) / variance_y_true

    def huber_loss(y_pred, y_true, delta=1.0):
        """
        Calculates the Huber loss.

        Args:
            y_pred (torch.Tensor): The predicted values.
            y_true (torch.Tensor): The true values.
            delta (float, optional): The delta value. Defaults to 1.0. The smaller the delta, the more the loss is penalized for large errors.

        Returns:
            torch.Tensor: The Huber loss.
        """
        error = y_true - y_pred
        is_small_error = torch.abs(error) < delta
        squared_loss = 0.5 * error ** 2
        linear_loss = delta * (torch.abs(error) - 0.5 * delta)
        return torch.where(is_small_error, squared_loss, linear_loss).mean()

    def quantile_loss(y_pred, y_true, quantile=0.5):
        """
        Calculates the quantile loss.

        Args:
            y_pred (torch.Tensor): The predicted values.
            y_true (torch.Tensor): The true values.
            quantile (float, optional): The quantile value. Defaults to 0.5.

        Returns:
            torch.Tensor: The quantile loss.
        """
        error = y_true - y_pred
        loss = torch.max((quantile - 1) * error, quantile * error)
        return torch.mean(loss)



class ClassificationMetrics():

    def class_weights( y_true):
        class_counts = torch.bincount(y_true)
        class_weights = 1. / class_counts.float()  # Convert to float to perform division
        return class_weights
        
    def accuracy(y_pred, y_true):
        """ Computes the accuracy of the classifier. """
        correct = y_pred.eq(y_true).sum()
        return correct.float() / y_true.numel()

    def precision(y_pred, y_true):
        """ Computes the precision of the classifier for binary classification. """
        true_positives = (y_pred * y_true).sum().float()
        predicted_positives = y_pred.sum().float()
        return true_positives / predicted_positives if predicted_positives != 0 else 0.0
    
    def multi_class_accuracy(confusion_matrix=None,y_pred=None, y_true=None, num_classes=None):
        """ Computes the accuracy of the classifier for multi-class classification. """
        # Precision: Diagonal elements / sum of respective column elements
        if confusion_matrix is None:
            conf_matrix=ClassificationMetrics.confusion_matrix(y_pred, y_true, num_classes)
        else:
            conf_matrix=confusion_matrix

        # Calculate correct predictions per class
        correct_predictions = torch.diag(conf_matrix)

        # Total actual instances for each class (sum over each row)
        total_true = conf_matrix.sum(dim=1)
        per_class_acc = correct_predictions / total_true

        return per_class_acc


    def multiclass_precision(confusion_matrix=None,y_pred=None, y_true=None, num_classes=None):
        """ Computes the precision of the classifier for multi-class classification. """
        # Precision: Diagonal elements / sum of respective column elements
        if confusion_matrix is None:
            conf_matrix=ClassificationMetrics.confusion_matrix(y_pred, y_true, num_classes)
        else:
            conf_matrix=confusion_matrix
        precision = torch.diag(conf_matrix) / conf_matrix.sum(0)
        precision[torch.isnan(precision)] = 0  # handle NaNs due to division by zero
        return precision

    def multiclass_recall(confusion_matrix=None,y_pred=None, y_true=None, num_classes=None):
        """ Computes the recall of the classifier for multi-class classification. """
        if confusion_matrix is None:
            conf_matrix=ClassificationMetrics.confusion_matrix(y_pred, y_true, num_classes)
        else:
            conf_matrix=confusion_matrix
        recall = torch.diag(conf_matrix) / conf_matrix.sum(1)
        recall[torch.isnan(recall)] = 0  # handle NaNs
        return recall

    def multiclass_f1_score(confusion_matrix=None,y_pred=None, y_true=None, num_classes=None):
        """ Computes the F1 score of the classifier for multi-class classification. """
        if confusion_matrix is None:
            conf_matrix=ClassificationMetrics.confusion_matrix(y_pred, y_true, num_classes)
        else:
            conf_matrix=confusion_matrix
        precision = ClassificationMetrics.multiclass_precision(conf_matrix)
        recall = ClassificationMetrics.multiclass_recall(conf_matrix)
        # F1 Score: Harmonic mean of precision and recall
        f1 = 2 * (precision * recall) / (precision + recall)
        f1[torch.isnan(f1)] = 0  # handle NaNs
        return f1

    def recall(y_pred, y_true):
        """ Computes the recall of the classifier for binary classification. """
        true_positives = (y_pred * y_true).sum().float()
        actual_positives = y_true.sum().float()
        return true_positives / actual_positives if actual_positives != 0 else 0.0

    def f1_score(y_pred, y_true):
        """ Computes the F1 score of the classifier for binary classification. """
        prec = ClassificationMetrics.precision(y_pred, y_true)
        rec = ClassificationMetrics.recall(y_pred, y_true)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0.0

    def confusion_matrix(y_pred, y_true, num_classes):
  
        return FEVAL.multiclass_confusion_matrix(y_pred, y_true, num_classes)
       

    def roc_auc_score(y_pred, y_true):
        """ Computes ROC AUC score for binary classification. """
        # This requires sklearn, as PyTorch does not have native AUC computation
        from sklearn.metrics import roc_auc_score
        y_true_np = y_true.cpu().detach().numpy()
        y_scores_np = y_pred.cpu().detach().numpy()[:, 1]  # Probabilities for the positive class
        return roc_auc_score(y_true_np, y_scores_np)

    def log_loss(y_pred, y_true):
        """ Computes the log loss (cross-entropy loss), assuming y_pred are probabilities. """
        return torch.nn.functional.binary_cross_entropy(y_pred, y_true)

    def matthews_corrcoef(y_pred, y_true):
        """ Computes the Matthews correlation coefficient for binary classification. """
        conf_matrix = ClassificationMetrics.confusion_matrix(y_pred, y_true, 2)
        tp = conf_matrix[1, 1].float()
        tn = conf_matrix[0, 0].float()
        fp = conf_matrix[0, 1].float()
        fn = conf_matrix[1, 0].float()
        
        numerator = (tp * tn - fp * fn)
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / denominator if denominator != 0 else 0.0
