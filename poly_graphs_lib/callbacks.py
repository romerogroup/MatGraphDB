import copy

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
        self.counter = 0
        self.status = 0

    def __call__(self, model, val_loss:float, mape_val_loss:float):
        """The class calling method

        Parameters
        ----------
        model : torch.nn.Module
            The pytorch model
        val_loss : float
            The validation loss
        mape_val_loss : float
            The map_val_loss

        Returns
        -------
        _type_
            _description_
        """
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_mape_loss = mape_val_loss
            self.best_model = copy.deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.best_mape_loss = mape_val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - val_loss < self.min_delta:
            self.counter +=1
            if self.counter >= self.patience:
                self.status = f'Stopped on {self.counter}'
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                return True
            self.status = f"{self.counter}/{self.patience}"
            # print( self.status )
        return False