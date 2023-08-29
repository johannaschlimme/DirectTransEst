import torch


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='best_metric_model.pth', val_interval=5):
        """
        Args:
            patience (int): How many epochs to wait before stopping after the metric stops improving.
            verbose (bool): If True, prints a message for each validation metric improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the model.
            val_interval (int): Number of epochs between validation phases.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_dice_max = float('-inf')
        self.delta = delta
        self.path = path
        self.val_interval = val_interval

    def __call__(self, val_dice, model):

        score = val_dice
        true_patience = self.patience / self.val_interval

        if self.best_score is None:
            self.best_score = score
            self.save_model(val_dice, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter + self.val_interval} out of {self.patience}')
            if self.counter >= true_patience and self.best_score > 0.5:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_model(val_dice, model)
            self.counter = 0

    def save_model(self, val_dice, model):
        '''Saves model when validation dice increases.'''
        if self.verbose:
            print(f'Validation dice increased ({self.val_dice_max:.6f} --> {val_dice:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_dice_max = val_dice
