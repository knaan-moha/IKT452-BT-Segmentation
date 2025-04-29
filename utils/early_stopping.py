
class EarlyStopping:
    def __init__(self, patience=5, mode='min'):
        self.patience = patience
        self.mode = mode
        self.best_loss = None
        self.counter = 0
        self.stop_training = False

        if mode == 'min':
            self.best_loss = float('inf')
        elif mode == 'max':
            self.best_loss = float('-inf')
        else:
            raise ValueError("Invalid mode. Use 'min' or 'max'.")

    def __call__(self, current_loss):
        if self.mode == 'min' and current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        elif self.mode == 'max' and current_loss > self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stop_training = True
            print("Early stopping the training process.")