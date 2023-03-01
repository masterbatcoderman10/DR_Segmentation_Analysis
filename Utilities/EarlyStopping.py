class EarlyStopping:

    def __init__(self, patience=2):

        self.patience = patience
        self.best = float('inf')
        self.counter = 0
    
    def computer(self, metric):

        if metric < self.best:
            self.best = metric
        else:
            self.counter += 1
        
        if self.counter > self.patience:
            return -1
        else:
            return 0


