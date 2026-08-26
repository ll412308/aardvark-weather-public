"""Simple validation-loss early stopping with checkpointable state."""


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss = None
        self.bad_validations = 0

    def update(self, val_loss):
        improved = (
            self.best_loss is None
            or val_loss < self.best_loss - self.min_delta
        )
        if improved:
            self.best_loss = float(val_loss)
            self.bad_validations = 0
        else:
            self.bad_validations += 1
        return improved, self.bad_validations >= self.patience

    def state_dict(self):
        return {
            "best_loss": self.best_loss,
            "bad_validations": self.bad_validations,
        }

    def load_state_dict(self, state):
        self.best_loss = state.get("best_loss")
        self.bad_validations = int(state.get("bad_validations", 0))
