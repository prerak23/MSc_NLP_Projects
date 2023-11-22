from torch.optim import SGD

from model import Model


class OptimArgs:
    def __init__(self, model: Model, learning_rate: float, weight_decay: float):
        self.parameters = model.parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay


class Optim(SGD):
    def __init__(self, optim_args: OptimArgs):
        super(Optim, self).__init__(optim_args.parameters(),
                                    lr=optim_args.learning_rate,
                                    weight_decay=optim_args.weight_decay)


