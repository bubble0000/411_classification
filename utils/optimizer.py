from torch import optim

def build_optimizer(config,param):
    if config['type']=="SGD":
        return optim.SGD(param,**config['kwargs'])
    elif config['type']=="Adam":
        return optim.Adam(param, **config['kwargs'])