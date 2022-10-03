import torch

def accuracy(output, target, topk=(1,)):
    """
    Function taken from pytorch examples:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified
    values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class epoch_log(object): 
    """Keeps track of the scores during an epoch and calculates the average"""
    def __init__(self): 
        self.initialize() 

    def initialize(self):
        self.value = 0 
        self.avg = 0 
        self.sum = 0 
        self.count = 0 
    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
