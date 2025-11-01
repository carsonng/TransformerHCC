import torch
import math
import json
from sklearn.metrics import confusion_matrix, roc_auc_score,recall_score
import numpy as np


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=5e-5):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):

        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)



def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def EMA_smooth(points, beta=0.3, bias=False):
    smoothed_points = []
    t = 0
    for point in points:
        t += 1
        if smoothed_points:
            previous = smoothed_points[-1]
            if bias:
                smoothed_points.append((previous * beta + point * (1 - beta)) / (1 - beta**t))
            else:
                smoothed_points.append(previous * beta + point * (1 - beta))
        else:
            smoothed_points.append(point)
    return smoothed_points


from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score


def calculate_metrics(y_true, y_pred, y_prob):

    y_prob = np.array(y_prob)
    y_prob_positive = y_prob[:, 1]

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    sensitivity = recall_score(y_true, y_pred, pos_label=1)

    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    auc = roc_auc_score(y_true, y_prob_positive)

    return sensitivity, specificity, auc, cm


def warmup_lr(optimizer, epoch, warmup_epochs, initial_lr, delta):
    if epoch <= warmup_epochs:
        lr = initial_lr + (delta * epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

