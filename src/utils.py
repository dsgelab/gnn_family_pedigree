import torch
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

class EarlyStopping:
    """Early stops training if validation loss doesn't improve after a given patience.
    SEE: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class WeightedBCELoss(torch.nn.Module):

    def __init__(self, num_samples_dataset, num_samples_minority_class, num_samples_majority_class, device):
        super(WeightedBCELoss,self).__init__()
        self.num_samples_dataset = num_samples_dataset
        self.total_positive_samples = num_samples_minority_class
        self.total_negative_samples = num_samples_majority_class
        self.device = device

    def forward(self, y_est, y):
        positive_weight = torch.tensor(self.total_negative_samples/self.total_positive_samples).to(self.device)
        bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
        weighted_bce_loss = bce_loss(y_est, y)
        return weighted_bce_loss


def get_classification_threshold_auc(y_pred, y_actual):
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    threshold = thresholds[ix]
    return threshold

def get_classification_threshold_precision_recall(y_pred, y_actual):
    thresholds = np.arange(0.1, 0.9, 0.001) # between 0.1 and 0.9 to exclude trivial values like 0 and 1
    scores = [f1_score(y_actual, (y_pred >= t).astype('int')) for t in thresholds]
    ix = np.argmax(scores)
    threshold = thresholds[ix]
    return threshold


def brier_skill_score(actual_y, predicted_prob_y):
    e = sum(actual_y) / len(actual_y)
    bs_ref = sum((e-actual_y)**2) / len(actual_y)
    bs = sum((predicted_prob_y-actual_y)**2) / len(actual_y)
    bss = 1 - bs / bs_ref
    return bss


def calculate_metrics(actual_y, predicted_y, predicted_prob_y):
    auc_roc = metrics.roc_auc_score(actual_y, predicted_prob_y)
    precision, recall, thresholds = metrics.precision_recall_curve(actual_y, predicted_prob_y)
    auc_prc = metrics.auc(recall, precision)
    mcc = metrics.matthews_corrcoef(actual_y, predicted_y)
    tn, fp, fn, tp = metrics.confusion_matrix(actual_y, predicted_y).ravel()
    ts = tp / (tp + fn + fp)
    recall = tp / (tp + fn)
    f1 = (2*tp) / (2*tp + fp + fn)
    bss = brier_skill_score(actual_y, predicted_prob_y)
    
    metric_results = {
        'metric_auc_roc':auc_roc, # typically reported, but can be biased for imbalanced classes
        'metric_auc_prc':auc_prc, # better suited for imbalanced classes
        'metric_f1':f1, # also should be better suited for imbalanced classes
        'metric_recall':recall, # important for medical studies, to reduce misses of positive instances
        'metric_mcc':mcc, # correlation that is suitable for imbalanced classes
        'metric_ts':ts, # suited for rare events, penalizing misclassification as the rare event (fp)
        'metric_bss':bss, # brier skill score, where higher score corresponds better calibration of predicted probabilities
        'true_negatives':tn, 
        'false_positives':fp, 
        'false_negatives':fn, 
        'true_positives':tp}
       
    return metric_results 


def plot_losses(train_losses, valid_losses, outprefix):
    plt.plot(train_losses, label='Train')
    plt.plot(valid_losses, label='Validate')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('{}_train_loss.png'.format(outprefix))
    plt.clf()


def enable_dropout(model):
    """
    Function to enable the dropout layers during test-time -
    this is needed to get uncertainty estimates with Monte Carlo dropout techniques
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
