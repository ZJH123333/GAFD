from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt

def get_acc(pred, label):
    return accuracy_score(label, pred)

def get_pre(pred, label):
    return precision_score(label, pred)

def get_rec(pred, label):
    return recall_score(label, pred)

def get_auc(prob, label):
    score = prob[:, 1]
    return roc_auc_score(label, score)

def get_f1(pred, label):
    return f1_score(label, pred)

def evaluate(pred, prob, label, acc_only = False):
    acc = get_acc(pred, label)
    pre, rec, auc, f1 = None, None, None, None
    if not acc_only:
        pre = get_pre(pred, label)
        rec = get_rec(pred, label)
        auc = get_auc(prob, label)
        f1 = get_f1(pred, label)
    return acc, pre, rec, auc, f1

def plot_roc_curve(prob, label):
    score = prob[:, 1]
    fpr, tpr, _ = roc_curve(label, score)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr, tpr, 'r')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.show()
