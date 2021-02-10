# plot roc curve based on prediction and actual label
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score

def plot_roc_curve(pred, ydata):
    
    # plot the roc curve
    #
    # input:
        # pred: prediction
        # ydata: actual data
    # output:
        # plots the chart
    # return:
        # None
    
    ns_probs = [0 for _ in range(len(ydata))]
    lr_probs = pred

    ns_auc = roc_auc_score(ydata, ns_probs)
    lr_auc = roc_auc_score(ydata, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(ydata, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(ydata, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    
def eval_metrics(y_pred, y_test, praba_threshold=0.5):
    predx = [x for x in y_pred > praba_threshold]
    cm = confusion_matrix(y_test, predx).ravel()
    print(f'ROC_AUC Score: {round(roc_auc_score(y_test, y_pred),3)}')
    print(f'-----------\nConfusion Matrix with thresthold of {praba_threshold}:')
    print(f'TN: {cm[0]}, FP: {cm[1]}\nFN: {cm[2]}, TP: {cm[3]}')
    print(f'{np.sum(predx)} out of {len(predx)} or {int(round(np.sum(predx)/len(predx)*100,0))}% of players are predicted to be 5+ Yrs')
    print(f'-----------\nRecall: {round(recall_score(y_test, predx), 4)}')
    print(f'Accuracy: {round(accuracy_score(y_test, predx), 4)}')
    print(f'Balanced Accuracy: {round(balanced_accuracy_score(y_test, predx), 4)}')
    print(f'F1: {round(f1_score(y_test, predx), 4)}')
    print(f'-----------')
    plot_roc_curve(y_pred, y_test)