# plot roc curve based on prediction and actual label

def plot_roc_curve(pred, ydata):
    from matplotlib import pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

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