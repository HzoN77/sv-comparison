import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_distributions(anomalyDistribution, dataLabels, outlierLabel):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    plottingLegends = ['inliers', 'outliers']
    sns.distplot(anomalyDistribution[dataLabels != outlierLabel], ax=ax1, bins=30,
                 hist_kws={"label": plottingLegends[0]})
    sns.distplot(anomalyDistribution[dataLabels == outlierLabel], ax=ax1, bins=30,
                 hist_kws={"label": plottingLegends[1]})

    ax1.legend()
    ax1.set_xlabel('Anomaly score')
    plt.grid()
    plt.draw()


def plot_ROC_curve(anomalyDistribution, labels, outlierLabel, bins=500):

    fpr = np.zeros(bins)
    tpr = np.zeros(bins)

    for i, th in zip(range(bins), np.linspace(anomalyDistribution.max(), anomalyDistribution.min(), bins)):
        tp = float(anomalyDistribution[(labels == outlierLabel) & (anomalyDistribution > th)].shape[0])
        tn = float(anomalyDistribution[(labels != outlierLabel) & (anomalyDistribution < th)].shape[0])
        fp = float(anomalyDistribution[(labels != outlierLabel) & (anomalyDistribution > th)].shape[0])
        fn = float(anomalyDistribution[(labels == outlierLabel) & (anomalyDistribution < th)].shape[0])

        if fp > 0:
            fpr[i] = fp / (tn + fp)
        if tp > 0:
            tpr[i] = tp / (tp + fn)

    auroc = np.trapz(tpr, fpr)

    # plot ROC
    fig, ax1 = plt.subplots(figsize=(7, 4))
    plt.plot(fpr, tpr, 'b-')
    plt.title('area under curve = ' + str(round(auroc, 3)))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid()
    plt.draw()


def plot_precision_recall_curve(anomalyDistribution, labels, outlierLabel, bins=500):

    precision = np.ones(bins)
    recall = np.zeros(bins)  # recall is another name for true positive rate.

    for i, th in zip(range(bins), np.linspace(anomalyDistribution.max(), anomalyDistribution.min(), bins)):
        tp = float(anomalyDistribution[(labels == outlierLabel) & (anomalyDistribution > th)].shape[0])
        tn = float(anomalyDistribution[(labels != outlierLabel) & (anomalyDistribution < th)].shape[0])
        fp = float(anomalyDistribution[(labels != outlierLabel) & (anomalyDistribution > th)].shape[0])
        fn = float(anomalyDistribution[(labels == outlierLabel) & (anomalyDistribution < th)].shape[0])

        if tp > 0:
            precision[i] = tp/(tp + fp)
        if tp > 0:
            recall[i] = tp / (tp + fn)

    aupr = np.trapz(precision, recall)

    # Plot precision-recall curve
    fig, ax1 = plt.subplots(figsize=(7, 4))
    plt.plot(recall, precision, 'b-')
    plt.title('area under PR-curve= ' + str(round(aupr, 3)))
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.grid()
    plt.draw()


def plot_risk_vs_coverage_curve(anomalyDistribution, predictions, labels, outlierLabel, bins=500):
    # Function that plots the coverage varying over risk exposure. Risk is defined as the chance to miss-classify an
    # an input.

    risk = np.empty(bins)
    coverage = np.empty(bins)
    nbrOutliersLeft = np.empty(bins)
    threshold = np.linspace(anomalyDistribution.max(), anomalyDistribution.min(), bins)

    for i, th in zip(range(bins), threshold):
        idx = anomalyDistribution <= th
        coverage[i] = sum(idx) / len(labels)
        risk[i] = (~np.equal(predictions[idx], labels[idx])).sum() / sum(idx)
        nbrOutliersLeft[i] = np.sum((idx == True) & (labels == outlierLabel))

    # plot risk-coverage
    fig, ax1 = plt.subplots(figsize=(7, 4))
    plt.plot(coverage, risk)
    plt.xlabel('coverage')
    plt.ylabel('risk')
    plt.grid()
    plt.draw()