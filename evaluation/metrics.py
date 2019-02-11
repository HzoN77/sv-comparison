import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


class MetricPlots(object):
    """
    Returns a ```Metric&Plots``` object with the option to plot.

    """
    def __init__(self, anomalyScoreData, predictionLabels, trueLabels, outlierLabel, bins=500):
            self.anomalyScoreData = anomalyScoreData
            self.predictionLabels = predictionLabels
            self.trueLabels = trueLabels
            self.outlierLabel = outlierLabel
            self.bins = bins
            self.tpr, self.fpr, self.precision = self.calculate_roc_metrics()

    def calculate_roc_values(self, bins=None):
        if not bins:
            bins = self.bins

        fpr = np.zeros(bins)
        tpr = np.zeros(bins)  # also known as recall
        precision = np.ones(bins)
        for i, th in zip(range(bins), np.linspace(self.anomalyScoreData.max(), self.anomalyScoreData.min(), bins)):
            tp = float(self.anomalyScoreData[(self.trueLabels == self.outlierLabel) & (self.anomalyScoreData > th)].shape[0])
            tn = float(self.anomalyScoreData[(self.trueLabels != self.outlierLabel) & (self.anomalyScoreData < th)].shape[0])
            fp = float(self.anomalyScoreData[(self.trueLabels != self.outlierLabel) & (self.anomalyScoreData > th)].shape[0])
            fn = float(self.anomalyScoreData[(self.trueLabels == self.outlierLabel) & (self.anomalyScoreData < th)].shape[0])

            if fp > 0:
                fpr[i] = fp / (tn + fp)
            if tp > 0:
                tpr[i] = tp / (tp + fn)
                precision[i] = tp / (tp + fp)

        return tpr, fpr, precision

    def plot_distributions(self, bins=30):
        fig, ax1 = plt.subplots(figsize=(7, 4))
        plottingLegends = ['inliers', 'outliers']
        sns.distplot(self.anomalyScoreData[self.trueLabels != self.outlierLabel], ax=ax1, bins=bins,
                     hist_kws={"label": plottingLegends[0]})
        sns.distplot(self.anomalyScoreData[self.trueLabels == self.outlierLabel], ax=ax1, bins=bins,
                     hist_kws={"label": plottingLegends[1]})

        ax1.legend()
        ax1.set_xlabel('Anomaly score')
        plt.grid()
        plt.draw()

    def plot_ROC_curve(self, bins=500):
        if bins != self.bins:
            self.calculate_roc_values(bins=bins)  # Update fpr, tpr and precision if different bin size.

        auroc = np.trapz(self.tpr, self.fpr)

        # plot ROC
        fig, ax1 = plt.subplots(figsize=(7, 4))
        plt.plot(self.fpr, self.tpr, 'b-')
        plt.title('area under curve = ' + str(round(auroc, 3)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.grid()
        plt.draw()

    def plot_precision_recall_curve(self, bins=500):
        if bins != self.bins:
            self.calculate_roc_values(bins=bins)  # Update fpr, tpr and precision.

        aupr = np.trapz(self.precision, self.tpr)

        # Plot precision-recall curve
        fig, ax1 = plt.subplots(figsize=(7, 4))
        plt.plot(self.tpr, self.precision, 'b-')
        plt.title('area under PR-curve= ' + str(round(aupr, 3)))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.grid()
        plt.draw()

    def plot_risk_vs_coverage_curve(self, bins=500):
        # Function that plots the coverage varying over risk exposure. Risk is defined as the chance to miss-classify an
        # an input.

        risk = np.empty(bins)
        coverage = np.empty(bins)
        nbrOutliersLeft = np.empty(bins)
        threshold = np.linspace(self.anomalyScoreData.max(), self.anomalyScoreData.min(), bins)

        for i, th in zip(range(bins), threshold):
            idx = self.anomalyScoreData <= th
            coverage[i] = sum(idx) / len(self.trueLabels)
            risk[i] = (~np.equal(self.predictionLabels[idx], self.trueLabels[idx])).sum() / sum(idx)
            nbrOutliersLeft[i] = np.sum((idx == True) & (self.trueLabels == self.outlierLabel))

        # plot risk-coverage
        fig, ax1 = plt.subplots(figsize=(7, 4))
        plt.plot(coverage, risk)
        plt.xlabel('coverage')
        plt.ylabel('risk')
        plt.grid()
        plt.draw()