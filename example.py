import numpy as np

from evaluation.metrics import plot_distributions, plot_ROC_curve, plot_precision_recall_curve, \
    plot_risk_vs_coverage_curve

if __name__=="__main__":
    # Make an example anomaly score distribution, and split between them.
    numSamples = 1000
    numLabels = 10
    outlierLabel = 123

    # Create two normal distributions, one Inlier and one Outlier.
    anomalyScoreInlier = np.random.normal(2, 2, numSamples)
    anomalyScoreOutlier = np.random.normal(4, 2, numSamples)

    # Create random labels for the distributions
    predictionsInlier = np.random.randint(0, numLabels, numSamples)
    predictionsOutliers = np.random.randint(0, numLabels, numSamples)

    trueLabelsInlier = predictionsInlier  # Assume 100% Accuracy for this case
    trueLabelsOutlier = outlierLabel * np.ones(numSamples)

    anomalyScores = np.concatenate([anomalyScoreInlier, anomalyScoreOutlier])
    predictions = np.concatenate([predictionsInlier, predictionsOutliers])
    trueLabels = np.concatenate([trueLabelsInlier, trueLabelsOutlier])

    plot_distributions(anomalyScores, trueLabels, outlierLabel)
    plot_ROC_curve(anomalyScores, trueLabels, outlierLabel)
    plot_precision_recall_curve(anomalyScores, trueLabels, outlierLabel)
    plot_risk_vs_coverage_curve(anomalyScores, predictions, trueLabels, outlierLabel)

