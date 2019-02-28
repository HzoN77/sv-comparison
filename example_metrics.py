import numpy as np

from utils.metrics import MetricPlots
import matplotlib.pyplot as plt

if __name__=="__main__":
    # Make an example anomaly score distribution, and split between them.
    numSamples = 1000
    numLabels = 10
    outlierLabel = -1

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

    mp = MetricPlots(anomalyScores, predictions, trueLabels, outlierLabel)
    mp.plot_distributions()
    mp.plot_ROC_curve()
    mp.plot_precision_recall_curve()
    mp.plot_risk_vs_coverage_curve()
    plt.show()  # Ensure program wont exit until plots are closed.