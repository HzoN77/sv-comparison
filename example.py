import numpy as np


if __name__=="__main__":
    # Make an example anomaly score distribution, and split between them.
    numSamples = 1000
    numLabels = 10
    outlierLabel = 123

    # Create two normal distributions, one Inlier and one Outlier.
    anomalyScoreInlier = np.random.normal(1, 1, numSamples)
    anomalyScoreOutlier = np.random.normal(6, 1, numSamples)

    # Create random labels for the distributions
    labelsIn = np.random.randint(0, numLabels, numSamples)
    labelsOOD = np.random.randint(0, numLabels, numSamples)

    trueLabelsInlier = labelsIn  # Assume 100% Accuracy for this case
    trueLabelsOutlier = outlierLabel * np.ones(numSamples)

    anomalyScores = np.concatenate([anomalyScoreInlier, anomalyScoreOutlier])
    labels = np.concatenate([labelsIn, labelsOOD])
    trueLabels = np.concatenate([trueLabelsInlier, trueLabelsOutlier])
