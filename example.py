import numpy as np


if __name__=="__main__":
    # Make an example anomaly score distribution, and split between them.
    numSamples = 1000
    numLabels = 10
    outlierLabel = 123

    # Create two normal distributions, one Inlier and one Outlier.
    aScoreIn = np.random.normal(1, 1, numSamples)
    aScoreOOD = np.random.normal(6, 1, numSamples)

    # Create random labels for the distributions
    labelsIn = np.random.randint(0, numLabels, numSamples)
    labelsOOD = np.random.randint(0, numLabels, numSamples)

    lt = labelsIn  # Assume 100% Accuracy for this case
    ltOOD = outlierLabel * np.ones(numSamples)

    anomalyScores = np.concatenate([aScoreIn, aScoreOOD])
    labels = np.concatenate([labelsIn, labelsOOD])
    trueLabels = np.concatenate([lt, ltOOD])
