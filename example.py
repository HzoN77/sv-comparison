import numpy as np


if __name__=="__main__":
    # Make an example anomaly score distribution, and split between them.
    numSamples = 1000
    numLabels = 10
    outlierLabel = 123

    aScoreIn = np.random.normal(1, 1, numSamples)
    labelsIn = np.random.randint(0, numLabels, numSamples)
    aScoreOOD = np.random.normal(6, 1, numSamples)

    lt = labelsIn  # Assume 100% Accuracy for this case
    labelsOOD = np.random.randint(0, numLabels, numSamples)
    ltOOD = outlierLabel * np.ones(numSamples)

    anomalyScores = np.concatenate([aScoreIn, aScoreOOD])
    labels = np.concatenate([labelsIn, labelsOOD])
    trueLabels = np.concatenate([lt, ltOOD])
