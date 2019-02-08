import numpy as np


if __name__=="__main__":
    # Make an example anomaly score distribution, and split between them.
    n = 1000
    l = 10

    aScoreIn = np.random.normal(1, 1, n)
    labelsIn = np.random.randint(0, l, n)
    aScoreOOD = np.random.normal(6, 1, n)

    lt = labelsIn  # Assume 100% Accuracy for this case
    labelsOOD = np.random.randint(0, l, n)
    ltOOD = 10 * np.ones(n)

    anomalyScores = np.concatenate([aScoreIn, aScoreOOD])
    labels = np.concatenate([labelsIn, labelsOOD])
    trueLabels = np.concatenate([lt, ltOOD])
