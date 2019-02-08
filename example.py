import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    plot_distributions(anomalyScores, trueLabels, outlierLabel)
