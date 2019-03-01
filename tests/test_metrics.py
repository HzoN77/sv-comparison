import unittest
import torchvision.transforms as transforms
import torch
from datasets.loader import *
import numpy as np
from utils.metrics import MetricPlots

class TestMetricsPlot(unittest.TestCase):
    def setUp(self):
        self.anomaly_score = np.linspace(0, 1, 100)
        self.labels = np.concatenate([np.ones(50), -1* np.ones(50)])
        self.predictions = np.ones(100)
        self.transform = transforms.ToTensor()

    def test_arrays(self):
        self.assertTrue(np.sum(self.labels) == 0)
        self.assertTrue((self.labels==self.predictions).sum() == 50)

    def test_instantiations(self):
        with self.assertRaises(AssertionError):
            mp = MetricPlots()  # Not giving any arguments
        with self.assertRaises(AssertionError):
            mp = MetricPlots(anomalyScoreData=None, predictionLabels=self.predictions,
                             trueLabels=self.labels, outlierLabel=-1)  # Missing AnomalyScore
        with self.assertRaises(AssertionError):
            mp = MetricPlots(anomalyScoreData=self.anomaly_score, predictionLabels=None,
                             trueLabels=self.labels, outlierLabel=-1)  # Missing predictions
        with self.assertRaises(AssertionError):
            mp = MetricPlots(anomalyScoreData=self.anomaly_score, predictionLabels=self.predictions,
                             trueLabels=None, outlierLabel=.1)  # Missing true labels
        with self.assertRaises(AssertionError):
            mp = MetricPlots(anomalyScoreData=self.anomaly_score, predictionLabels=self.predictions,
                             trueLabels=self.labels, outlierLabel=None)  # Missing outlier label

    def test_length(self):
        vec1 = np.ones(10)
        vec2 = np.ones(5)
        with self.assertRaises(ValueError):
            mp = MetricPlots(vec1, vec1, vec2, -1)  # Length not same for all vectors.
        with self.assertRaises(ValueError):
            mp = MetricPlots(vec1, vec2, vec1, -1)  # Length not same for all vectors.
        with self.assertRaises(ValueError):
            mp = MetricPlots(vec2, vec1, vec1, -1)  # Length not same for all vectors.



if __name__ == "__main__":
    unittest.main()