import unittest
import torch
import numpy as np
import torchvision.transforms as transforms

from utils.pytorch_eval import pytorch_predict
from utils.metrics import MetricPlots
from models.cifar_example_net import Net
from datasets.loader import load_cifar10, load_tiny_imagenet
from supervisors.baseline import BaseLine


# Mimicing the code used as example for this repository
class TestSupervisorExample(unittest.TestCase):
    def setUp(self):
        net = Net()
        net.load_state_dict(torch.load('../models/saved/cifar10/simpleNet-10-epochs.pth'))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        testloader_inlier = load_cifar10(transform=transform, train=False)

        transform_tiny = transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        testloader_outlier = load_tiny_imagenet(transform=transform_tiny, train=False)

        supervisor = BaseLine()

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        outputsInlier, labelsInlier = pytorch_predict(model=net, device=device, dataloader=testloader_inlier)
        anomalyScoreInlier = supervisor.anomaly_score(torch.stack(outputsInlier).cpu().numpy())

        outputsOutliers, labels = pytorch_predict(model=net, device=device, dataloader=testloader_outlier)
        anomalyScoreOutlier = supervisor.anomaly_score(torch.stack(outputsOutliers).cpu().numpy())

        # Create predicted classes for the distributions
        _, predictionsInliers = torch.max(torch.stack(outputsInlier), 1)
        _, predictionsOutliers = torch.max(torch.stack(outputsOutliers), 1)

        labelsOutliers = np.array([-1] * len(outputsOutliers))
        labelsInlier = torch.FloatTensor(labelsInlier)
        self.outlierLabel = -1

        # Concatenate the anomaly score, predictions and true labels
        self.anomalyScores = np.concatenate([np.array(anomalyScoreInlier),
                                             np.array(anomalyScoreOutlier)])

        self.predictions = np.concatenate([predictionsInliers.cpu().numpy(),
                                           predictionsOutliers.cpu().numpy()])

        self.trueLabels = np.concatenate([labelsInlier.cpu().numpy(), labelsOutliers])

        self.mp = MetricPlots(self.anomalyScores, self.predictions, self.trueLabels, self.outlierLabel)

    def test_metrics_plot(self):

        self.assertTrue(self.mp.fpr[-1] == 1)  # Check that FPR, TPR go from 0 to 1
        self.assertTrue(self.mp.tpr[-1] == 1)

        self.assertTrue(self.mp.fpr[0] == 0)
        self.assertTrue(self.mp.tpr[0] == 0)

        self.assertTrue(len(self.mp.tpr) == self.mp.bins)  # Check that all params are same dimension
        self.assertTrue(len(self.mp.fpr) == self.mp.bins)
        self.assertTrue(len(self.mp.precision) == self.mp.bins)
        self.assertTrue(len(self.mp.risk) == self.mp.bins)
        self.assertTrue(len(self.mp.coverage) == self.mp.bins)

        self.assertTrue(self.mp.risk.max() <= 1)  # Risk between 1 and 0.
        self.assertTrue(self.mp.risk.min() >= 0)

        self.assertTrue(self.mp.coverage.max() <= 1)  # Risk between 1 and 0.
        self.assertTrue(self.mp.coverage.min() >= 0)


    def test_scores(self):
        self.assertTrue(len(self.anomalyScores) == len(self.predictions))  # Check that each prediction got one anomaly score
        self.assertTrue(len(self.anomalyScores) == len(self.trueLabels))
        self.assertTrue(len(self.trueLabels) == len(self.predictions))

        self.assertTrue(self.anomalyScores.max() <= 1)  # Anomaly scores between 1 and 0.
        self.assertTrue(self.anomalyScores.min() >= 0)






