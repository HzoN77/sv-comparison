import torch
from models.cifar_example_net import Net
import matplotlib.pyplot as plt

if __name__=="__main__":
    # Read in a pre-trained network
    net = Net()
    net.load_state_dict(torch.load('models/saved/cifar10/simpleNet-10-epochs.pth'))

    # Import and create transforms for inliers and outliers.
    import torchvision.transforms as transforms
    from datasets.loader import load_cifar10, load_tiny_imagenet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testloader_inlier = load_cifar10(transform=transform, train=False)

    # Note that Tiny ImageNet is 64x64, thus require resize.
    transform_tiny = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testloader_outlier = load_tiny_imagenet(transform=transform_tiny, train=False)

    # Import function for pytorch predictions.
    from utils.pytorch_eval import pytorch_predict
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Create Anomaly Score, based on the Simple Supervisor (BaseLine)
    from supervisors.baseline import BaseLine
    import torch
    baseline = BaseLine()

    outputsInlier, labelsInlier = pytorch_predict(model=net, device=device, dataloader=testloader_inlier)
    anomalyScoreInlier = baseline.anomaly_score(torch.stack(outputsInlier).cpu().numpy())

    outputsOutliers, labels = pytorch_predict(model=net, device=device, dataloader=testloader_outlier)
    anomalyScoreOutlier = baseline.anomaly_score(torch.stack(outputsOutliers).cpu().numpy())

    # Create predicted classes for the distributions
    _, predictionsInliers = torch.max(torch.stack(outputsInlier), 1)
    _, predictionsOutliers = torch.max(torch.stack(outputsOutliers), 1)

    # Outlier labels for outlier samples.
    import numpy as np
    labelsOutliers = np.array([-1] * len(outputsOutliers))

    # Concatenate the anomaly score, predictions and true labels
    anomalyScores = np.concatenate([np.array(anomalyScoreInlier), np.array(anomalyScoreOutlier)])

    predictions = np.concatenate([predictionsInliers.cpu().numpy(),
                                  predictionsOutliers.cpu().numpy()])
    labelsInlier = torch.FloatTensor(labelsInlier)
    trueLabels = np.concatenate([labelsInlier.cpu().numpy(), labelsOutliers])

    # Create a MatricPlots object
    from utils.metrics import MetricPlots
    outlierLabel = -1
    mp = MetricPlots(anomalyScores, predictions, trueLabels, outlierLabel)

    # Draw all the plots.
    mp.plot_distributions()
    mp.plot_ROC_curve()
    mp.plot_precision_recall_curve()
    mp.plot_risk_vs_coverage_curve()
    plt.show()  # Ensure program is running until plots are closed
