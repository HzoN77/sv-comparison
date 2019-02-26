from supervisors.supervisor import BaseSupervisor


class BaseLine(BaseSupervisor):
    def __init__(self):
        pass

    def anomaly_score(self, predictions):
        outlier_scores = []
        for pred in predictions:
            score = (pred - pred.min()) / (pred - pred.min()).sum()
            outlier_scores.append(1 - score.max())
        return outlier_scores
