import numpy as np


class Metric:
    def __init__(self, conf_th: float, iou_th: float = 0.25) -> None:
        self.conf_th = conf_th
        self.iou_th = iou_th

    def reset(self) -> None:
        self.cmat = np.zeros((5, 5)).astype(np.int16)

    def update(self, row):
        self.reset()

        label = row["label"]
        preds = np.array(row["preds"])
        reliabilities = np.array(row["reliability"])
        ious = np.array(row["iou"])

        if 0 in row["preds"]:
            self.cmat[label, 0] += 1
        elif (ious > self.iou_th).sum() >= 2:
            pred = preds[reliabilities.argmax()]
            self.cmat[label, pred] += 1
        else:
            for pred, iou in zip(row["preds"], row["iou"]):
                if iou >= self.iou_th:
                    self.cmat[label, pred] += 1
                else:
                    self.cmat[0, pred] += 1
                    self.cmat[label, 0] += 1

        for i, diagnosis in enumerate(self.cmat):
            if i == 0:
                continue

            while diagnosis.sum() >= 2:
                self.cmat[i, 0] -= 1
