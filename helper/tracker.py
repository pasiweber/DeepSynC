import numpy as np

class AESyncLossTracker:
    def __init__(self, total=None, ae=None, sync=None):
        # Initialize attributes with default empty lists if None is provided
        self.total = total if total is not None else []
        self.ae_loss = ae if ae is not None else []
        self.sync_loss = sync if sync is not None else []

    def to_dict(self):
        # Converts the instance attributes into a dictionary
        return {"total": self.total, "ae": self.ae_loss, "sync": self.sync_loss}

    def __repr__(self):
        return f"AESyncLossTracker(Total_Loss={self.total}, AEReconstruction={self.ae}, Sync_Loss={self.sync})"


class EvaluationTracker:
    def __init__(
        self, ari_labeled=None, ari_total=None, ami_labeled=None, ami_total=None, no_labeled_pts=None, dataset_size=None
    ):
        # Initialize attributes with default empty lists if None is provided
        self.ari_labeled = ari_labeled if ari_labeled is not None else []
        self.ari_total = ari_total if ari_total is not None else []
        self.ami_labeled = ami_labeled if ami_labeled is not None else []
        self.ami_total = ami_total if ami_total is not None else []
        self.no_labeled_pts = no_labeled_pts if no_labeled_pts is not None else []
        self.dataset_size = dataset_size
        self.predicted_labels = None

    def to_dict(self):
        # Converts the instance attributes into a dictionary
        return {
            "ari_labeled": self.ari_labeled,
            "ari_total": self.ari_total,
            "ami_labeled": self.ami_labeled,
            "ami_total": self.ami_total,
            "predicted_labels": self.predicted_labels,
            "no_labeled_pts": self.no_labeled_pts,
            "dataset_size": int(self.dataset_size),
        }

    def set_predicted_labels(self, predictions):
        self.predicted_labels = predictions

    def __repr__(self):
        return (
            f"EvaluationTracker(ari_labeled={self.ari_labeled}, "
            f"ari_total={self.ari_total}, ami_labeled={self.ami_labeled}, "
            f"ami_total={self.ami_total})",
            f"labeled/total={self.no_labeled_pts}/{self.dataset_size}",
        )


def update_eval_tracker(gt_labels, epoch_labels, eval_tracker):
    from sklearn.metrics import adjusted_mutual_info_score as ami
    from sklearn.metrics import adjusted_rand_score as ari

    labeled_pts_mask = epoch_labels > -1
    lbld_pts_ami = ami(gt_labels[labeled_pts_mask], epoch_labels[labeled_pts_mask])
    all_pts_ami = ami(gt_labels, epoch_labels)
    lbld_pts_ari = ari(gt_labels[labeled_pts_mask], epoch_labels[labeled_pts_mask])
    all_pts_ari = ari(gt_labels, epoch_labels)

    eval_tracker.ami_labeled.append(lbld_pts_ami)
    eval_tracker.ami_total.append(all_pts_ami)
    eval_tracker.ari_labeled.append(lbld_pts_ari)
    eval_tracker.ari_total.append(all_pts_ari)
    eval_tracker.no_labeled_pts.append(int(np.sum(labeled_pts_mask)))
    eval_tracker.set_predicted_labels(epoch_labels.tolist())
    return eval_tracker