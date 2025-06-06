import os
import torch
import numpy as np
from SHiP import SHiP
from tqdm import tqdm

from helper.deepsync_utils import (
    find_local_core_points_same,
    get_high_cof_labels,
    not_assigning_new_points
)
from helper.tracker import update_eval_tracker
from helper.deep import encode_batchwise
from helper.utils import save_dict_as_json

def fit_deepsync(
    device,
    model,
    data,
    dataset_name,
    trainloader,
    testloader,
    optimizer,
    n_check,
    T_stop,
    k,
    percent,
    training_iterations,
    loss_fn,
    labels_assignment_method,
    losses_tracker,
    eval_tracker,
    directory,
    deep_sync_model_path,
    id=0,
    use_tqdm=False,
):
    print("DeepSynC is running . . . ")
    trackers_path = os.path.join(directory, "trackers")
    os.makedirs(trackers_path, exist_ok=True)

    embedded, gt_labels = encode_batchwise(testloader, model, device)
    labels_over_iterations = np.zeros((len(data), 1))
    core_points_mask, _ = find_local_core_points_same(embedded, k, percent)
    original_labels = torch.zeros_like(gt_labels) - 1
    core_points = embedded[np.where(np.diag(core_points_mask) == 1)[0]]
    ship = SHiP(data=core_points, treeType="DCTree")
    ship_labels = ship.fit_predict(power=2, partitioningMethod="ThresholdElbow")
    original_labels[np.where(np.diag(core_points_mask) == 1)[0]] = torch.FloatTensor(ship_labels)
    eval_tracker = update_eval_tracker(gt_labels, original_labels.numpy(), eval_tracker)

    labels_over_iterations[:, 0] = original_labels
    i = 0
    for i in tqdm(range(training_iterations), 
                  desc = f"{dataset_name=},{k=},{percent=},{n_check=},{T_stop=}",
                  position = id,
                  disable = not use_tqdm
                  ):
        for batch, _, ids in trainloader:
            iteration_labels = labels_over_iterations[:, i][ids]
            batch_data = batch.to(device)
            loss, losses_tracker = loss_fn(model, batch_data, iteration_labels, losses_tracker, device)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # label assignment
        train_embedded_data, _ = encode_batchwise(testloader, model, device)
        if i >= n_check:
            crop = i
        else:
            crop = None
        high_confidence_labels = get_high_cof_labels(labels_over_iterations, n_check, crop)
        current_epoch_labels = labels_over_iterations[:, i]

        # 1 - consider points with low confidence as -1 so that they can be assigned to other clusters
        current_epoch_labels[~high_confidence_labels] = -1
        current_epoch_labels = labels_assignment_method(train_embedded_data, current_epoch_labels)
        previous_epoch_labels = labels_over_iterations[:, i]

        # 2 - prevent high confidence labels from changing
        current_epoch_labels[high_confidence_labels] = previous_epoch_labels[high_confidence_labels]

        # 3 - prevent labeled points from getting unlabeled
        labeled_to_unlabeled_points_mask = np.logical_and(current_epoch_labels == -1, previous_epoch_labels != -1)
        current_epoch_labels[labeled_to_unlabeled_points_mask] = previous_epoch_labels[labeled_to_unlabeled_points_mask]

        # 4 - Store the labels
        labels_over_iterations = np.hstack(
            (labels_over_iterations, current_epoch_labels.reshape(-1, 1))
        )
        eval_tracker = update_eval_tracker(gt_labels, current_epoch_labels, eval_tracker)

        # 5 - Early Stopping
        if np.all(high_confidence_labels):
            print(f"Early stopping after {i} iterations, all points are labeled confidently.")
            break
        if not_assigning_new_points(eval_tracker, T_stop):
            print(
                f"Early stopping after {i} iterations, the algorithm is not assigning any new points for {T_stop} iterations."
            )
            break

    # Save Trackers & model
    save_dict_as_json(losses_tracker.to_dict(), os.path.join(trackers_path, "loss_tracker.json"))
    save_dict_as_json(eval_tracker.to_dict(), os.path.join(trackers_path, "eval_tracker.json"))
    torch.save(model.state_dict(), deep_sync_model_path)
    return model, labels_over_iterations
