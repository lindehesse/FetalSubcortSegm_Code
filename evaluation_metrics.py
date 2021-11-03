
import numpy as np
import surface_distance


def get_surface_metrics(prediction, labels):
    """[summary]

    Args:
        prediction ([type]): [B,C,X,Y,Z]
        labels ([type]): [B,C, X,Y,Z] boundary labels

    returns:
        [B,C, metrics]
    """

    names = ['av_distance', 'haussdorf_95',
             'surface_dice']

    evals_array = np.zeros((
        prediction.shape[0], prediction.shape[1], len(names)))

    for batchnum in range(prediction.shape[0]):
        for channel in range(prediction.shape[1]):
            pred = prediction[batchnum, channel].astype(int)
            lab = labels[batchnum, channel].astype(int)

            dist = surface_distance.compute_surface_distances(
                pred, lab, spacing_mm=(0.6, 0.6, 0.6))

            av_surfdist = surface_distance.compute_average_surface_distance(
                dist)
            haussdorf = surface_distance.compute_robust_hausdorff(dist, 95)
            overlap = surface_distance.compute_surface_overlap_at_tolerance(
                dist, 0.6)

            evals = [(av_surfdist[0] + av_surfdist[1]) / 2,
                     haussdorf, (overlap[0] + overlap[1]) / 2]

            evals_array[batchnum, channel] = evals

    return names, evals_array
