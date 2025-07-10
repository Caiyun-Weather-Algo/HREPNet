import numpy as np 
from scipy.stats import rankdata


def get_confusion_table(thresholds, target, output):
    counts = np.zeros((len(thresholds), 4))
    for j, v in enumerate(thresholds):
        counts[j,0] += ((target >= v) & (output >= v)).sum() # hit
        counts[j,1] += ((target >= v) & (output < v)).sum() # missing
        counts[j,2] += ((target < v) & (output >= v)).sum() # false
        counts[j,3] += ((target < v) & (output < v)).sum()
    return counts


def rainfall_metrics(confusion_table):
    """common metrics for rainfall evaluation

    Args:
        confusion_table (_type_): [n_thresholds, 4]
                                  -[:, 0], hits
                                  -[:, 1], missing
                                  -[:, 2], false alarm

    Returns:
        csi: 
        pod:
        far:
    """
    csi = confusion_table[:, 0]/(confusion_table[:, :3].sum(axis=1))
    pod = confusion_table[:, 0]/(confusion_table[:, 0] + confusion_table[:, 1])
    far = confusion_table[:, 2]/(confusion_table[:, 0] + confusion_table[:, 2])
    return csi, pod, far


def rankz(obs,ensemble,mask=None):
    ''' Parameters
    ----------
    obs : array of observations 
    ensemble : array of ensemble, with the first dimension being the 
        ensemble member and the remaining dimensions being identical to obs
    mask : boolean mask of shape of obs, with zero/false being where grid cells are masked.  

    Returns
    -------
    histogram data for ensemble.shape[0] + 1 bins. 
    The first dimension of this array is the height of 
    each histogram bar, the second dimension is the histogram bins. 
         '''
    if mask is not None:
        mask=np.bool_(mask)

        obs=obs[mask]
        ensemble=ensemble[:,mask]
    
    combined=np.vstack((obs[np.newaxis],ensemble))

    # print('computing ranks')
    ranks=np.apply_along_axis(lambda x: rankdata(x,method='min'),0,combined)

    # print('computing ties')
    ties=np.sum(ranks[0]==ranks[1:], axis=0)
    ranks=ranks[0]
    tie=np.unique(ties)

    for i in range(1,len(tie)):
        index=ranks[ties==tie[i]]
        # print('randomizing tied ranks for ' + str(len(index)) + ' instances where there is ' + str(tie[i]) + ' tie/s. ' + str(len(tie)-i-1) + ' more to go')
        ranks[ties==tie[i]]=[np.random.randint(index[j],index[j]+tie[i]+1,tie[i])[0] for j in range(len(index))]

    return np.histogram(ranks, bins=np.linspace(0.5, combined.shape[0]+0.5, combined.shape[0]+1))


if __name__ == "__main__":
    counts = [[2, 5], [3, 9]]
    print(rainfall_metrics(counts))