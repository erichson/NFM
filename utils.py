import pickle

import numpy as np
import torch


def _get_calibration(y, p_mean, num_bins=20, axis=-1, individual=False, attributes=False, debug=False):
    """Compute the calibration.
    Modified from: https://github.com/xwinxu/bayesian-sde/blob/main/brax/utils/utils.py
    References:
    https://arxiv.org/abs/1706.04599
    https://arxiv.org/abs/1807.00263
    Args:
      y: class labels (binarized)
      p_mean: numpy array, size (batch_size, num_classes)
            containing the mean output predicted probabilities
      num_bins: number of bins
      axis: Axis of the labels
      individual: Return ECE for individual labels
      attributes: Return micro average of ECE scores across attributes
      debug: Return debug information
    Returns:
      cal: a dictionary
      {
        reliability_diag: realibility diagram
        ece: Expected Calibration Error
        nb_items: nb_items_bin/np.sum(nb_items_bin)
    }
    """
    assert not (individual and micro)

    if attributes:
        y = y.reshape(-1)
        p_mean = p_mean.reshape(-1, p_mean.shape[2])

    # compute predicted class and its associated confidence (probability)
    conf, class_pred = p_mean.max(axis)

    assert y.shape[0] == p_mean.shape[0]
    assert len(p_mean.shape) == len(y.shape) + 1
    assert p_mean.shape[1] > 1  # features
    assert len(y.shape) == 1

    tau_tab = torch.linspace(0, 1, num_bins + 1, device=p_mean.device)

    conf = conf[None]
    for _ in range(len(y.shape)):
        tau_tab = tau_tab.unsqueeze(-1)

    sec = (conf < tau_tab[1:]) & (conf >= tau_tab[:-1])

    nb_items_bin = sec.sum(1)

    mean_conf = (conf * sec).sum(1) / nb_items_bin
    acc_tab = ((class_pred == y)[None] * sec).sum(1) / nb_items_bin

    _weights = nb_items_bin.float() / nb_items_bin.sum(0)
    ece = ((mean_conf - acc_tab).abs() * _weights).nansum(0)
    if not individual:
        # pytorch doesn't have a built in nanmean
        ece[ece.isnan()] = 0
        ece = ece.mean(0)

    cal = {
        'reliability_diag': (mean_conf, acc_tab),
        'ece': ece,
        '_weights': _weights,
    }
    if debug:
        cal.update({'conf': conf, 'sec': sec, 'tau_tab': tau_tab, 'acc_tab': acc_tab, 'p_mean': p_mean})
    return cal


def _plot_reliability(save_dir, m, reliability_diag, ece, freq, dset_label_info):
    plt.clf()

    with open(f"{save_dir}/cache_ece_{m}.pickle", "wb") as f:
        d = {"reliability_diag": reliability_diag, "ece": ece, "freq": freq, "dset_label_info": dset_label_info}
        pickle.dump(d, f, protocol=4)

    conf, acc = reliability_diag
    conf[conf.isnan()] = 0
    acc[acc.isnan()] = 0

    num_bins = 20
    tau_tab = torch.linspace(0, 1, num_bins + 1)
    binned_confs = tau_tab[torch.searchsorted(tau_tab, conf)]

    font = {'family': 'normal', 'size': 20}
    plt.rc('font', **font)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    plt.bar(binned_confs, acc, color="blue", linewidth=.1, edgecolor="black", align="edge", width=1 / num_bins)
    plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), "--")
    plt.annotate(f"ECE: {ece * 100:.2f}%",
                 xy=(140, 240),
                 xycoords='axes points',
                 size=20,
                 ha='right',
                 va='top',
                 bbox=dict(boxstyle='round', fc="#f6b2b2", color="#f6b2b2"))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks([.0, .5, 1.])
    plt.yticks([.0, .5, 1.])
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('')
    plt.gcf().subplots_adjust(right=.95, left=.14, bottom=.15)
    plt.savefig(f"{save_dir}/ece_{m}.png", dpi=200)
    plt.savefig(f"{save_dir}/ece_{m}.pdf")
    plt.close(fig)


def plot_reliability(save_dir, m, reliability_diag=None, ece=None, freq=None, dset_label_info=None, from_cache=True):
    if from_cache:
        with open(f"{save_dir}/reliability.npy", "rb") as f:
            d = np.load(f, allow_pickle=True).item()
    else:
        d = {"reliability_diag": reliability_diag, "ece": ece, "freq": freq, "dset_label_info": dset_label_info}

    _plot_reliability(save_dir, m, **d)


def plot_cifar10c(save_dir, m, model_to_noise_to_severity_dict):
    # TODO
    return
