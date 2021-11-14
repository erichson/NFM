from os import set_blocking
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _get_calibration(y, p_mean, num_bins=10, axis=-1, individual=False, attributes=False, debug=False):
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

    mean_conf = (conf * sec).sum(1) / (nb_items_bin+1e-8)
    acc_tab = ((class_pred == y)[None] * sec).sum(1) / (nb_items_bin+1e-8)

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


def _plot_reliability(save_dir, m, reliability_diag, ece, n_bins=11): #, freq, dset_label_info):
    plt.clf()

    # with open(f"{save_dir}/cache_ece_{m}.pickle", "wb") as f:
    #     d = {"reliability_diag": reliability_diag, "ece": ece} # , "freq": freq, "dset_label_info": dset_label_info}
    #     pickle.dump(d, f, protocol=4)

    conf, acc = reliability_diag
    conf[conf.isnan()] = 0
    acc[acc.isnan()] = 0

    num_bins = n_bins
    tau_tab = torch.linspace(0, 1, num_bins + 1)
    fix_conf = torch.linspace(0, 1, num_bins)

    # tau_tab = tau_tab.to(conf.device)
    conf = conf.cpu()
    binned_confs = tau_tab[torch.searchsorted(tau_tab, conf)]

    font = {'family': 'normal', 'size': 20}
    plt.rc('font', **font)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    binned_confs = binned_confs.numpy()
    binned_accs = acc.cpu().numpy()
    # plt.bar(binned_confs.mean(0), binned_accs.mean(0), yerr=binned_accs.std(0), color="blue", linewidth=.1, edgecolor="black", align="edge", width=1 / num_bins)
    plt.bar(fix_conf, binned_accs.mean(0), color="blue", linewidth=.1, edgecolor="black", align="edge", width= 0.2 )
    # plt.bar(binned_confs.data.numpy(), acc.data.cpu().numpy(), color="blue", edgecolor="black", align="edge", width=1 / num_bins)
    plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), "--")
    plt.annotate(f"ECE: {ece.cpu().numpy().mean() * 100:.2f}%",
                 xy=(140, 240),
                 xycoords='axes points',
                 size=20,
                 ha='right',
                 va='top',
                 bbox=dict(boxstyle='round', fc="#f6b2b2", color="#f6b2b2"))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks([.0,0.2,0.4,0.6,0.8,1.])
    plt.yticks([.0,0.2,0.4,0.6,0.8,1.])
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('')
    plt.gcf().subplots_adjust(right=.95, left=.14, bottom=.15)
    plt.savefig(f"{save_dir}/ece/ece_calibration_{m}.png", dpi=200)
    plt.savefig(f"{save_dir}/ece/ece_calibration_{m}.pdf")
    plt.close(fig)


def plot_reliability(save_dir, m, reliability_diag=None, ece=None, freq=None, dset_label_info=None, from_cache=True):
    if from_cache:
        # with open(f"{save_dir}/reliability.npy", "rb") as f:
        with open(f"{save_dir}/ece/cache_ece_{m}.pickle", "rb") as f:
            d = np.load(f, allow_pickle=True).item()
    else:
        print('No reliability exists!')
        # d = {"reliability_diag": reliability_diag, "ece": ece} # , "freq": freq, "dset_label_info": dset_label_info}
        # with open(f"{save_dir}/cache_ece_{m}.pickle", "wb") as f:
        #     np.save(f, d)

    _plot_reliability(save_dir, m, **d)

##### averaging across corruptions #####
PLOT_SEVERITIES = [0, 1, 2, 3, 4, 5]
SEVERITIES = [1, 2, 3, 4, 5]
PLOT_CORRUPTIONS = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
from src.get_data import NOISE_TYPES

def severity_vs_error(info_corrupt):
    # for one model: avg cross all noise types for each severity level
    # x: 0 - 5, y: error (%)
    mean_corrupt_error = []
    std_corrupt_error = []
    mean_corrupt_ece = []
    std_corrupt_ece = []
    for severity in SEVERITIES:
        corrupt_error, corrupt_ece = [], []
        for noise in NOISE_TYPES:
            _corrupt_error = info_corrupt[noise][severity]['acc']
            _corrupt_ece = info_corrupt[noise][severity]['ece']
            corrupt_error.append(_corrupt_error)
            corrupt_ece.append(_corrupt_ece)
        mean_corrupt_error.append(np.mean(corrupt_error))
        std_corrupt_error.append(np.std(corrupt_error))
        mean_corrupt_ece.append(np.mean(corrupt_ece))
        mean_corrupt_ece.append(np.std(corrupt_ece))
    assert len(mean_corrupt_error) == len(SEVERITIES)
    assert len(std_corrupt_error) == len(SEVERITIES)
    assert len(mean_corrupt_ece) == len(SEVERITIES)
    assert len(std_corrupt_ece) == len(SEVERITIES)
    return mean_corrupt_error, std_corrupt_error, mean_corrupt_ece, std_corrupt_ece

def plot_cifar10c(save_dir, models, legend_loc="lower left", name='sev_acc', _xticks=["1", "2", "3", "4", "5"]):

    puzzmix = "red"
    augmix = "blue"
    manimix_base = "green"
    nfm_a02_1_04_02 = "violet"
    nfm_1e8_1_04_02 = "navy"
    nfm_a1_0_0_0 = "coral"
    nfm_a1_1_0_0 = "orange"
    nfm_a1_1_04_02 = "grey"

    legend = {0: 'manifold', 1: 'nfm_a02_1_04_02', 2: 'nfm_a1_0_0_0', 3: 'nfm_a1_1_0_0',
              4: 'nfm_a1_1_04_02', 5: 'nfm_1e8_1_04_02', 6: 'augmix', 7: 'puzzmix'}

    print(models)
    # ['arch_preactresnet18_alpha_0.0_manimixup_0_addn_0.0_multn_0.0_seed_1.pt', 
    # 'arch_preactresnet18_alpha_0.2_manimixup_1_addn_0.4_multn_0.2_seed_4.pt', 
    # 'arch_preactresnet18_alpha_1.0_manimixup_0_addn_0.0_multn_0.0_seed_1.pt', 
    # 'arch_preactresnet18_alpha_1.0_manimixup_1_addn_0.0_multn_0.0_seed_1.pt', 
    # 'arch_preactresnet18_alpha_1.0_manimixup_1_addn_0.4_multn_0.2_seed_1.pt', 
    # 'arch_preactresnet18_alpha_1e-08_manimixup_0_addn_0.4_multn_0.0_seed_1.pt', 
    # 'augmix_seed_1.pt', 'puzzelmix_cifar10_seed_1.pt']
    N = len(models)
    bar_width = 0.1

    # ax.set_prop_cycle(color=["2e1e3b", "#413d7b", "#37659e", "#348fa7", "#40b7ad", "#8bdab2"])

    x = np.arange(len(SEVERITIES))
    
    with open(f'{save_dir}/cifar10c.npy', 'rb') as f:
        info_corrupt = np.load(f, allow_pickle=True).item()
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.set_prop_cycle(color=[manimix_base, nfm_a02_1_04_02, nfm_a1_0_0_0, nfm_a1_1_0_0, nfm_a1_1_04_02, nfm_1e8_1_04_02, augmix, puzzmix])

    for i, model in enumerate(range(N)):
        _model = models[model]
        _model_info = info_corrupt[_model]
        y, yerr, ece, ece_err = severity_vs_error(_model_info)
        # ax.errorbar(x, y, yerr=yerr, label=_model, elinewidth=2)
        ax.bar(np.arange(len(SEVERITIES)) + i * bar_width, np.array(y) * 100, width=bar_width, edgecolor='white', label=legend[i])
        # ax.errorbar(x, y, yerr=yerr, label=_model, fmt='o', elinewidth=2)

    plt.legend(loc=legend_loc, prop={'size': 13})
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Severity Level")
    plt.xticks([r + 3 * bar_width for r in range(len(SEVERITIES))], _xticks)
    plt.ylim([0, 100])

    plt.tight_layout()
    plt.savefig("%s/cifar10c/%s-acc.pdf" % (save_dir, name))
    plt.savefig("%s/cifar10c/%s-acc.png" % (save_dir, name))
    # plt.savefig("%s/%s.svg" % (plot_dir, name), format='svg')
    plt.clf()

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.set_prop_cycle(color=[manimix_base, nfm_a02_1_04_02, nfm_a1_0_0_0, nfm_a1_1_0_0, nfm_a1_1_04_02, nfm_1e8_1_04_02, augmix, puzzmix])

    for i, model in enumerate(range(N)):
        _model = models[model]
        _model_info = info_corrupt[_model]
        y, yerr, ece, ece_err = severity_vs_error(_model_info)
        # ax.errorbar(x, y, yerr=yerr, label=_model, elinewidth=2)
        ax.bar(np.arange(len(SEVERITIES)) + i * bar_width, np.array(ece) * 100, width=bar_width, edgecolor='white', label=legend[i])
        # ax.errorbar(x, y, yerr=yerr, label=_model, fmt='o', elinewidth=2)

    plt.legend(loc=legend_loc, prop={'size': 13})
    ax.set_ylabel("Expected Calibration Error (%)")
    ax.set_xlabel("Severity Level")
    plt.xticks([r + 3 * bar_width for r in range(len(SEVERITIES))], _xticks)
    plt.ylim([0, 100])

    plt.tight_layout()
    plt.savefig("%s/cifar10c/%s-ece.pdf" % (save_dir, name))
    plt.savefig("%s/cifar10c/%s-ece.png" % (save_dir, name))
    # plt.savefig("%s/%s.svg" % (plot_dir, name), format='svg')
    plt.clf()

    plt.close(fig)
    
    return
