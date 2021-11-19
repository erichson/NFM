import argparse
import os

import numpy as np
import torch
import torch.optim
import torch.utils.data
from torch import nn

from os import set_blocking
import pickle
import pandas as pd
import seaborn as sns

from torch.nn import functional as F


from src.get_data import NOISE_TYPES, SEVERITIES
from src.get_data import getData

import collections




def cls_validate(val_loader, model, time_begin=None):
    model.eval()
    acc1_val = 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(images)

            model_logits = output[0] if (type(output) is tuple) else output
            pred = model_logits.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            acc1_val += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            n += len(images)

    avg_acc1 = (acc1_val / n)
    return avg_acc1


##### averaging across corruptions #####
PLOT_SEVERITIES = [0, 1, 2, 3, 4, 5]
SEVERITIES = [1, 2, 3, 4, 5]
PLOT_CORRUPTIONS = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
from src.get_data import NOISE_TYPES

print(NOISE_TYPES)
NOISE_TYPES_LABEL = NOISE_TYPES
NOISE_TYPES = ['gaussian_noise', 'jpeg_compression', 'impulse_noise',  'shot_noise', 'snow', 'speckle_noise']
NOISE_TYPES_LABEL = ['gaussian', 'jpeg', 'impulse',  'shot',  'snow', 'speckle']



def severity_vs_error(info_corrupt):
    # for one model: avg cross all noise types for each severity level
    # x: 0 - 5, y: error (%)
    mean_corrupt_error = []
    std_corrupt_error = []

    for severity in SEVERITIES:
        corrupt_error, corrupt_ece = [], []
        for noise in NOISE_TYPES:
            noise = str(noise)
            _corrupt_error = info_corrupt[noise][severity]#['acc']
            corrupt_error.append(_corrupt_error)
        mean_corrupt_error.append(np.mean(corrupt_error))
        std_corrupt_error.append(np.std(corrupt_error))

    assert len(mean_corrupt_error) == len(SEVERITIES)
    assert len(std_corrupt_error) == len(SEVERITIES)
    return mean_corrupt_error, std_corrupt_error


def type_vs_error(info_corrupt, severity):
    # for one model: avg cross all noise types for each severity level
    # x: 0 - 5, y: error (%)
    corrupt_error = []
    for noise in NOISE_TYPES:
        noise = str(noise)
        _corrupt_error = info_corrupt[noise][severity]#['acc']
        _corrupt_ece = info_corrupt[noise][severity]#['ece']
        corrupt_error.append(_corrupt_error)
    print(len(corrupt_error))
    assert len(corrupt_error) == len(NOISE_TYPES)
    return corrupt_error


def plot_cifar10c(save_dir, models, legend_loc="lower left", name='sev_acc', _xticks=["1", "2", "3", "4", "5"]):

    puzzelmix = "#bf5b17"
    manifold = "#8b1c5a"
    mixup = "#3182bd"
    baseline = "black"
    cutmix = "#984ea3"
    nfm = "#de2d26"
    nfm2 = '#31a354'


    models = ['Baseline.pt', 'CutMix.pt', 'Manifold.pt', 'Mixup.pt', 'PuzzelMix.pt', 'NFM.pt', 'nfm2.pt']

    legend = {0: 'Baseline', 1: 'CutMix', 2: 'M. Mixup', 3: 'Mixup',
            4: 'PuzzelMix', 5: 'NFM', 6: 'NFM (*)'}

    print(models)
    N = len(models)
    bar_width = 0.12

    # ax.set_prop_cycle(color=["2e1e3b", "#413d7b", "#37659e", "#348fa7", "#40b7ad", "#8bdab2"])
    x = np.arange(len(SEVERITIES))
    
    with open(f'{save_dir}/cifar10c.npy', 'rb') as f:
        info_corrupt = np.load(f, allow_pickle=True).item()
    
    # plot error first
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6.5))
    ax.set_prop_cycle(color=[baseline, cutmix, manifold, mixup, puzzelmix, nfm, nfm2])

    for i, model in enumerate(range(N)):
        _model = models[model]
        _model_info = info_corrupt[_model]
        y, yerr = severity_vs_error(_model_info)
        # ax.errorbar(x, y, yerr=yerr, label=_model, elinewidth=2)
        ax.bar(np.arange(len(SEVERITIES)) + i * bar_width, np.array(y) * 100, width=bar_width, edgecolor='white', label=legend[i])
        # ax.errorbar(x, y, yerr=yerr, label=_model, fmt='o', elinewidth=2)

    plt.legend(loc='upper right', fontsize=13)
    #ax.set_ylabel("Accuracy (%)")
    #ax.set_xlabel("Severity Level")
    plt.xticks([r + 3 * bar_width for r in range(len(SEVERITIES))], _xticks)
    plt.ylim([35, 100])
    plt.tick_params(axis='y', labelsize=26) 
    plt.tick_params(axis='x', labelsize=26) 
    plt.tight_layout()
    plt.savefig("%s/cifar10c/%s-acc.pdf" % (save_dir, name))
    plt.savefig("%s/cifar10c/%s-acc.png" % (save_dir, name))
    # plt.savefig("%s/%s.svg" % (plot_dir, name), format='svg')
    plt.clf()



    # plot error for different noie types
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6.5))
    ax.set_prop_cycle(color=[baseline, cutmix, manifold, mixup,  puzzelmix, nfm, nfm2])

    for i, model in enumerate(range(N)):
        _model = models[model]
        _model_info = info_corrupt[_model]
        y = type_vs_error(_model_info, 3)
        print(y)
        ax.bar(np.arange(len(NOISE_TYPES)) + i * bar_width, np.array(y) * 100, width=bar_width, edgecolor='white', label=legend[i])

    #plt.legend(loc='upper right', prop={'size': 13})
    #ax.set_ylabel("Accuracy (%)")
    #ax.set_xlabel("Noise Type")
    plt.xticks([r + 3 * bar_width for r in range(len(NOISE_TYPES))], NOISE_TYPES_LABEL, rotation=45)
    plt.ylim([20, 100])
    plt.tick_params(axis='y', labelsize=26) 
    plt.tick_params(axis='x', labelsize=20) 
    plt.tight_layout()
    plt.savefig("%s/cifar10c/%s-acc_type.pdf" % (save_dir, name))
    plt.savefig("%s/cifar10c/%s-acc_type.png" % (save_dir, name))
    # plt.savefig("%s/%s.svg" % (plot_dir, name), format='svg')
    plt.clf()
    print(NOISE_TYPES)

def main_cifar10c(folder, save_dir):
    os.makedirs(os.path.join(args.save_dir, 'cifar10c'), exist_ok=True)

    models = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    models = sorted(models)
    print(models)

    results = collections.defaultdict(dict)

    for index, m in enumerate(models):
        model = torch.load(folder + m)
        print(m)
        model.eval()
        for noise in NOISE_TYPES:
            results[m][noise] = collections.defaultdict(dict)
            for severity in SEVERITIES:
                _, test_loader = getData(name='cifar10c', train_bs=128, test_bs=1024, severity=severity, noise=noise)
                result_m = cls_validate(test_loader, model)
                results[m][noise][severity] = result_m
        with open(f"{save_dir}/cifar10c/robust_{m}.pickle", "wb") as f:
            np.save(f, result_m)

    return results

def main_plot_cifar10c(folder, save_dir):
    models = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    models = sorted(models)
    plot_cifar10c(save_dir, models)
    print(save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Noisy Feature Mixup")
    parser.add_argument("--dir", type=str, default='cifar10_models/', required=False, help='model dir')
    parser.add_argument("--save_dir", default='.', help='if plotting, save here. Call this the exp id.')
    parser.add_argument("--plot", action="store_true", default=False, help="whether or not to plot calibration, else just evaluates on cifar10c")
    parser.add_argument("--batch_size", default=128, type=int, help='batch size')
    args = parser.parse_args()

    test_batch_size = args.batch_size
    os.makedirs(args.save_dir, exist_ok=True)

    if args.plot:
        main_plot_cifar10c(args.dir, args.save_dir)
        print(f'plots saved in {args.save_dir}')
    else:
         results_dict2 = main_cifar10c(args.dir, args.save_dir)
         with open(os.path.join(args.save_dir, 'cifar10c.npy'), 'wb') as f:
             np.save(f, results_dict2)
             print(f'saved results successfully')

