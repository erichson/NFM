import argparse
import os

import numpy as np
import torch
import torch.optim
import torch.utils.data

from src.get_data import NOISE_TYPES, SEVERITIES
from src.get_data import getData
from utils import _get_calibration, plot_reliability, plot_cifar10c

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

def main_calibration(folder, save_dir, n_bins=11):

    models = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    models = sorted(models)
    print(models)

    _, test_loader = getData(name='cifar10', train_bs=1, test_bs=test_batch_size)

    info_dict = {}

    os.makedirs(os.path.join(args.save_dir, 'ece'), exist_ok=True)

    for index, m in enumerate(models):
        model = torch.load(folder + m)

        print(m)
        model.eval()
        result_m = get_calibration(test_loader, model, n_bins)
        info_dict[m] = result_m
        with open(f"{save_dir}/ece/cache_ece_{m}.pickle", "wb") as f:
            np.save(f, result_m)

    return info_dict

def get_calibration(data_loader, model, debug=False, n_bins=11):
    model.eval()
    mean_conf, mean_acc = [], []
    ece = []
    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)  # TODO: one hot or class number? We want class number

            output = model(images)

            model_logits = output[0] if (type(output) is tuple) else output
            preds = torch.nn.functional.softmax(model_logits.data)  # .max(1, keepdim=True)[1]  # get the index of the max log-probability
            print(preds.shape)
            # preds = model_logits.data.mean(0, keepdim=True)  # .max(1, keepdim=True)[1]  # get the index of the max log-probability
            calibration_dict = _get_calibration(target, preds, debug=False, num_bins=n_bins)
            mean_conf.append(calibration_dict['reliability_diag'][0])
            mean_acc.append(calibration_dict['reliability_diag'][1])

            ece.append(calibration_dict['ece'])

    calibration_results = {'reliability_diag': (torch.vstack(mean_conf), torch.vstack(mean_acc)), 'ece': torch.vstack(ece)}
    return calibration_results


def main_plot_reliability(folder, save_dir):
    models = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    models = sorted(models)
    for index, m in enumerate(models):
        print(f'model: {m} ({index + 1}/{len(models)})')
        plot_reliability(save_dir, os.path.basename(m))


def main_plot_cifar10c(folder, save_dir):
    models = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    models = sorted(models)
    plot_cifar10c(save_dir, models)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Noisy Feature Mixup")
    parser.add_argument("--dir", type=str, default='cifar10_models/', required=False, help='model dir')
    parser.add_argument("--save_dir", default='.', help='if plotting, save here. Call this the exp id.')
    parser.add_argument("--plot_calib", action="store_true", default=False, help="whether or not to plot calibration, else just evaluates on cifar10c")
    parser.add_argument("--batch_size", default=128, type=int, help='batch size')

    args = parser.parse_args()

    test_batch_size = args.batch_size
    os.makedirs(args.save_dir, exist_ok=True)

    if args.plot_calib:
        # main_plot_reliability(args.dir, args.save_dir)
        main_plot_cifar10c(args.dir, args.save_dir)
        print(f'plots saved in {args.save_dir}')
    else:
        results_dict1 = main_calibration(args.dir, args.save_dir)
        with open(os.path.join(args.save_dir, 'reliability.npy'), 'wb') as f:
            np.save(f, results_dict1)
            print(f'saved results successfully')
        # results_dict2 = main_cifar10c(args.dir, args.save_dir)
        # with open(os.path.join(args.save_dir, 'cifar10c.npy'), 'wb') as f:
        #     np.save(f, results_dict2)
        #     print(f'saved results successfully')

