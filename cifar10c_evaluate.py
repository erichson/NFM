import argparse
import os

import numpy as np
import torch
import torch.optim
import torch.utils.data

from get_data import NOISE_TYPES, SEVERITIES
from src.get_data import getData
from utils import _get_calibration


def main(folder, eval_fn=cls_sp_validate):

    models = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    models = sorted(models)
    print(models)

    _, test_loader = getData(name='cifar10', train_bs=128, test_bs=1024)

    info_dict = {}

    for index, m in enumerate(models):
        model = torch.load(folder + m)

        print(m)
        model.eval()
        result_m = eval_fn(test_loader, model)
        info_dict[m] = result_m

    return info_dict


def main_cifar10c(folder, eval_fn=cls_sp_validate):

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
                result_m = eval_fn(test_loader, model)
                results[m][noise][severity] = result_m

    return results


def get_calibration(data_loader, model, debug=False):
    model.eval()
    calibration = None
    preds_mean = None
    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)  # TODO: one hot or class number? We want class number

            output = model(images)

            model_logits = output[0] if (type(output) is tuple) else output
            preds = model_logits.data.mean(0, keepdim=True)  # .max(1, keepdim=True)[1]  # get the index of the max log-probability
            if preds_mean = None:
                preds_mean = preds
                continue
            preds_mean = torch.cat((preds_mean, preds)) # TODO: are these unnormalized log probabilitiies? We want normalized probabilities

    reliability = _get_calibration(target, preds_mean, debug=False)
    return reliability


def main_plot(folder, save_dir):
    models = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    models = sorted(models)
    for index, m in enumerate(models):
        print(f'model: {m} ({index + 1}/{len(models)})')
        plot_reliability(save_dir, os.path.basename(m))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Noisy Feature Mixup")
    parser.add_argument("--dir", type=str, default='cifar10_models/', required=True, help='model dir')
    parser.add_argument("--save_dir", default='./', help='if plotting, save here')
    parser.add_argument("--plot", action="store_true", default=False, help="whether or not to plot")

    args = parser.parse_args()

    results_dict = main(args.dir, get_calibration)
    with open(os.path.join(args.save_dir, 'reliability.npy'), 'wb') as f:
        np.save(f, results_dict)
        print(f'saved results successfully')

    if args.plot:
        main_plot(args.dir, args.save_dir)
        print(f'plots saved in {args.save_dir}')
