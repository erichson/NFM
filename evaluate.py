import argparse
import os

import numpy as np
import torch
import torch.optim
import torch.utils.data

from src.get_data import getData
from utils import _get_calibration


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


def cls_noisy_validate(val_loader, model, time_begin=None):

    perturbed_test_accs = []
    for eps in [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35]:
        model.eval()
        acc1_val = 0
        n = 0
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                images += eps * torch.cuda.FloatTensor(images.shape).normal_()
                output = model(images)

                model_logits = output[0] if (type(output) is tuple) else output
                pred = model_logits.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                acc1_val += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                n += len(images)

        avg_acc1 = (acc1_val / n)
        perturbed_test_accs.append(avg_acc1)

    print(perturbed_test_accs)
    return perturbed_test_accs


def sp(image, amount):
    row, col = image.shape
    s_vs_p = 0.5
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    idx = np.random.choice(range(32 * 32), np.int(num_salt), False)
    out = out.reshape(image.size, -1)
    out[idx] = np.min(out)
    out = out.reshape(32, 32)

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    idx = np.random.choice(range(32 * 32), np.int(num_pepper), False)
    out = out.reshape(image.size, -1)
    out[idx] = np.max(out)
    out = out.reshape(32, 32)
    return out


def sp_wrapper(data, amount):
    np.random.seed(12345)
    for i in range(data.shape[0]):
        data_numpy = data[i, 0, :, :].data.cpu().numpy()
        noisy_input = sp(data_numpy, amount)
        data[i, 0, :, :] = torch.tensor(noisy_input).float().to('cuda')

    return data


def cls_sp_validate(val_loader, model, time_begin=None):

    perturbed_test_accs = []
    for eps in [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35]:
        model.eval()
        acc1_val = 0
        n = 0
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                images = sp_wrapper(images, eps)
                output = model(images)

                model_logits = output[0] if (type(output) is tuple) else output
                pred = model_logits.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                acc1_val += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                n += len(images)

        avg_acc1 = (acc1_val / n)
        perturbed_test_accs.append(avg_acc1)

    print(perturbed_test_accs)
    return perturbed_test_accs


def main(folder, eval_fn=cls_sp_validate):

    models = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    models = sorted(models)
    print(models)

    _, test_loader = getData(name='cifar10', train_bs=128, test_bs=1024)

    info_dict = {}

    for index, m in enumerate(models):
        model = torch.load(folder + m)

        #print("Beginning noisy evaluation")
        print(m)
        model.eval()
        #acc1 = cls_validate(test_loader, model)
        #_ = cls_noisy_validate(test_loader, model)
        result_m = eval_fn(test_loader, model)
        info_dict[m] = result_m

    return info_dict


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
