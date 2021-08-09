import torch

def _accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion):
    model.eval()
    acc1_val = 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(images)
            acc1 = _accuracy(output, target)
            n += images.size(0)
            acc1_val += float(acc1[0] * images.size(0))

    avg_acc1 = (acc1_val / n)
    return avg_acc1    

def lr_scheduler(epoch, optimizer, decay_eff=0.1, decayEpoch=[]):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch in decayEpoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_eff
            print('New learning rate is: ', param_group['lr'])
    return optimizer
        