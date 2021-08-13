import numpy as np
import torch
import os
import argparse
import timeit

from torch import nn
from torch.autograd import Variable

from src.get_data import getData
import src.cifar_models
from src.noisy_mixup import mixup_criterion
from src.tools import validate, lr_scheduler

#==============================================================================
# Training settings
#==============================================================================

parser = argparse.ArgumentParser(description='CIFAR10 Example')
#
parser.add_argument('--name', type=str, default='cifar10', metavar='N', help='dataset')
#
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
#
parser.add_argument('--test_batch_size', type=int, default=512, metavar='N', help='input batch size for testing (default: 1000)')
#
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 90)')
#
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
#
parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay value (default: 0.1)')
#
parser.add_argument('--lr_decay_epoch', type=int, nargs='+', default=[100, 150, 180], help='decrease learning rate at these epochs.')
#
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
#
parser.add_argument('--arch', type=str, default='preactresnet18', metavar='N', help='model name')
#
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 0)')
#
parser.add_argument('--alpha', type=float, default=0.0, metavar='S', help='for mixup')
#
parser.add_argument('--manifold_mixup', type=int, default=0, metavar='S', help='manifold mixup (default: 0)')
#
parser.add_argument('--add_noise_level', type=float, default=0.0, metavar='S', help='level of additive noise')
#
parser.add_argument('--mult_noise_level', type=float, default=0.0, metavar='S', help='level of multiplicative noise')
#
args = parser.parse_args()

#==============================================================================
# set random seed to reproduce the work
#==============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True

seed_everything(args.seed)

#==============================================================================
# get device
#==============================================================================
def get_device():
    """Get a gpu if available."""
    if torch.cuda.device_count()>0:
        device = torch.device('cuda')
        print("Connected to a GPU")
    else:
        print("Using the CPU")
        device = torch.device('cpu')
    return device

device = get_device()

#==============================================================================
# get dataset
#==============================================================================
train_loader, test_loader = getData(name=args.name, train_bs=args.batch_size, test_bs=args.test_batch_size)  


#==============================================================================
# get model
#==============================================================================
if args.name == 'cifar10': 
    num_classes=10
elif args.name == 'cifar100': 
    num_classes=100

print(num_classes)

model = src.cifar_models.__dict__[args.arch](num_classes=num_classes).cuda()


#==============================================================================
# Model summary
#==============================================================================
print(model)    
print('**** Setup ****')
print('Total params: %.2fK' % (sum(p.numel() for p in model.parameters())*10**-3))
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())*10**-6))
print('************')    

#==============================================================================
# setup optimizer
#==============================================================================
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)


#==============================================================================
# criterion
#==============================================================================
criterion = nn.CrossEntropyLoss().to(device)

#==============================================================================
# start training
#==============================================================================
count = 0
loss_hist = []
test_acc = []

t0 = timeit.default_timer()
for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    correct = 0.0
    total_num = 0
    
    for step, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)


        if args.alpha == 0.0:   
            outputs = model(inputs)
        else:
            outputs, targets_a, targets_b, lam = model(inputs, targets=targets, mixup_alpha=args.alpha,
                                                      manifold_mixup=args.manifold_mixup,
                                                      add_noise_level=args.add_noise_level,
                                                      mult_noise_level=args.mult_noise_level)
        
        if args.alpha>0:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)        


        optimizer.zero_grad()
        loss.backward()          
        optimizer.step() # update weights

        # compute statistics
        train_loss += loss.item()*targets.size()[0]
        total_num += targets.size()[0]
        _, predicted = outputs.max(1)
        
        if args.alpha>0:
            correct += lam * predicted.eq(targets_a.data).sum().item() + (1-lam) * predicted.eq(targets_b.data).sum().item()
        else:
            correct += predicted.eq(targets).sum().item()     

    train_loss = train_loss / total_num
    accuracy = correct / total_num
    test_accuracy = validate(test_loader, model, criterion)
    print('Epoch: ', epoch, '| train loss: %.3f' % train_loss, '| test acc.: %.3f' % test_accuracy)

    # schedule learning rate decay    
    optimizer = lr_scheduler(epoch, optimizer, decay_eff=args.lr_decay, decayEpoch=args.lr_decay_epoch)

print('total time: ', timeit.default_timer()  - t0 )

#==============================================================================
# store results
#==============================================================================
DESTINATION_PATH = args.name + '_models/'
OUT_DIR = os.path.join(DESTINATION_PATH, f'arch_{args.arch}_alpha_{args.alpha}_manimixup_{args.manifold_mixup}_addn_{args.add_noise_level}_multn_{args.mult_noise_level}_seed_{args.seed}')

if not os.path.isdir(DESTINATION_PATH):
        os.mkdir(DESTINATION_PATH)
torch.save(model, OUT_DIR+'.pt')