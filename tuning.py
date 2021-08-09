

## Train Pre Activated ResNets on CIFAR10
export CUDA_VISIBLE_DEVICES=0; python train_cifar.py --arch preactresnet18 --alpha 0.0 --add_noise_level 0.0 --mult_noise_level 0.0 --seed 1
export CUDA_VISIBLE_DEVICES=1; python train_cifar.py --arch preactresnet18 --alpha 1.0 --add_noise_level 0.0 --mult_noise_level 0.0 --seed 1
export CUDA_VISIBLE_DEVICES=2; python train_cifar.py --arch preactresnet18 --alpha 1.0 --add_noise_level 0.15 --mult_noise_level 0.1 --seed 1
export CUDA_VISIBLE_DEVICES=3; python train_cifar.py --arch preactresnet18 --alpha 1.0 --add_noise_level 0.4 --mult_noise_level 0.2 --manifold_mixup 1 --seed 1

export CUDA_VISIBLE_DEVICES=1; python train_cifar.py --arch preactresnet18 --alpha 2.0 --add_noise_level 0.0 --mult_noise_level 0.0 --manifold_mixup 1 --seed 1
export CUDA_VISIBLE_DEVICES=0; python train_cifar.py --arch preactresnet18 --alpha 0.00000001 --add_noise_level 0.15 --mult_noise_level 0.0 --manifold_mixup 0 --seed 1

