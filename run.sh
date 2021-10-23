# 用于curriculum baseline++
CUDA_VISIBLE_DEVICES=0 python3 train_fourstages.py --no 1 --name fourstages --dataset cifar10 --model resnet --layers 56 --kl_div
CUDA_VISIBLE_DEVICES=0 python3 train_fourstages.py --no 2 --name fourstages --dataset cifar10 --model resnet --layers 56 
CUDA_VISIBLE_DEVICES=0 python3 train_tristages.py --no 0 --name tristages --dataset cifar10 --model resnet --layers 56 --kl_div
CUDA_VISIBLE_DEVICES=0 python3 train_tristages.py --no 2 --name tristages --dataset cifar10 --model resnet --layers 56
# CUDA_VISIBLE_DEVICES=0 python3 train_aug_raw_debug.py --no 4 --name aug_raw_record --dataset cifar100 --model wideresnet --layers 28 --widen-factor 10 --droprate 0.3 --cos_lr
# python3 train_curriculum_v1.py --no 3 --name curriculum --descrip tristages --dataset cifar100 --model wideresnet --layers 28 --widen-factor 10 --droprate 0.3 --cos_lr --aug_epoch 30 --aug_mode mixup
# CUDA_VISIBLE_DEVICES=0 python3 train_curriculum.py --no 0 --name curriculum --descrip non_aug --dataset cifar100 --model wideresnet --layers 28 --widen-factor 10 --droprate 0.3 --cos_lr --aug_epoch 1000
# CUDA_VISIBLE_DEVICES=0 python3 train_curriculum.py --no 1 --name curriculum --descrip full_aug --dataset cifar100 --model wideresnet --layers 28 --widen-factor 10 --droprate 0.3 --cos_lr --aug_epoch 0
# CUDA_VISIBLE_DEVICES=0 python3 train_curriculum.py --no 2 --name curriculum --descrip half_non_half_full_aug --dataset cifar100 --model wideresnet --layers 28 --widen-factor 10 --droprate 0.3 --cos_lr --aug_epoch 45
# CUDA_VISIBLE_DEVICES=0 python3 train_rand.py --no 2 --name ResRandaugtest --dataset cifar100 --model wideresnet --layers 28 --widen-factor 10 --droprate 0.3 --randaugment --cos_lr --N 2 --M 26 --rcutout 16 --Res --augment_prop 0.2
# CUDA_VISIBLE_DEVICES=0 python3 train_rand.py --no 3 --name ResRandaugtest --dataset cifar100 --model wideresnet --layers 28 --widen-factor 10 --droprate 0.3 --randaugment --cos_lr --N 2 --M 26 --rcutout 16 --Res --augment_prop 0.3
# CUDA_VISIBLE_DEVICES=0 python3 train_rand.py --no 4 --name ResRandaugtest --dataset cifar100 --model wideresnet --layers 28 --widen-factor 10 --droprate 0.3 --randaugment --cos_lr --N 2 --M 26 --rcutout 16 --Res --augment_prop 0.4
# CUDA_VISIBLE_DEVICES=0 python3 train_rand.py --no 5 --name ResRandaugtest --dataset cifar100 --model wideresnet --layers 28 --widen-factor 10 --droprate 0.3 --randaugment --cos_lr --N 2 --M 26 --rcutout 16 --Res --augment_prop 0.5
# CUDA_VISIBLE_DEVICES=0 python3 train_rand.py --no 6 --name ResRandaugtest --dataset cifar100 --model wideresnet --layers 28 --widen-factor 10 --droprate 0.3 --randaugment --cos_lr --N 2 --M 26 --rcutout 16 --Res --augment_prop 0.6
# CUDA_VISIBLE_DEVICES=0 python3 train_rand.py --no 7 --name ResRandaugtest --dataset cifar100 --model wideresnet --layers 28 --widen-factor 10 --droprate 0.3 --randaugment --cos_lr --N 2 --M 26 --rcutout 16 --Res --augment_prop 0.7
# CUDA_VISIBLE_DEVICES=0 python3 train_rand.py --no 8 --name ResRandaugtest --dataset cifar100 --model wideresnet --layers 28 --widen-factor 10 --droprate 0.3 --randaugment --cos_lr --N 2 --M 26 --rcutout 16 --Res --augment_prop 0.8
# CUDA_VISIBLE_DEVICES=0 python3 train_rand.py --no 9 --name ResRandaugtest --dataset cifar100 --model wideresnet --layers 28 --widen-factor 10 --droprate 0.3 --randaugment --cos_lr --N 2 --M 26 --rcutout 16 --Res --augment_prop 0.9