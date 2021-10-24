# CUDA_VISIBLE_DEVICES=0 python3 train_tristages.py --no 4 --name tristages --dataset cifar10 --model resnet --layers 164
# CUDA_VISIBLE_DEVICES=0 python3 train_tristages.py --no 5 --name tristages --dataset cifar10 --model resnet --layers 1001
CUDA_VISIBLE_DEVICES=0 python3 fine_tune.py --no 3 --name finetune --dataset cifar10 --model resnet --layers 56 --resume /home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/ISDA\ test/cifar10_resnet-56_tristages/no_2_lambda_0_0.5_standard-Aug_/checkpoint/checkpoint.pth.tar # --optimizer_resume
# CUDA_VISIBLE_DEVICES=0 python3 fine_tune.py --no 1 --name finetune --dataset cifar10 --model resnet --layers 56 --resume /home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/ISDA\ test/cifar10_resnet-56_tristages/no_2_lambda_0_0.5_standard-Aug_/checkpoint/checkpoint.pth.tar
# CUDA_VISIBLE_DEVICES=0 python3 train_feature.py --no 0 --name baseline --dataset cifar10 --model resnet --layers 56
