import wandb
import numpy as np
file = '/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/ISDA test/cifar100_wideresnet-28-10_ResRandaugtest/no_11_lambda_0_0.5_standard-Aug__dropout__randaugment__cos-lr__N=2_M=26_Res_CIFAR_Batch=128_rcutout=16/accuracy_epoch.txt'
data = np.loadtxt(file)
wandb.init(project="test-project", resume = 'must', id = '24d4qyx6')
for i,acc in enumerate(data):
    wandb.log({'test_accuracy2':acc,'acc_step': i})
    
# import wandb
# import numpy as np
# ids =['urltfrcn','1k4k9nj6','3cjbvfoa','1ivom637','2cpl6b4n','ni1audyt','2ms0ndtg','1u32bjuj','3qscz1n6']# 0.1 0.2,0.9
# i = 7
# file = '/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/ISDA test/cifar100_wideresnet-28-10_ResRandaugtest/no_'+str(i+1)+'_lambda_0_0.5_standard-Aug__dropout__randaugment__cos-lr__N=2_M=26_Res_CIFAR_Batch=128_rcutout=16/accuracy_epoch.txt'
# data = np.loadtxt(file)
# wandb.init(project="test-project", resume = 'must', id = ids[i])
# best_acc = 0
# for i,acc in enumerate(data):
#     if acc >best_acc:
#         best_acc = acc
#     # wandb.log({'test_accuracy2':acc,'acc_step': i})
# wandb.run.summary["best_accuracy"] = best_acc