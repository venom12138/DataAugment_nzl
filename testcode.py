import torchvision.datasets as datasets
from mydataset import myCIFAR10
import numpy as np
import mmap
import sys
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

# wandb.init(project="test-project")
# train_step = 0
# test_step = 0
# wandb.run.name = 'testcommit'#+str(time1.tm_year)+str(time1.tm_mon)+str(time1.tm_mday)+str(time1.tm_hour)+str(time1.tm_min)
# for i in range(100):
#     for j in range(500):
#         wandb.log({'j':j},step=train_step)
#         train_step += 1
#     for k in range(1000):
#         wandb.log({'k':k},step=test_step)
#         test_step += 1
#     wandb.log({'i':i})
# Example of target with class probabilities
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
print(target)
output = F.cross_entropy(input, target)
