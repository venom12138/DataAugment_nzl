import torchvision.datasets as datasets
from mydataset import myCIFAR10
import numpy as np
import mmap
import sys

with open("/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/save_feature/feature1.npy", "r+b") as f:
    # memory-map the file, size 0 means whole file
    mmap = mmap.mmap(f.fileno(), 0)
    features = np.frombuffer(mmap)
    features2 = np.load("/home/yu-jw19/venom/ISDA-for-Deep-Networks/CIFAR/save_feature/feature1.npy",'r')  
    print(sys.getsizeof(features))
    print(sys.getsizeof(features2[0,1,1,1]))