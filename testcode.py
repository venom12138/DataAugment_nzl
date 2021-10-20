import torchvision.datasets as datasets
from mydataset import myCIFAR10
import numpy as np
a =np.random.randn(1,5,3,3)
print(a)
print(np.squeeze(a).shape)
print(a.shape)