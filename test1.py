import torch 
from einops import repeat, rearrange 

t1=torch.arange(8).view(2,2,2)

t2=t1.expand(4,-1,-1)
t3=t1.expand(4,2,2)

breakpoint()