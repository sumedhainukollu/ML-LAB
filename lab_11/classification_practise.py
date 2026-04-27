import math

import numpy as np
from collections import Counter

def entropy(y):
    ent=0.0
    tot=len(y)
    count=Counter(y)
    for c in count.values():
        p=c/tot
        ent-=p * math.log2(p)
def split_data(x,y,threshold):
    left_y,left_x=[],[]
    right_y,right_x=[]


