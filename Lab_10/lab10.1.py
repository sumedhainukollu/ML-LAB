import math
from collections import Counter

features = [1,2,3,4]
targets=["yes","no","yes","no"]

def entropy(features,targets):
    tot=len(targets)
    count=Counter(targets)
    entropy=0.0
    for c in count.values():
        p=c/tot
        entropy-=p*math.log2(p)
    return entropy



print(entropy(features,targets))