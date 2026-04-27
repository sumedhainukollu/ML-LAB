import math

import numpy as np
from collections import Counter



def entropy(targets):
    tot=len(targets)
    count=Counter(targets)
    entropy=0.0
    for c in count.values():
        p=c/tot
        entropy-= p * math.log2(p)
    return entropy

def split_data(features,targets,threshold):
    left_y,left_x= [],[]
    right_y,right_x=[],[]
    for i in range (len(features)):
        if features[i]<=threshold:
            left_x.append(features[i])
            left_y.append(features[i])
        else:
            right_x.append(features[i])
            right_y.append(features[i])
    return left_x,right_x,left_y,right_x,right_y

def best_split(features,targets):
    best_thresh=None
    best_gain=-1
    parent_entropy=entropy(features)


    for thresh in features:
        left_x,right_x,left_y,right_y=split_data(features,targets,thresh)


        if len(left_y)== 0 or len(right_y) == 0:
            continue
        w_l= len(left_y)/len(targets)
        w_r=len(right_y)/len(targets)


        child_entropy=w_l*entropy(left_y)+ w_r*entropy(right_y)
        gain=parent_entropy-child_entropy
        if gain>best_gain:
            best_gain=gain
            best_thresh=thresh
        return best_thresh
def build_trees(features,targets):
    # same value stop
    if targets.count(targets[0]) == len(targets):
        return targets[0]

    #split
    thresh = best_split(features, targets)

    # If no split works
    if thresh is None:
        return max(set(targets), key=targets.count)

def predict_tree(trees,x):
    if





