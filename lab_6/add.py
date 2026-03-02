def normalization(lst):
    x_min=lst[0]
    for i in lst:
        if i<x_min:
            x_min=i
    x_max=lst[0]
    for i in lst:
        if i>x_max:
            x_max=i
    norm_vals=[]
    for i in lst:
        x_norm=(i-x_min)/(x_max-x_min)
        norm_vals.append(x_norm)
    return norm_vals

lst=list(map(int,input("enter : ").split()))
result=normalization(lst)
print(result)