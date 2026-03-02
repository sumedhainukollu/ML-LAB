def data_norm(val_list):
    #find xmax
    val_min = val_list[0]
    for val in val_list:
        if val < val_min:
            val_min = val

    # find xmax
    val_max = val_list[0]
    for val in val_list:
        if val > val_max:
            val_max = val

    # apply normalization
    normalized_list = []
    for val in val_list:
        xnorm = (val - val_min) / (val_max - val_min)
        normalized_list.append(xnorm)

    return normalized_list



val_list = list(map(int, input("Enter values separated by space: ").split()))

result = data_norm(val_list)
print("normalized val:", result)