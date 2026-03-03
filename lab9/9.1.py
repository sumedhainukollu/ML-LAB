import pandas as pd
df=pd.read_csv("simulated_dataset.csv")
left_t_80=df[df["BP"]>=80]
right_t_80=df[df["BP"]<80]
print(left_t_80,right_t_80)
print(left_t_80.shape,right_t_80.shape)

left_t_78=df[df["BP"]>=78]
right_t_78=df[df["BP"]<78]
print(left_t_78,right_t_78)
print(left_t_78.shape,right_t_78.shape)

left_t_82=df[df["BP"]>=82]
right_t_82=df[df["BP"]<82]
print(left_t_82,right_t_82)
print(left_t_82.shape,right_t_82.shape)

