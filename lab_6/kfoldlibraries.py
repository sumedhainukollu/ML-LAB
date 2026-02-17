from sklearn.datasets import  load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

#load data
iris=load_iris()
x=iris.data
print(x.shape)
y=iris.target
print(y.shape)
#model training
model=LogisticRegression()
#kfold validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

cross_val_results = cross_val_score(model, x, y, cv=kf)

print("Accuaracy:")
i=0
while i<len(cross_val_results):
    accuracy = cross_val_results[i] * 100
    print("Fold", i+1, "Accuracy =", accuracy)

    i = i + 1

tot = 0

i = 0

while i < len(cross_val_results):
    tot = tot + cross_val_results[i]
    i = i + 1
avg = tot/ len(cross_val_results)
print("averageaccuracy =", avg * 100)

