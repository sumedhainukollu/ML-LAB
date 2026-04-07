from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree


def load_data():

    iris=load_iris()

    x=iris.data

    y=iris.target

    return x,y

def split_data(x,y):

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    scaled=StandardScaler()

    x_train_scaled=scaled.fit_transform(x_train)

    x_test_scaled=scaled.transform(x_test)

    return x_train_scaled,x_test_scaled,y_train,y_test

def train_adaboost(x_train_scaled,x_test_scaled,y_train,y_test):

    baseestimator=DecisionTreeClassifier(max_depth=3)

    model=AdaBoostClassifier(estimator=baseestimator,n_estimators=50,learning_rate=0.9)
    model.fit(x_train_scaled,y_train)
    ypred=model.predict(x_test_scaled)
    score=accuracy_score(y_test,ypred)
    print("score of the model is:",score)




def main():

    x,y=load_data()

    x_train_scaled,x_test_scaled,y_train,y_test=split_data(x,y)

    train_adaboost(x_train_scaled,x_test_scaled,y_train,y_test)
main()
