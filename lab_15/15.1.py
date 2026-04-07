from ISLP import load_data

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.tree import plot_tree
from sklearn.metrics import r2_score,root_mean_squared_error


def load_dta():

    boston=load_data("Boston")
    x = boston.drop("medv", axis=1)
    y = boston["medv"]

    return x,y

def split_data(x,y):

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    scaled=StandardScaler()

    x_train_scaled=scaled.fit_transform(x_train)

    x_test_scaled=scaled.transform(x_test)

    return x_train_scaled,x_test_scaled,y_train,y_test

def train_gradientboost(x_train_scaled,x_test_scaled,y_train,y_test):

    #baseestimator=DecisionTreeClassifier(max_depth=3)

    model=GradientBoostingRegressor(n_estimators=50,learning_rate=1,max_depth=4,random_state=42)
    model.fit(x_train_scaled,y_train)
    ypred=model.predict(x_test_scaled)
    r2score=r2_score(y_test,ypred)
    rmsesquare=root_mean_squared_error(y_test,ypred)
    print("score of the model is:",r2score,rmsesquare)






def main():

    x,y=load_dta()

    x_train_scaled,x_test_scaled,y_train,y_test=split_data(x,y)

    train_gradientboost(x_train_scaled,x_test_scaled,y_train,y_test)
main()

