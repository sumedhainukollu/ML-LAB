import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


df = pd.read_csv("/home/sumedha/Downloads/twitter_training.csv")

df = df[['text', 'sentiment']]
df = df.dropna()

X = df['text']
y = df['sentiment']


vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)


models = {
    "Linear": SVC(kernel="linear"),
    "RBF": SVC(kernel="rbf"),
    "Polynomial": SVC(kernel="poly", degree=2)
}


for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(name, "Accuracy:", acc)