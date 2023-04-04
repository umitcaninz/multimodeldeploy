import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile


pd.set_option("display.max_columns",None)
warnings.simplefilter(action="ignore",category=Warning)

data = pd.read_csv("C:/Users/can/Desktop/diabetes.csv")
df = data.copy()


y = df["Outcome"]
X = df.drop(["Outcome"],axis=1)

cart_model = DecisionTreeClassifier(random_state = 1).fit(X,y)


y_pred = cart_model.predict(X)

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.30,random_state=45)
cart_model = DecisionTreeClassifier(random_state=17).fit(X_train,y_train)

cv_results = cross_validate(cart_model,
                           X,
                           y,
                           cv=10,
                            scoring = ["accuracy","f1","roc_auc"]
                           )


cart_params = {"max_depth":range(1,11),
              "min_samples_split":range(2,20)}

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,

                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)

cart_best_grid.best_params_

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_,random_state=17).fit(X,y)
cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X,y)
cv_results = cross_validate(cart_final,
                           X,y,
                            cv=5,
                           scoring=["accuracy","f1","roc_auc"])


import pickle
pickle.dump(cart_final,open("diabetes.pkl","wb"))
model = pickle.load(open("diabetes.pkl","rb"))

x = [6,148,78,35,0,30,0.62,50]
model.predict(pd.DataFrame(x).T)
cart_final.predict(pd.DataFrame(x).T)

inputt=[int(x) for x in "1 12 32 60 12 123 131 12".split(' ')]
final=[pd.DataFrame(inputt).T]

model.predict_proba(final)

model.predict(inputt)