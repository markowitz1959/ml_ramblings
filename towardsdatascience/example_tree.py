# taken from
# https://towardsdatascience.com/train-a-regression-model-using-a-decision-tree-70012c22bcc1


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

from sklearn.tree import export_graphviz
import graphviz


df = pd.read_csv('~/tmp/cali_housing.csv')

X = df[['Longitude']]
y = df['MedHouseVal']

dtr1 = DecisionTreeRegressor(max_depth=4, random_state=1)
dtr1.fit(X, y)

# -- visualize model

df.plot.scatter(x='Longitude', y='MedHouseVal', label='data')
plt.plot(df['Longitude'].sort_values(),
         dtr1.predict(df['Longitude'].sort_values().to_frame()),
         color='red', label='model',
         linewidth=2)

dot_data = export_graphviz(dtr1, feature_names=['Longitude'],
                           filled=True, rounded=True)

graph = graphviz.Source(dot_data)
graph.render("tree")


# -- with training and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10,
                                                    random_state=0,
                                                    shuffle=True)

max_depths = range(1, 20)
training_error = []
for max_depth in max_depths:
    model_1 = DecisionTreeRegressor(max_depth=max_depth)
    model_1.fit(X, y)
    training_error.append(mse(y, model_1.predict(X)))

testing_error = []
for max_depth in max_depths:
    model_2 = DecisionTreeRegressor(max_depth=max_depth)
    model_2.fit(X_train, y_train)
    testing_error.append(mse(y_test, model_2.predict(X_test)))


plt.plot(max_depths, training_error, color='blue', label='Training error')
plt.plot(max_depths, testing_error, color='green', label='Testing error')
plt.xlabel('Tree depth')
plt.axvline(x=7, color='orange', linestyle='--')
plt.annotate('optimum = 7', xy=(7.5, 1.17), color='red')
plt.ylabel('Mean squared error')
plt.title('Hyperparameter Tuning', pad=15, size=15)
plt.legend()


# -- k fold validation

from sklearn.model_selection import GridSearchCV

model = DecisionTreeRegressor()

gs = GridSearchCV(model,
                  param_grid = {'max_depth': range(1, 11),
                                'min_samples_split': range(10, 60, 10)},
                  cv=5,
                  n_jobs=1,
                  scoring='neg_mean_squared_error')

gs.fit(X_train, y_train)

print(gs.best_params_)
print(-gs.best_score_)


new_model = DecisionTreeRegressor(max_depth=9,
                                  min_samples_split=50)
#or new_model = gs.best_estimator_
new_model.fit(X_train, y_train)

plt.plot(df['Longitude'].sort_values(),
         new_model.predict(df['Longitude'].sort_values().to_frame()),
         color='red', label='model',
         linewidth=2)

plt.legend()
plt.title('Best Fitting', pad=15, size=15)
plt.savefig('new_model.png')



