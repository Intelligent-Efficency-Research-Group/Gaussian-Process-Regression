import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math

merged = pd.read_csv(r'\\drumlin\vseu\ClancyD\Downloads\merged.csv')
merged = merged[merged['Energy (kWh)'] != 0]
merged = merged.reset_index()
merged = merged[merged['Total Units'] != 0]
merged = merged.reset_index()

kernelmatern = 3.0 * Matern(length_scale=3.0, length_scale_bounds=(1e-1, 50.0), nu=1.5)
gprmatern = GaussianProcessRegressor(kernel=kernelmatern, n_restarts_optimizer=1000, normalize_y=True, alpha=1e-5)

energy = merged.iloc[:]['Energy (kWh)']
energy = np.array(energy).reshape(-1, 1)
total = merged.iloc[:]['Total Units']
total = np.array(total).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(total, energy, random_state=42)
Xtrain = np.array(X_train).reshape(-1, 1)
Xtest = np.array(X_test).reshape(-1, 1)
ytrain = np.array(y_train).reshape(-1, 1)
ytest = np.array(y_test).reshape(-1, 1)

gprmatern.fit(Xtrain, ytrain)

predictions = np.array(gprmatern.predict(Xtest)).reshape(52, 1)
r2rbf = r2_score(ytest, predictions)
mse = mean_squared_error(ytest, predictions)
rmse = math.sqrt(mse)
mae = mean_absolute_error(ytest, predictions)
print(r2rbf)
print(mse)
print(rmse)
print(mae)

X = np.linspace(start=0, stop=150, num=1000).reshape(-1, 1)
predict = gprmatern.predict(X)

mean_prediction, std_prediction = gprmatern.predict(X, return_std=True)

plt.plot(X, mean_prediction, label="Mean prediction", color='blue')
plt.plot(X, predict)
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.scatter(total, energy, s=0.8, color='magenta', label='Observations')
plt.legend()
plt.show()

tester = gprmatern.predict(Xtest)
plt.scatter(Xtest, tester, color='red', label='predictions')
plt.scatter(Xtest, ytest, color='green', label='Actual Usage')
plt.legend()
plt.show()

plotpredictions = np.array(gprmatern.predict(Xtest)).reshape(52, 1)
wait = np.array([Xtest, plotpredictions])
df = pd.DataFrame({'x':Xtest.reshape(52), 'y':plotpredictions.reshape(52)})
df = df.sort_values('x')
df.plot(x='x', y='y', label='Predicted Usage')
plt.scatter(Xtest, ytest, color='green', label='Observed Usage', s=5)
plt.legend()
plt.show()