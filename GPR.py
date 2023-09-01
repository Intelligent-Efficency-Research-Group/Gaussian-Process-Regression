import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

merged = pd.read_csv(r'\\drumlin\vseu\ClancyD\Downloads\merged.csv')
merged = merged[merged['Energy (kWh)'] !=0]
merged = merged.reset_index()
merged = merged[merged['Total Units'] !=0]
merged = merged.reset_index()

kernelrbf = 1 * RBF(length_scale=3.0, length_scale_bounds=(1e-2, 1e2))
gprrbf = GaussianProcessRegressor(kernel=kernelrbf, n_restarts_optimizer=100, normalize_y = True,alpha = 1e-5)
kernelmatern = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),nu=1.5)
gprmatern = GaussianProcessRegressor(kernel=kernelmatern, n_restarts_optimizer=100, normalize_y = True,alpha = 1e-5)

energy = merged.iloc[:]['Energy (kWh)']
energy = np.array(energy).reshape(-1,1)
total = merged.iloc[:]['Total Units']
total = np.array(total).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(total, energy)
Xtrain = np.array(X_train).reshape(-1,1)
Xtest = np.array(X_test).reshape(-1,1)
ytrain = np.array(y_train).reshape(-1,1)
ytest = np.array(y_test).reshape(-1,1)

gprrbf.fit(Xtrain,ytrain)
gprmatern.fit(Xtrain,ytrain)

predictions = np.array(gprrbf.predict(Xtest)).reshape(52,1)
r2rbf = r2_score(ytest,predictions)
print(r2rbf) #Not so good

X = np.linspace(start=0, stop=150, num=1000).reshape(-1,1)
predict = gprrbf.predict(X)

mean_prediction, std_prediction = gprrbf.predict(X, return_std=True)

plt.plot(X, mean_prediction, label="Mean prediction",color='blue')
plt.plot(X,predict)
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.scatter(total,energy,s= 0.8,color='magenta',label='Observations')
plt.legend()
plt.show()

tester = gprrbf.predict(Xtest)
plt.scatter(Xtest,tester,color='red',label = 'predictions')
plt.scatter(Xtest,ytest,color='green',label = 'Actual Usage')
plt.legend()
plt.show()
