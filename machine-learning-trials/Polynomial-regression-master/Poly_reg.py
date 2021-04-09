import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def tahmin_ayar(x):
    deneme = np.array([x])
    deneme.resize(1, 1)
    return deneme
    
veriler=pd.read_csv("maaslar.csv")

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

X=x.values
Y=y.values

poly_reg=PolynomialFeatures(degree=3)
x_poly=poly_reg.fit_transform(X)

lin_reg=LinearRegression()
lin_reg.fit(x_poly,y)

plt.scatter(X,Y,c='red')
plt.plot(X,lin_reg.predict(x_poly),c='blue')
plt.show()

print(lin_reg.predict(poly_reg.fit_transform(tahmin_ayar(11))))
