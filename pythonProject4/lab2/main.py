import pandas as pd

data = pd.read_csv('titanic.csv')
data1 = data.fillna(0)
data1 = data1[data1['Age'] > 30]
a = data1.head(10)
print(a)
data1 = data1.sort_values(by = 'Fare', ascending= True)
print(data1)
a = data1.groupby('Pclass')
x = a.agg({'Age' : 'mean'})
print(x)


