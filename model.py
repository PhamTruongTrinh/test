import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("1.csv")

# take a look at the dataset
#df.head()

#use required features
cdf = df[['SoTcD_1','SoTcR_1','DTB_1','SoTcD_2','SoTcR_2','DTB_2','SoTcD_3','SoTcR_3','DTB_3','SoTcD_4','SoTcR_4','DTB_4','SoTcD_5','SoTcR_5','DTB_5','KetQua']]

#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
x = cdf.iloc[:, :15]
y = cdf.iloc[:, -1]

regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to disk
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model.pkl','wb'))

'''
#test
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6, 8, 10.1]]))
'''
