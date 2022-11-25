import pandas as pd #visualization
from sklearn.model_selection import train_test_split #Train/Test/Split
from sklearn.linear_model import LinearRegression #Actual Algorithm
import joblib

df = pd.read_csv('Test.csv')
df.head()

x = df[["TotalCases", "NewCases", "LocalTransmission", "ReturningResidents", "ROFW", "Deaths", "APOR", "LDOrder", "Increaseincases", "IncreaseinPositivityRate"]]
x = x.replace(["Travel Protocol", "General", "Modified", "Enhanced"], [0, 1, 2, 3])
y = df["Positivity"]

lr = LinearRegression()
LinearRegression()
x_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0) #80% to be trained
lr.fit(x_train, y_train)

y_pred_train = lr.predict(X_test)
print(y_pred_train)





joblib.dump(lr, "clf.pkl")