import pandas as pd 

mydata = pd.read_csv("experience_salary.csv")
x = mydata[["X"]]
y = mydata[["Y"]]
print(mydata)