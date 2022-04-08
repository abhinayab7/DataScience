import pandas as pd, time, sys

# =============================================================================
# dataset = "http://garuda.pythonanywhere.com/static/house-prices.csv"
# target = "price"
# housePickleFile = "AutoHouseprices.pkl"
# =============================================================================

dataset = sys.argv[1]
housePickleFile = sys.argv[2]
target = sys.argv[3]


df=pd.read_csv(dataset)

df.shape

df.head()

y=df[target]

x=df.drop(columns=[target])

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)

model.score(x,y)

model.predict([[3000,3]])

metadata={"description":input("Enter the description : "),
           "version":input("Enter the version : "),
           "accuracy":model.score(x,y),
           "dataset":dataset,
           "algorithm":str(type(model)).replace('<','').replace('>',''),
           "timestamp":time.ctime(time.time())}

import pickle

f=open(housePickleFile,"wb")

pickle.dump(model,f)
pickle.dump(metadata,f)

f.close()
