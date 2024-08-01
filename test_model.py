import pandas as pd
import pickle

df=pd.read_json("data/final_dataset.json")
df["TypeOfSale"]=df.TypeOfSale.apply(lambda x : 1 if "residential_sale" in x else (2 if "monthly_rent" in x else None))
df.dropna(subset=["TypeOfSale"],inplace=True)

preprocessor = pickle.load(open("data/preprocessor.pkl","rb"))
model = pickle.load(open("data/random_forest.pkl","rb"))
X=df[preprocessor.feature_names_in_]
Y=df["Price"]

test = X.sample(1)
print(f"""
      House details : 
      f{test.to_dict()}
      
      Real price : {Y[test.index[0]]} €
      
      Estimated price : {model.predict(preprocessor.transform(test))[0]} €
            """)
