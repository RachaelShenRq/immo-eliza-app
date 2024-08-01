from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score

def print_score(y_test, y_pred, model_name):
    mae=mean_absolute_error(y_true=y_test,y_pred=y_pred)
    r_2=r2_score(y_true=y_test,y_pred=y_pred)
    print(f"""
    ################# Score for {model_name} ###############
        R²  =   {r_2}
        MAE =   {mae}
        
    #######################################################    
        """)

preprocessor = pickle.load(open("data/preprocessor.pkl","rb"))


df=pd.read_json("data/final_dataset.json")
df["TypeOfSale"]=df.TypeOfSale.apply(lambda x : 1 if "residential_sale" in x else (2 if "monthly_rent" in x else None))
df.dropna(subset=["TypeOfSale"],inplace=True)

X = df[preprocessor.feature_names_in_]
y = df["Price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)
print(X_train.shape,X_test.shape)

X_train = preprocessor.transform(X_train)
X_test=preprocessor.transform(X_test)

print("Test random forest")

model=RandomForestRegressor()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print_score(y_test,y_pred,"random forest")
pickle.dump(model,open("data/random_forest.pkl","wb"))

print("Test Gradient Boosting")
model=GradientBoostingRegressor()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print_score(y_test,y_pred,"Gradien Boosting")
pickle.dump(model,open("data/gb.pkl","wb"))