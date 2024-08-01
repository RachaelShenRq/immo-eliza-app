import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pickle

df=pd.read_json("data/final_dataset.json")
df["TypeOfSale"]=df.TypeOfSale.apply(lambda x : 1 if "residential_sale" in x else (2 if "monthly_rent" in x else None))
df.dropna(subset=["TypeOfSale"],inplace=True)

target=["Price"]
int_cols=["TypeOfProperty","TypeOfSale","BedroomCount"]
float_cols=["LivingArea"]

features_to_keep=int_cols+float_cols
print(f"Number of columns to keep : {len(features_to_keep)}\n")


### Create PreProcessor 

ct_int_cols=ColumnTransformer(("int_imput",SimpleImputer(strategy="most_frequent",copy=False)),remainder="passthrough")
ct_float_cols=ColumnTransformer(("imput_float",SimpleImputer(strategy="mean",copy=False)))

preprocessor=ColumnTransformer( transformers=
    [
        ("int_col",SimpleImputer(strategy="most_frequent"),int_cols),
        ("float_cols",SimpleImputer(strategy="mean"),float_cols),
    ],
    remainder="passthrough"
)
X=preprocessor.fit_transform(df[features_to_keep])
df_preprocessed=pd.DataFrame(X,columns=preprocessor.get_feature_names_out())
print(df_preprocessed.head())

#SAVE PREPROCESSOR
pickle.dump(preprocessor,open("data/preprocessor.pkl","wb"))
print("Preprocessor Saved!")