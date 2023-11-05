

import pandas as pd
import numpy as np
from datetime import datetime
## scopndary

import os

## sklearn -- preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import  SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

TRAIN_PATH = os.path.join(os.getcwd(),'Melbourne_housing_FULL.csv')
df = pd.read_csv(TRAIN_PATH)
df2= df.copy(deep = True)


def fur(x):
    try:
       return x.replace(' ','_')
    except:
        return np.nan
    
df2['Suburb']= df2['Suburb'].apply(fur)


def fur2(x):
    try:
       return '_'.join(x.lower().split(' ')[1:] )
    except:
        return np.nan
    
df2['Address']= df2['Address'].apply(fur2)   

df2.rename(columns = {'SellerG':'Seller'}, inplace = True)

df2['Date']=pd.to_datetime(df2["Date"])

## Check bedrooms
df2['Bedroom2']= df2.rename(columns={'Bedroom2':'Two_bedroom_unit'}, inplace=True)

ids_wrong = df2[(df2['Landsize']==0.0)|(df2['Landsize']==df2['Landsize'].max())].index.tolist()
df2.drop(index=ids_wrong, axis=0, inplace=True)


ids_wrong = df2[(df2['Landsize']<=0.0)].index.tolist()
df2.drop(index=ids_wrong, axis=0, inplace=True)

def fur3(x):
    try:
       return x.split(' ')[0]
    except:
        return np.nan
    
df2['CouncilArea']= df2['CouncilArea'].apply(fur3)

def fur4(x):
    try:
       return x.split(' ')[1]
    except:
        return np.nan
    
df2['Region']= df2['Regionname'].apply(fur4) 

def fur5(x):
    try:
       return x.split(' ')[0]
    except:
        return np.nan
df2['District']= df2['Regionname'].apply(fur5)

df2['Price']=df2['Price']/1000

df2['price_per_m2']= (df2['Price']*1000)/df2['Landsize']

df2['Date']=pd.to_datetime(df2["Date"])
df2['Month']=df2['Date'].dt.month
df2['Year']=df2['Date'].dt.year

df2= df2.applymap(lambda s:s.lower() if type(s) == str else s)


current_year = float(datetime.now().year)
df2['Bulding_Age']= current_year-df2['YearBuilt']

df2.drop(df2[['Address','Date','Bedroom2','price_per_m2','Bulding_Age','Postcode','Year']], axis=1, inplace=True)


## split to faetures and target
X = df2.drop(columns=['Price'], axis=1)   ## faetures
y = df2['Price']   ## traget

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=110)


num_cols = X_train.select_dtypes(include='number').columns.tolist()
categ_cols = list(set(X_train.select_dtypes(exclude='number').columns.tolist())-set(['Type']))
other=['Type']


num_pipeline = Pipeline(steps=[
                    
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('transform', PowerTransformer(method='yeo-johnson', standardize=True))
                ])


## Catgeorical pipeline
categ_pipeline = Pipeline(steps=[
                
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
               ])


other_pipeline = Pipeline(steps=[
                
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder())
               ])


## Combine all

preprocessing = ColumnTransformer(transformers=[('Numerical', num_pipeline, num_cols),
                                 ('Categorical', categ_pipeline, categ_cols),('other', other_pipeline, other)], remainder= 'passthrough')




## fit and transform
ML=preprocessing.fit(X_train)


def process_new(x_new):
    ''' This function is to process the new data using pipeline 
    Args:
    *****
        (X_new: 2d arary) --> The required instance to be processed
    Returns:
    *******
        (X_processed: 2d array) --> The procesed instance, ready for inference

    '''

    ## Call the pipeline
    df_new= pd.DataFrame([x_new], columns=X_train.columns)

    # adjust the data type
    df_new['Suburb']= df_new['Suburb'].astype('object')
    df_new['Rooms']= df_new['Rooms'].astype('int64')
    df_new['Type']= df_new['Type'].astype('object')
    df_new['Method']= df_new['Method'].astype('object')
    df_new['Seller']= df_new['Seller'].astype('object')
    df_new['Distance']= df_new['Distance'].astype('float')
    df_new['Two_bedroom_unit']= df_new['Two_bedroom_unit'].astype('float64')
    df_new['Bathroom']= df_new['Bathroom'].astype('float64')
    df_new['Car']= df_new['Car'].astype('float64')
    df_new['Landsize']= df_new['Landsize'].astype('float64')
    df_new['BuildingArea']= df_new['BuildingArea'].astype('float64')
    df_new['YearBuilt']= df_new['YearBuilt'].astype('float64')
    df_new['CouncilArea']= df_new['CouncilArea'].astype('object')
    df_new['Lattitude']= df_new['Lattitude'].astype('float64')
    df_new['Longtitude']= df_new['Longtitude'].astype('float64')
    df_new['Regionname']= df_new['Regionname'].astype('object')
    df_new['Propertycount']= df_new['Propertycount'].astype('float64')
    df_new['Region']= df_new['Region'].astype('object')
    df_new['District']= df_new['District'].astype('object')
    df_new['Month']= df_new['Month'].astype('int64')
   


    x_proceed= ML.transform(df_new)
    return    x_proceed



