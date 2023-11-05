from utils import process_new
import joblib
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from datetime import datetime

#### Load Thae Model
Model01_Path=os.path.join(os.getcwd(),'models','XGB.pkl')
model_XGB= joblib.load(Model01_Path)
Model02_Path=os.path.join(os.getcwd(),'models','LightGBM.pkl')
model_GBM= joblib.load(Model02_Path)


### Load The data
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

#### Identify Function
def regression_price():

    
    st.set_page_config(
    layout='wide',
    page_title='Melbourn buiding price prediction' ,
    page_icon='💰'
    )
    st.write('<h1 style = "text-align: center;">Hello Streamlit Dash Board For Melbourn buiding price prediction!</h1>', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    page = st.sidebar.radio('Select Page', ['Home_Page','Dataset Overview', 'Describtive Statistics', 'Charts','Model_Prediction_For_Price'])
    
    if page == 'Home_Page' :
        st.image('mel.PNG',width=1500)
        text = '''
#                                       selling of units Pricing Dataset, melbourn

### `This is Supervised Task (Regression) `

###  About Dataset

Context

This data was scraped from publicly available results posted every week from Domain.com.au, I've cleaned it as best I can, now it's up to make data analysis magic. The dataset includes Address, Type of Real estate, Suburb, Method of Selling, Rooms, Price, Real Estate Agent, Date of Sale and distance from C.B.D.

### About the project
- This is a machine learning project to predict unit/property selling price in Melbourn. 

- This project aims to answers question about how much a unit price would be if given information such as location, number of bedrooms, etc? This would help potential tenant and also the owner to get the best price of their units, comparable to the market value. 


### Content

There are 20 features with one unique ids (ads_id) and one target feature (monthly_rent)

-Suburb: Suburb

Address: Address

Rooms: Number of rooms

Price: Price in Australian dollars

Method:

- S - property sold;
- SP - property sold prior;
- PI - property passed in;
- PN - sold prior not disclosed;
- SN - sold not disclosed;
- NB - no bid;
- VB - vendor bid;
- W - withdrawn prior to auction;
- SA - sold after auction;
- SS - sold after auction price not disclosed.
- N/A - price or highest bid not available.
Type:

- br - bedroom(s);
- h - house,cottage,villa, semi,terrace;
- u - unit, duplex;
- t - townhouse;
- dev site - development site;
- o res - other residential.
- SellerG: Real Estate Agent

Date: Date sold

Distance: Distance from CBD in Kilometres

Regionname: General Region (West, North West, North, North east …etc)

Propertycount: Number of properties that exist in the suburb.

Bedroom2 : Scraped # of Bedrooms (from different source)

Bathroom: Number of Bathrooms

Car: Number of carspots

Landsize: Land Size in Metres

BuildingArea: Building Size in Metres

YearBuilt: Year the house was built

CouncilArea: Governing council for the area

Lattitude: Self explanitory

Longtitude: Self explanitory

- Inspiration
 in the past there was no easy way to understand whether certain unit pricing is making sense or not. With this dataset, I wanted to be able to answer the following questions:

- What are the biggest factor affecting the unit/rent pricing?
- here we want to analyze Mellbourne data to cover points:
- Which suburbs are the best to buy in?
- Which ones are value for money?
- Where's the expensive side of town?
- where should I buy a 2 bedroom unit?
        '''
   
        
        st.markdown(text)

    elif page == 'Dataset Overview':
           
        space1, col, space2 = st.columns([2, 20, 2])
        col.dataframe(df.head(), width=8000, height=200)

    elif page== 'Describtive Statistics':
            col1, space, col4= st.columns([5,1,5])
            with col1: 
                    st.dataframe(df2.describe(include='number'), width=500, height=200)
                   

            with col4: 
                    st.dataframe(df2.describe().transpose().style.background_gradient(cmap='hot'))
           
                   

           

    elif page == 'Model_Prediction_For_Price':
        ## Choose Model
        model_type = st.selectbox('Choose the Model', options=['GBM', 'XGB'])

        ## Input fields
        Suburb = st.selectbox('Suburb', options=df2['Suburb'].unique())
        Rooms = st.slider('Rooms Numbers', min_value=1, max_value=11, step=1)
        Type = st.selectbox('Type', options=['h', 'u','t'])
        Method = st.selectbox('Method-Of-Selling',options= ['s', 'sp', 'pi', 'vb', 'sa'])
        Seller = st.selectbox('Which_Seller',options=df2['Seller'].unique())
        
        Distance = st.number_input('Distance_From_Center', value=20.0)
        Two_bedroom_unit = st.slider('Two_bedroom_unit Numbers', min_value=1, max_value=11, step=1)
        Bathroom = st.slider('Bathroom', min_value=0, max_value=11, step=1)
        Car = st.slider('Car_Parking', min_value=0, max_value=11, step=1)
        Landsize = st.number_input('Landsize', value=200.0)
        BuildingArea = st.number_input('BuildingArea', value=200.0)
        YearBuilt = st.number_input('YearBuilt', value=1970.0)
        CouncilArea= st.selectbox('Name of CouncilArea', options=['yarra', 'moonee', 'port', 'darebin', 'hobsons', 'stonnington',
       'boroondara', 'monash', 'glen', 'whitehorse', 'maribyrnong',
       'bayside', 'moreland', 'manningham', 'melbourne', 'banyule',
       'brimbank', 'kingston', 'hume', 'knox', 'maroondah', 'casey',
       'melton', 'greater', 'nillumbik', 'whittlesea', 'frankston',
       'macedon', 'wyndham', 'cardinia', 'moorabool', 'mitchell'])
        
        Lattitude=st.number_input('Lattitude', value=-37.7641)
        Longtitude=st.number_input('Lattitude', value=144.9112)
        Regionname= st.selectbox('Region_name', options=df2['Regionname'].unique())
        
        Propertycount= st.number_input('Property_count', value=4019.0)
        Region=st.selectbox('Main_Region',options= ['Metropolitan', 'Victoria'])
        District=st.selectbox('District_Direction',options= ['Southern', 'Northern', 'Western', 'Eastern', 'South-Eastern'])
        Month= st.slider('Month_of_the_year', min_value=1, max_value=12, step=1)


    
        st.markdown('<hr>', unsafe_allow_html=True)
        if st.button('Predict Unit Price ...'):

            ## Concatenate the users data
            new_data = np.array([Suburb, Rooms, Type, Method, Seller, Distance,
            Two_bedroom_unit, Bathroom, Car, Landsize, BuildingArea,
           YearBuilt, CouncilArea, Lattitude, Longtitude, Regionname,
            Propertycount, Region, District,Month])
            
            ## Call the function from utils.py to apply the pipeline
            x_proceed = process_new(x_new=new_data)

            ## Predict using Model
            if model_type == 'XGB':
                y_pred = model_XGB.predict(x_proceed)
            elif model_type == 'GBM':
                y_pred = model_GBM.predict(x_proceed)

            st.success(f'Your Buiding Price in uero is about ... {y_pred*1000}')
                  

               


#### Router
if  __name__== '__main__' :
   ### Call The Function
      regression_price()

