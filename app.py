#from ctypes import WinDLL, c_char, c_long, c_double, POINTER, \
     #WINFUNCTYPE, byref, create_string_buffer, pointer, string_at
#from ctypes.wintypes import LPCSTR, LPSTR, DWORD, CHAR, PDWORD
#from ctypes.util import find_library
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
#import warnings
#warnings.filterwarnings('ignore')
import plotly
#from plotly.offline import download_plotlyjs, init_notebook_mode , plot,iplot
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.colors import n_colors
from plotly.subplots import make_subplots
import cufflinks as cf
cf.go_offline()
#import itertools
#------------------------------------------------------------------------------------------------------------------
#streamlit run "c:\Users\dhakksinesh\Desktop\PRO 1\cre.py"
st.header('CRIME ANALYSIS AND PREDICTION USING MACHINE LEARNING ALGORITHM')
st.caption("By TEAM E-14")
#----------------------------------------------------------dataset---------------------------------------------------
st.subheader('DATA ANALYSIS')
dataset = pd.read_csv("20_Victims_of_rape - Copy.csv")
dataset.index = np.arange(1, len(dataset)+1)
st.subheader("\n")
st.caption("Rape Cases Reported - Dataset")
st.dataframe(dataset)
#-----------------------------------------------------------year-sum-------------------------------------------------
df1 = pd.read_csv('20_Victims_of_rape.csv')
df1= df1[df1['Subgroup']=='Total Rape Victims']
g= pd.DataFrame(df1.groupby(['Year'])['Rape_Cases_Reported'].sum().reset_index())
st.subheader("\n")
st.subheader("\n")
st.caption("Total Rape Cases Reported by Year")
fig=px.bar(g,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['blue'])
st.plotly_chart(fig, use_container_width=True)
#-------------------------------------------------------state-sum-------------------------------------------------------
st.caption("Total Rape Cases Reported by State")
df = pd.read_csv('20_Victims_of_rape - Copy.csv')
year = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
choice_year = st.selectbox(' ',year)

if choice_year=='2010':
    df = pd.read_csv('20_Victims_of_rape - Copy.csv')
    fig= px.bar(df,x='Area_Name',y='Total_10',color_discrete_sequence=['yellow'])
    st.plotly_chart(fig, use_container_width=True)

if choice_year=='2011':
    df = pd.read_csv('20_Victims_of_rape - Copy.csv')
    fig= px.bar(df,x='Area_Name',y='Total_11',color_discrete_sequence=['yellow'])
    st.plotly_chart(fig, use_container_width=True)

if choice_year=='2012':
    df = pd.read_csv('20_Victims_of_rape - Copy.csv')
    fig= px.bar(df,x='Area_Name',y='Total_12',color_discrete_sequence=['yellow'])
    st.plotly_chart(fig, use_container_width=True)

if choice_year=='2013':
    df = pd.read_csv('20_Victims_of_rape - Copy.csv')
    fig= px.bar(df,x='Area_Name',y='Total_13',color_discrete_sequence=['yellow'])
    st.plotly_chart(fig, use_container_width=True)

if choice_year=='2014':
    df = pd.read_csv('20_Victims_of_rape - Copy.csv')
    fig= px.bar(df,x='Area_Name',y='Total_14',color_discrete_sequence=['yellow'])
    st.plotly_chart(fig, use_container_width=True)

if choice_year=='2015':
    df = pd.read_csv('20_Victims_of_rape - Copy.csv')
    fig= px.bar(df,x='Area_Name',y='Total_15',color_discrete_sequence=['yellow'])
    st.plotly_chart(fig, use_container_width=True)

if choice_year=='2016':
    df = pd.read_csv('20_Victims_of_rape - Copy.csv')
    fig= px.bar(df,x='Area_Name',y='Total_16',color_discrete_sequence=['yellow'])
    st.plotly_chart(fig, use_container_width=True)

if choice_year=='2017':
    df = pd.read_csv('20_Victims_of_rape - Copy.csv')
    fig= px.bar(df,x='Area_Name',y='Total_17',color_discrete_sequence=['yellow'])
    st.plotly_chart(fig, use_container_width=True)

if choice_year=='2018':
    fig= px.bar(df,x='Area_Name',y='Total_18',color_discrete_sequence=['yellow'])
    st.plotly_chart(fig, use_container_width=True)

if choice_year=='2019':
    fig= px.bar(df,x='Area_Name',y='Total_19',color_discrete_sequence=['yellow'])
    st.plotly_chart(fig, use_container_width=True)
#-----------------------------------------------------sample-pred-------------------------------------------------------
dataset = pd.read_csv('20_Victims_of_rape - Copy.csv')
dataset.index = np.arange(1, len(dataset)+1)
dataset.describe()
#dataset.dtypes
#for x in dataset:
    #if dataset[x].dtypes == "int64":
        #dataset[x] = dataset[x].astype(float)

dataset = dataset.select_dtypes(exclude=['object'])
dataset = dataset.fillna(dataset.mean())
X = dataset.drop('Total_19', axis=1)
y = dataset['Total_19']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
st.subheader("\n")
st.subheader("\n")
st.caption("Sample Prediction Table")
dataset=pd.DataFrame({'Current':y_test, 'Predicted':y_pred.round()})
st.write(dataset.sort_index())
#-----------------------------------------------------------rf----------------------------------------------------------
from sklearn import metrics
metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#Calculate absolute errors
errors = abs(y_pred - y_test)
#Mean absolute error (mae)
round(np.mean(errors), 2)
#Mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
#Accuracy
accuracy = 100 - np.mean(mape)
acc=(round(accuracy, 2))
st.subheader("\n")
st.subheader("\n")
st.subheader('RANDOM FOREST')
st.subheader("\n")
st.caption("Accuracy %")
st.success(acc)

from sklearn.metrics import r2_score
st.subheader("\n")
st.subheader("\n")
st.caption("Regression Score")
r=r2_score(y_test, y_pred)
st.info(r)
#-------------------------------------------------------state-pred---------------------------------------------------
df = pd.read_csv('20_Victims_of_rape - Copy.csv')
df.index = np.arange(1, len(df)+1)
state = df['Area_Name']
st.subheader("\n")
st.subheader("\n")
st.subheader("STATE PREDICTION")
choice_state = st.selectbox('',state)
#--------------------------------------------------------1----------------------------------------------------------
if choice_state=='Andaman & Nicobar Islands':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Andaman & Nicobar Islands']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Andaman & Nicobar Islands']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------2--------------------------------------------------------
if choice_state=='Andhra Pradesh':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Andhra Pradesh']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Andhra Pradesh']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------3--------------------------------------------------------
if choice_state=='Arunachal Pradesh':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Arunachal Pradesh']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Arunachal Pradesh']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)    
#----------------------------------------------------------4--------------------------------------------------------
if choice_state=='Assam':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Assam']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Assam']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df) 
#----------------------------------------------------------5--------------------------------------------------------
if choice_state=='Bihar':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Bihar']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Bihar']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df) 
#----------------------------------------------------------6--------------------------------------------------------
if choice_state=='Chandigarh':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Chandigarh']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Chandigarh']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df) 
#----------------------------------------------------------7--------------------------------------------------------
if choice_state=='Chhattisgarh':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Chhattisgarh']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Chhattisgarh']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df) 
#----------------------------------------------------------8--------------------------------------------------------
if choice_state=='Dadra & Nagar Haveli':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Dadra & Nagar Haveli']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Dadra & Nagar Haveli']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df) 
#----------------------------------------------------------9--------------------------------------------------------
if choice_state=='Daman & Diu':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Daman & Diu']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Daman & Diu']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df) 
#----------------------------------------------------------10--------------------------------------------------------
if choice_state=='Delhi':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Delhi']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Delhi']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df) 
#----------------------------------------------------------11--------------------------------------------------------
if choice_state=='Goa':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Goa']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Goa']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------12--------------------------------------------------------
if choice_state=='Gujarat':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Gujarat']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Gujarat']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------13--------------------------------------------------------
if choice_state=='Haryana':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Haryana']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Haryana']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------14--------------------------------------------------------
if choice_state=='Himachal Pradesh':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Himachal Pradesh']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Himachal Pradesh']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------15--------------------------------------------------------
if choice_state=='Jammu & Kashmir':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Jammu & Kashmir']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Jammu & Kashmir']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------16--------------------------------------------------------
if choice_state=='Jharkhand':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Jharkhand']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Jharkhand']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------17--------------------------------------------------------
if choice_state=='Karnataka':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Karnataka']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Karnataka']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------18--------------------------------------------------------
if choice_state=='Kerala':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Kerala']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Kerala']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------19--------------------------------------------------------
if choice_state=='Lakshadweep':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Lakshadweep']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Lakshadweep']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------20--------------------------------------------------------
if choice_state=='Madhya Pradesh':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Madhya Pradesh']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Madhya Pradesh']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------21--------------------------------------------------------
if choice_state=='Maharashtra':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Maharashtra']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Maharashtra']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------22--------------------------------------------------------
if choice_state=='Manipur':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Manipur']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Manipur']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------23--------------------------------------------------------
if choice_state=='Meghalaya':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Meghalaya']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Meghalaya']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------24--------------------------------------------------------
if choice_state=='Mizoram':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Mizoram']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Mizoram']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------25--------------------------------------------------------
if choice_state=='Nagaland':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Nagaland']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Nagaland']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------26--------------------------------------------------------
if choice_state=='Odisha':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Odisha']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Odisha']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------27--------------------------------------------------------
if choice_state=='Puducherry':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Puducherry']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Puducherry']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------28--------------------------------------------------------
if choice_state=='Punjab':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Punjab']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Punjab']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------29--------------------------------------------------------
if choice_state=='Rajasthan':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Rajasthan']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Rajasthan']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#----------------------------------------------------------30--------------------------------------------------------
if choice_state=='Sikkim':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Sikkim']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Sikkim']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#-----------------------------------------------------------31-------------------------------------------------------
if choice_state=='Tamil Nadu':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Tamil Nadu']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Tamil Nadu']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#-----------------------------------------------------------32-------------------------------------------------------
if choice_state=='Tripura':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Tripura']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Tripura']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#-----------------------------------------------------------33-------------------------------------------------------
if choice_state=='Uttar Pradesh':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Uttar Pradesh']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Uttar Pradesh']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#-----------------------------------------------------------34-------------------------------------------------------
if choice_state=='Uttarakhand':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='Uttarakhand']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='Uttarakhand']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#-----------------------------------------------------------35-------------------------------------------------------
if choice_state=='West Bengal':
    df = pd.read_csv('20_Victims_of_rape.csv')
    df = df[df['Area_Name']=='West Bengal']
    df = df[df['Subgroup']=='Total Rape Victims']
    st.subheader("\n")
    st.caption("Total Cases Reported")
    fig= px.bar(df,x='Year',y='Rape_Cases_Reported',color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
    df = pd.read_csv("20_Victims_of_rape - Copy.csv")
    df.index = np.arange(1, len(df)+1)
    df = df[df['Area_Name']=='West Bengal']
    st.caption("State Data")
    st.write(df)
    df = df.select_dtypes(exclude=['object'])
    df = df.fillna(dataset.mean())
    X = df.drop('Total_19', axis=1)
    y = df['Total_19']
    i=regressor.predict(X).round()
    st.subheader("\n")
    st.caption("State Prediction Table")
    df=pd.DataFrame({'Current':y, 'Predicted':i})
    st.write(df)
#-----------------------------------------------------------------------------------------------------------------
st.subheader("\n")
st.subheader("\n")
st.subheader("ACCURACY COMPARISON")
#--------------------------------------------------------svm-------------------------------------------------------
dataset = pd.read_csv('20_Victims_of_rape - Copy.csv')
dataset.describe()
dataset = dataset.select_dtypes(exclude=['object'])
dataset = dataset.fillna(dataset.mean())
X = dataset.drop('Total_19', axis=1)
y = dataset['Total_19']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
dataset=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
from sklearn import metrics
metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
np.sqrt(metrics.mean_squared_error(y_test, y_pred))
errors = abs(y_pred - y_test)
round(np.mean(errors), 2)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
acc=(round(accuracy, 2))
st.subheader("\n")
st.caption("SVM Accuracy %")
st.warning(acc)
#------------------------------------------------LinearRegression-----------------------------------------------------
dataset = pd.read_csv('20_Victims_of_rape - Copy.csv')
dataset.describe()
dataset = dataset.select_dtypes(exclude=['object'])
dataset = dataset.fillna(dataset.mean())
X = dataset.drop('Total_19', axis=1)
y = dataset['Total_19']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
dataset=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
from sklearn import metrics
metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
np.sqrt(metrics.mean_squared_error(y_test, y_pred))
errors = abs(y_pred - y_test)
round(np.mean(errors), 2)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
acc=(round(accuracy, 2))
st.subheader("\n")
st.caption("Linear Regression Accuracy %")
st.warning(acc)
#---------------------------------------------K-Nearest Neighbors---------------------------------------------------
dataset = pd.read_csv('20_Victims_of_rape - Copy.csv')
dataset.describe()
dataset = dataset.select_dtypes(exclude=['object'])
dataset = dataset.fillna(dataset.mean())
X = dataset.drop('Total_19', axis=1)
y = dataset['Total_19']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
dataset=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
from sklearn import metrics
metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)
np.sqrt(metrics.mean_squared_error(y_test, y_pred))
errors = abs(y_pred - y_test)
round(np.mean(errors), 2)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
acc=(round(accuracy, 2))
st.subheader("\n")
st.caption("KNN Accuracy %")
st.warning(acc)
#-----------------------------------------------------------------------------------------------------------------
