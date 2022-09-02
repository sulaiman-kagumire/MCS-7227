import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, cross_val_predict, ShuffleSplit

from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor


from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, mean_squared_log_error
import pickle as pkl
import streamlit as st
import numpy as np

pickle_in = open('model.pkl', 'rb') 
classifier = pkl.load(pickle_in)

# @st.cache(allow_output_mutation=True)

# '2001_IV', '2001_NRMVotes', '2001_OV', '2006_IV', '2006_NRMVotes', '2011_IV', '2016_NRMVOTES', '2016_NRM_Score', '2016_Opp_Votes'
def prediction(IV_2001, NRMVotes_2001, OV_2001, IV_2006, NRMVotes_2006 ,IV_2011, NRMVotes_2016, NRM_Score_2016, Opp_Votes_2016):
    
    
    predictions = classifier.predict(
        [[IV_2001, NRMVotes_2001, OV_2001, IV_2006, NRMVotes_2006 ,IV_2011, NRMVotes_2016, NRM_Score_2016, Opp_Votes_2016]])
    

    print(predictions)
         
    return predictions
# main function defines our webpage
def main():
    html_temp ="""
    <div style ="background-color:red;background-image: linear-gradient(45deg, #f3ec78, #af4261);background-size: 100%;  background-repeat: repeat;;
    -webkit-background-clip: text;-webkit-text-fill-color: transparent; -moz-background-clip: text;
    -moz-text-fill-color: transparent;"> 
    <h1 style ="text-align: center;font-family: "Archivo Black", sans-serif;font-weight: normal;font-size: 6em; ">NRM Presidential Score per District</h1> 
    </div> """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    
    # allow user input 
#         2001_IV = st.selectbox('Gender',('Male','Female'))
    IV_2001  = st.slider('2001 Invalid Votes' , min_value=1, max_value=10000000, value=10, step=1)
    NRMVotes_2001  = st.slider('2001 NRM Votes', min_value=1, max_value=1000000, value=10, step=1)
    OV_2001  = st.slider('2001 Opposition Votes' ,min_value=1, max_value=1000000, value=10, step=1)
    IV_2006 = st.slider('2006 Invalid Votes' ,min_value=1, max_value=1000000, value=10, step=1)
    NRMVotes_2006 = st.slider('2006 NRM Votes' ,min_value=1, max_value=1000000, value=10, step=1)
    IV_2011 = st.slider('2011 Invalid Votes' ,min_value=1, max_value=1000000, value=10, step=1)
    NRMVotes_2016 = st.slider('2016 NRM Votes' ,min_value=1, max_value=1000000, value=10, step=1)
    NRM_Score_2016  = st.slider('2016 NRM Score' ,min_value=1, max_value=1000000, value=10, step=1)
    Opp_Votes_2016  = st.slider('2016 Opp Votes',min_value=1, max_value=1000000, value=10, step=1)
    
    # Make the prediction and store it when clicked
    if st.button("Predict"):
        result = prediction(IV_2001, NRMVotes_2001, OV_2001, IV_2006, NRMVotes_2006 ,IV_2011, NRMVotes_2016, NRM_Score_2016, Opp_Votes_2016)
        st.success(f'NRM Score for this District is: {result}')
        
if __name__=='__main__': 
    main()
