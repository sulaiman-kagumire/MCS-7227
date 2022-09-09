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
    html_temp = """  
    <div style = "background-colour: black; padding: 16px">  
    <h1 style = "color: Yellow; text-align: centre; "> NRM Presidential Score Predictor App   
     </h1>  
    </div>  
    """ 
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    
    # allow user input 
#         2001_IV = st.selectbox('Gender',('Male','Female'))
    IV_2001  = st.slider('IV 2001',min_value=1, max_value=100000, value=1, step=1)  
    NRMVotes_2001  = st.slider(' NRMVotes_2001',min_value=1, max_value=1000000, value=1, step=1)  
    OV_2001  = st.slider('OV_2001',min_value=1, max_value=100, value=1000000, step=1)
    IV_2006 = st.slider('IV_2006',min_value=1, max_value=100, value=1000000, step=1)
    NRMVotes_2006 = st.slider('NRMVotes_2006',min_value=1, max_value=1000000, value=1, step=1)
    IV_2011 = st.slider('IV_2011',min_value=1, max_value=100, value=1000000, step=1)
    NRMVotes_2016 = st.slider('NRMVotes_2016',min_value=1, max_value=1000000, value=1, step=1) 
    NRM_Score_2016  =st.slider('NRM_Score_2016',min_value=0.0, max_value=1.0, value=0.01, step=0.01)
    Opp_Votes_2016  = st.slider('Opp_Votes_2016',min_value=1, max_value=1000000, value=1, step=1)
    
    # Make the prediction and store it when clicked
    if st.button("Predict"):
        result = prediction(IV_2001, NRMVotes_2001, OV_2001, IV_2006, NRMVotes_2006 ,IV_2011, NRMVotes_2016, NRM_Score_2016, Opp_Votes_2016)
        st.success(f'NRM Score for this District is: {list(result)}')
        
if __name__=='__main__': 
    main()
    
