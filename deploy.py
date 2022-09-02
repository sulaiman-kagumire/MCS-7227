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
      # Now, we will give the title to out web page  
    smt.title("NRM President Score per District")  
        
    # Now, we will be defining some of the frontend elements of our web           
    # page like the colour of background and fonts and font    size, the padding and    
    # the text to be displayed  
    html_temp = """  
    <div style = "background-colour: #FFFF00; padding: 16px">  
    <h1 style = "color: #000000; text-align: centre; "> NRM Presidential score per District   
     </h1>  
    </div>  
    """  
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    
    # allow user input 
#     2001_IV = st.selectbox('Gender',('Male','Female'))
    IV_2001  = st.text_input('2001 Invalid Votes', " Type Here")
    NRMVotes_2001  = st.text_input('2001 NRM Votes', " Type Here")
    OV_2001  = st.text_input('2001 Opposition Votes', " Type Here")
    IV_2006 = st.text_input('2006 Invalid Votes', " Type Here")
    NRMVotes_2006 = st.text_input('2006 NRM Votes', " Type Here")
    IV_2011 = st.text_input('2011 Invalid Votes', " Type Here")
    NRMVotes_2016 = st.text_input('2016 NRM Votes', " Type Here")
    NRM_Score_2016  = st.text_input('2016 NRM Score', " Type Here")
    Opp_Votes_2016  = st.text_input('2016 Opp Votes', " Type Here")
    
    # Make the prediction and store it when clicked
    if st.button("Predict"):
        result = prediction(IV_2001, NRMVotes_2001, OV_2001, IV_2006, NRMVotes_2006 ,IV_2011, NRMVotes_2016, NRM_Score_2016, Opp_Votes_2016)
        st.success(f'NRM Score for this District is: {result}')
        
if __name__=='__main__': 
    main()
