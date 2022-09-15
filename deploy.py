import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, cross_val_predict, ShuffleSplit

from sklearn.svm import LinearSVR, NuSVR, SVR

import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor

from xgboost import XGBRFRegressor, XGBRegressor

from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, mean_squared_log_error
import pickle as pkl
import streamlit as st
import numpy as np

df3 = pd.read_csv('dfp.csv')

y2 = df3['2021_NRM_Score']
X = df3.drop('2021_NRM_Score', axis=1)

x_scaled = StandardScaler().fit_transform(X)
# pca = PCA(n_components=7)
# x_pca = pca.fit_transform(x_scaled)

xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y2, test_size=0.2, random_state=42)

# xgr = XGBRegressor()
# # cbr = CatBoostRegressor()
# lgr = LGBMRegressor()

# rf = RandomForestRegressor()
gbr = GradientBoostingRegressor()
# exr  = ExtraTreesRegressor()
# dt = DecisionTreeRegressor()
# lsvr = LinearSVR()
# nsvr = NuSVR()
# svr = SVR()
# knr = KNeighborsRegressor()

# xgr = xgr.fit(xtrain, ytrain)
# # cbr.fit(xtrain, ytrain)
# lgr = lgr.fit(xtrain, ytrain)
# rf = rf.fit(xtrain, ytrain)
gbr = gbr.fit(xtrain, ytrain)
# exr = exr.fit(xtrain, ytrain)
# dt = dt.fit(xtrain, ytrain)
# lsvr.fit(xtrain, ytrain)
# nsvr = nsvr.fit(xtrain, ytrain)
# svr = svr.fit(xtrain, ytrain)
# knr = knr.fit(xtrain, ytrain)

# print(xgr.score(xtrain, ytrain))
# # print(cbr.score(xtrain, ytrain))
# print(lgr.score(xtrain, ytrain))
# print(rf.score(xtrain, ytrain))
# print(gbr.score(xtrain, ytrain))
# print(exr.score(xtrain, ytrain))
# print(dt.score(xtrain, ytrain))
# print(lsvr.score(xtrain, ytrain))
# print(nsvr.score(xtrain, ytrain))
# print(svr.score(xtrain, ytrain))
# print(knr.score(xtrain, ytrain))
# pickle.dump(gbr, open('model.pkl', 'wb'))
pickle.dump(gbr, open('presidential_model.pkl', 'wb'))

pickle_in = open('presidential_model.pkl', 'rb') 
classifier = pkl.load(pickle_in)

# @st.cache(allow_output_mutation=True)

# '2001_IV', '2001_NRMVotes', '2001_OV', '2006_IV', '2006_NRMVotes', '2011_IV', '2016_NRMVOTES', '2016_NRM_Score', '2016_Opp_Votes'
def prediction(IV_2011, NRM_Score_2011, NRMVotes_2016, OppVotes_2016):
    
    
    predictions = classifier.predict(
        [[IV_2011, NRM_Score_2011, NRMVotes_2016, OppVotes_2016]])
    

    print(predictions)
         
    return predictions
# main function defines our webpage
def main():
    html_temp = """  
    <div style = "background-colour: Black; padding: 16px">  
    <h1 style = "color: Red; text-align: centre; "> NRM Presidential Score Predictor App   
     </h1>  
    </div>  
    """ 
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    
    # allow user input 
#         2001_IV = st.selectbox('Gender',('Male','Female'))
#     IV_2001  = st.slider('IV 2001',min_value=1, max_value=100000, value=1, step=1)  
#     NRMVotes_2001  = st.slider(' NRMVotes_2001',min_value=1, max_value=1000000, value=1, step=1)  
#     OV_2001  = st.slider('OV_2001',min_value=1, max_value=100, value=1000000, step=1)
#     IV_2006 = st.slider('IV_2006',min_value=1, max_value=100, value=1000000, step=1)
#     NRMVotes_2006 = st.slider('NRMVotes_2006',min_value=1, max_value=1000000, value=1, step=1)
    IV_2011 = st.slider('IV_2011',min_value=1, max_value=100, value=1000000, step=1)
    NRM_Score_2011  =st.slider('NRM_Score_2011',min_value=0.0, max_value=1.0, value=0.01, step=0.01)
    NRMVotes_2016 = st.slider('NRMVotes_2016',min_value=1, max_value=1000000, value=1, step=1) 
    OppVotes_2016  = st.slider('Opp_Votes_2016',min_value=1, max_value=1000000, value=1, step=1)
    
#     IV_2011, NRM_Score_2011, NRMVotes_2016, OppVotes_2016
    
    # Make the prediction and store it when clicked
    if st.button("Predict"):
        result = prediction(IV_2011, NRM_Score_2011, NRMVotes_2016, OppVotes_2016)
        st.success(f'2021 NRM Score for this District is: {100*(np.round(result,3))}%')
        
if __name__=='__main__': 
    main()
    
