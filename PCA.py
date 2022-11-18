from operator import le
import streamlit as st
import pandas as pd
import seaborn as sns
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import datasets
from sklearn.decomposition import PCA



def data_select(df):
    Data_flexible_column = {}
    
    for col_name in cols[:-1]:
        col_checkbox = st.checkbox(col_name)
        if col_checkbox:
            Data_flexible_column[col_name] = df[col_name]

    Data_flexible_column[cols[-1]] = df[cols[-1]]
    Data_use_for_model = pd.DataFrame(Data_flexible_column)    
    
    st.write("Data Selected")
    st.write(Data_use_for_model)
    return Data_use_for_model
    


def Logistic(df):
    

    df = data_select(df)
    
    X_pca = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    
    
    pca = PCA(2)
    pca.fit(X_pca)
    X_pca = pca.transform(X_pca)

    
    
    train_select = st.slider("Choose percentage of train size",0.0,1.0,0.8,0.1)
    X_train,X_test,y_train,y_test = train_test_split(X_pca,y,train_size=train_select,random_state=0)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    accuracy = model.score(X_test, y_test)
    st.write("Accuracy ", accuracy.round(2))
    st.write("Precision: ", precision_score(y_test, y_predict, average='macro'))
    st.write("Recall: ", recall_score(y_test, y_predict, average='macro'))
    st.write("F1-score: ", f1_score(y_test, y_predict, average='macro'))
    
    
    plot_confusion_matrix(model, X_test, y_test)
    st.pyplot()
    


st.title("DATASET")

wine = datasets.load_wine()
df = pd.DataFrame(data= np.c_[wine['data'], wine['target']],
                     columns= wine['feature_names'] + ['target'])
cols=df.columns
st.write(df)
    
    
Logistic(df)
    