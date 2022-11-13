#import libararies
import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Create the heading
st.write("""
#Explore the different ML model and datasets Let's see which is best
""")
# Dataset name
datasets_name = st.sidebar.selectbox(
    "select Dataset",
    ("Iris","Breast Cancer","Wine")
)

#Classifier name
classifier_name = st.sidebar.selectbox(
    "select lassifier",
    ("KNN","SVM","Random Forest")
)

#Import the dataset
def get_dataset(dataset_name):
    data= None
    if dataset_name== "Iris": 
        data = datasets.load_iris()
    elif dataset_name== "Wine":
        data=datasets.load_wine()
    else:
        data= datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x,y
#Now we will call the function and put the variable equal to the x,y
x,y = get_dataset(datasets_name)
# Find the data shape and print its value
st.write("shape of dataset:",x.shape)
st.write("number of classes",len(np.unique(y)))
    
def add_parameter_ui(classifier_name):
    params=dict()
    if classifier_name=="SVM":
        C=st.sidebar.slider("C,0.01,10.0")
        params["C"]=C
    elif classifier_name=="KNN":
        K=st.sidebar.slider("K,1,15")
        params["K"]=K
    else:
        max_depth=st.sidebar.slider("max_depth",2,15)
        params["max_deth"]=max_depth
        n_estimators=st.sidebar.slider("n_estimators",1,100)
        params["n_estimators"]=n_estimators

    return params
params=add_parameter_ui(classifier_name)

def get_classifier(classsifier_name,params):
    clf=None
    if classifier_name=="SVM":
        clf=SVC(C=params["C"])
    elif classifier_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])
    else:
        clf=RandomForestClassifier(n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],random_state=1234)
    return clf
clf=get_classifier(classifier_name,params)

#Apply train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1234)
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
 
acc=accuracy_score(y_test,y_pred)
st.write(f'classifier={classifier_name}')
st.write(f'Accuracy=',acc)

pca=PCA(2)
x_projected=pca.fit_transform(x)
x1= x_projected[:,0]
x2= x_projected[:,1]

fig=plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("principal components 1")
plt.ylabel("principal components 2")

plt.colorbar()
st.pyplot(fig)




 