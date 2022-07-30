from flask import Flask, render_template , request
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import pandas as pd


crop_recommendation_model_path = 'E:/University_Project/Webpage/model/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))



app = Flask(__name__)




@app.route("/")
def index():
    return render_template('index.html')

@app.route("/page10")
def page10():
    return render_template('page10.html')

@app.route("/page11")
def page11():
    return render_template('page11.html')    

@app.route("/page2", methods = ['POST','GET'])
def page2():
    if request.method == "POST":
        y = request.form["y"]
    
    if request.method == "POST":
        d = request.form["d"]
        s = request.form["s"]
    
    if request.method == 'POST':
        a = float(request.form['a'])
        p = float(request.form['p'])
        
    df=pd.read_csv("E:/University_Project/Dataset/Crop/data2.csv")
    df = df.loc[(df['District_Name'] == d) & (df['Season'] == s)]  
    features = df[['Crop_Year','Area','Production_N']]
    target = df['Crop']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.25,random_state =2)
    RF = RandomForestClassifier(n_estimators=20, random_state=0)
    RF.fit(Xtrain,Ytrain)
    predicted_values = RF.predict(Xtest)
    data = np.array([[y,a,p]])
    prediction = RF.predict(data)
    return render_template('page2.html', prediction = prediction[0])

@app.route("/page3", methods = ['POST','GET'])
def page3():
    if request.method == "POST":
        y = request.form["di"]
    
    if request.method == "POST":
        d = request.form["d"]
    
    if request.method == 'POST':
        t = float(request.form['t'])
        a = float(request.form['a'])
        r = float(request.form['r'])
        s = float(request.form['s'])
        h = float(request.form['h'])
        
    
    df=pd.read_csv("E:/University_Project/Dataset/Dataset1.csv")
    df = df.loc[(df['District_Name'] == d)]  
    features = df[['Disease','Area','Temparetue','Rainfall','Sun hours','Humidity']]
    target = df['Crop']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.25,random_state =3)
    RF = RandomForestClassifier(n_estimators=20, random_state=0)
    RF.fit(Xtrain,Ytrain)
    predicted_values = RF.predict(Xtest)
    data = np.array([[y,a,t,r,h,s]])
    prediction = RF.predict(data)
    return render_template('page3.html', prediction = prediction[0])

if __name__=='__main__':
    app.run(debug=True)