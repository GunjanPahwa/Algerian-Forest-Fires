from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

#import ridge regressor and standard scaler pickle
with open('ridge.pkl', 'rb') as f:
    ridge_model = pickle.load(f)

with open('scaler.pkl','rb') as f:
    standard_scaler=pickle.load(f)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=["GET","POST"]) #Get means the default page i will be getting and post is the page i will be getting after posting a query
def predict_datapoint():
    if request.method=="POST": #whenever it is post, i would need to interact with my ridge model and give the prediction and get the output
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC =float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        return render_template('home.html',results=result[0])
    else: #if get request, return the default page
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0") #default host address 0.0.0.0 means that this is mapped to local ip address of any machine we are working on



