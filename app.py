from flask import Flask,request,jsonify,render_template
import json
import numpy as np
import pickle
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


import templates

app=Flask(__name__)
model=pickle.load(open("breast_cancer_pred.pickle",'rb'))
data=json.load(open("data.json",'r'))['data_col']


@app.route('/')

def home():
    if(request.method=="GET"):
        return render_template('index.html')


@app.route('/predict',methods=['POST'])

def prediction():
    features=[float(x) for x in request.form.values()]
    feature_value=[np.array(features)]
    features_name=data
    df=pd.DataFrame(feature_value,columns=features_name)
    output=model.predict(df)

    return render_template('classify.html',value=output)





if __name__=="__main__":
    print("Starting flask server for Breast Cancer prediction")
    app.run(debug=True)
