import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
import numpy as np

import pandas as pd
from flask import Flask, request, render_template
import pickle
from keras.models import load_model

app = Flask(__name__)
model = pickle.load(open("supervised.pkl", "rb"))

#scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    #input_features = [120, 200, 50, 200, 4, 80, 1, 1, 1, 1]
    features_value = [np.array(input_features)]
    print("Captured feature vector : ", features_value)
    
    features_name = [ 'sysBP',
       'glucose',
       'age',
       'totChol',
       'cigsPerDay',
      'diaBP',
        'prevalentHyp',
       'diabetes',
      'BPMeds',
       'male'
     ]
                     
                       
    df = pd.DataFrame(features_value, columns=features_name)
    #df = scaler.transform(df)
    output = model.predict(df)
        
    if output == 1:
        res_val = "** likely to develop heart disease within a ten year interval **"
    else:
        res_val = "not likely to develop heart disease within a ten year interval "
        

    return render_template('index.html', prediction_text='Patient is  {}'.format(res_val))

if __name__ == "__main__":
    app.run()
