import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

def ValuePredictor(new_l): 
    to_predict = np.array(new_l).reshape(1,-1)
    loaded_model = pickle.load(open("Rf_model.pkl", "rb")) 
    result = loaded_model.predict(to_predict) 
    return result[0] 

app = Flask(__name__)
@app.route('/')
def home_():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values())
        new_l = list(map(float, to_predict_list))
        result = ValuePredictor(new_l)
        if (result):
            prediction = result
        else:
            prediction = result
            # prediction = 'Sorry no result'          
    return render_template("result.html", prediction = prediction ) 
       

if __name__ == "__main__":
    app.run(debug=True)