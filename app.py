from flask import *
import requests
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

df = pd.read_csv("dataset/cleaned_symptoms.csv")
next_df = None
prev_symptom = None
current_symptom = None

@app.route('/',methods=["GET","POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        
        global prev_symptom
        global flag
        global messages
        global current_symptom
        final = False
        if flag==0:
            details = {
                "type": "first",
                "symptom": user_input,
                "truth_value": 1
            }
            flag=1
            resp = requests.post(url="http://0.0.0.0:8040/get/next/symptom",data=json.dumps(details),headers={'content-type': 'application/json'})
            messages.append({"by":"user","text":user_input})
            current_symptom = user_input
            resp = json.loads(resp.text)
            

        else:
            details = {
                "type": "next",
                "symptom": current_symptom,
                "truth_value": 1 if user_input=="yes" else 0
            }
            messages.append({"by":"user","text":user_input})
            resp = requests.post(url="http://0.0.0.0:8040/get/next/symptom",data=json.dumps(details),headers={'content-type': 'application/json'})
            resp = json.loads(resp.text)
            prev_symptom = current_symptom
            

        if resp["status"]=="symptom":
            messages.append({"by":"bot","text":"Are you experiencing %s"%(resp["value"])})
            current_symptom = resp["value"]
        else:
            df = pd.read_csv("./dataset/symptom_Description.csv")
            desc = df[df["Disease"]==resp["value"]]["Description"].to_list()[0]
            
            messages.append({"by":"bot","text":"You might have %s. %s"%(resp["value"],desc)})
            final=True
        
        return render_template("index.html",messages=messages,final=final)
        
        
    flag = 0
    messages = []
    return render_template("index.html", final=False)

@app.route('/get/next/symptom',methods=["POST"])
def next_symptom():
    X = df.drop(columns=["disease"])
    Y = df["disease"]
    global next_df
    
    coeff_df =pd.DataFrame(mutual_info_classif(X, Y).reshape(-1, 1),
                         columns=['Coefficient'], index=X.columns)
    
    data = request.json
    
    if data["type"]=="first":
        best_symptom = data["symptom"]
        truth_value = 1
        next_df = df.copy(deep=True)
        
    else:
        truth_value = int(data["truth_value"])
        best_symptom = data["symptom"]
        
    resp,next_df = select_next_symptom(best_symptom,truth_value,next_df)
    
    return resp


def get_best_feature_from_IG(df):
    X = df.drop(columns=["disease"])
    Y = df["disease"]
    coeff_df =pd.DataFrame(mutual_info_classif(X, Y).reshape(-1, 1),
                         columns=['Coefficient'], index=X.columns)
    return coeff_df.sort_values(by='Coefficient', ascending=False).iloc[0].name

def select_next_symptom(current_symptom, truth_value, current_df):
    new_df = current_df[current_df[current_symptom]==truth_value]
    if len(new_df["disease"].unique())==1:
        return {"status":"disease","value":list(new_df["disease"])[0]},None
    
    best_feature = get_best_feature_from_IG(new_df)
    return {"status":"symptom","value":best_feature}, new_df.drop(columns=[current_symptom])
    

if __name__== "__main__":
    app.run(host="0.0.0.0", debug = True, port = 8040)