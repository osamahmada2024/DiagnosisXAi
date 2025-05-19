from flask import Flask, render_template, request,jsonify
from datetime import datetime
import re, os
import pandas as pd
import json
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import tensorflow as tf

app = Flask(__name__)


with open("release_evidences.json",'r') as e:
    Evidence= json.load(e)

# default_values = []
eng_questions = []
# values = []

df = pd.read_csv("cols (1).csv")

messages = pd.read_csv("messages")

@app.route('/leaders')
def index_leader():
    return render_template('TeamPage.html',df = df,messages = messages)

model = tf.keras.models.load_model("model_72%_Final_test_accuracy.h5")

Target = pd.read_csv("Data_Target.csv")["y"]

def Predict(X_new):

    y_pred = model.predict(X_new).flatten()
    top5_idx = np.argsort(y_pred)[-7:][::-1]
    y_pred = y_pred[top5_idx]
    y_pred
    y_pred = pd.DataFrame({
        'Diseases Excepected' : Target[top5_idx],
        'Probability' : y_pred
    })
    colors  = ['#0A84FF','#5E5CE6','#00FFD1','#FF3B30','#FF9500', '#FFD700', '#7CFC00']

    fig1 = px.bar(
        y_pred,
        x='Diseases Excepected',
        y='Probability',
        color='Diseases Excepected',
        color_discrete_sequence=colors,
        template='plotly_dark',
    )

    fig1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(color='white'),
            showgrid=False,
            linecolor='white'
        ),
        yaxis=dict(
            range=[0, 1.05], 
            tickfont=dict(color='white'),
            showgrid=False,
            linecolor='white'
        ),
        legend=dict(font=dict(color='white')),
        bargap=0.2 
    )

    fig1.update_traces(
        marker_line_color="white",
        marker_line_width=1,

        opacity=1.0 
    )

    fig2 = px.pie(
        y_pred,
        names='Diseases Excepected',
        values='Probability',
        color_discrete_sequence=colors
    )
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            font=dict(color='white')
        ),
    )

        
    return y_pred,fig1,fig2
    
   
question_map = {
    meta["question_en"] : meta 
    for meta in Evidence.values()
}

default_values = []
values = []

df.drop([0,1],axis = 0,inplace = True)

for i in range(df.shape[0]):
    meta = question_map.get(df.iloc[i][1])
    default_values.append(meta["default_value"])
    values.append(meta["value_meaning"])



@app.route('/')
def index():
    return render_template('index.html',df = df,default_values = default_values, values = values,result = "")

encoder  = pickle.load(open("encoders.pkl", "rb"))
def encoding(col, value):
    if pd.isna(value):
        return np.array([encoder[col].transform(['unknown'])[0]])
    
    value_list = [value]
    if value not in encoder[col].classes_:
        value_list = ['unknown']
    return encoder[col].transform(value_list)


@app.route('/predict', methods=[ 'POST'])
def index_predict():
    result = None
    symptoms = []
    symptom = request.form.get('age')
    symptoms.append(np.int64(symptom))
    symptom = request.form.get('gender')
    symptom = encoding("SEX",symptom)[0]
    symptoms.append(symptom)
    for i in range(9):  
        symptom = request.form.get(f'symptom{i}')
        symptom = encoding(df['cols'][2+i],symptom)[0]
        symptoms.append(symptom)
    y_pred, fig1, fig2 = Predict(np.array(symptoms).reshape(1,11) )
    y_pred = y_pred.to_dict(orient='records')
    graph1_json = json.dumps(fig1, cls= plotly.utils.PlotlyJSONEncoder)
    graph2_json = json.dumps(fig2, cls= plotly.utils.PlotlyJSONEncoder)
    print(y_pred)
    return render_template('index.html', df = df,default_values = default_values, values = values, graph1_json= graph1_json,
                            predictions = y_pred ,graph2_json=graph2_json)

@app.route('/log',methods = ['Post'])
def log_messages ():
    data = request.get_json()
    
    name = data.get('name','').strip()
    email = data.get('email','').strip()
    message = data.get('message','').strip()
    
    if not name or not email or not message :
        return jsonify(success = False,error = "please Fill all Fields"),400
    
    if not re.match(r'^\S+@\S+.\S+$',email):
        return jsonify(success = False,error = "Email is Not Valid"),400
    

    rows = {
        'time':datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'name' : name,
        'email' : email,
        'message' : message,
    }
    
    df = pd.DataFrame([rows])
    df.to_csv(
        "messages.csv",
        mode = 'a',
        header = not os.path.exists("messages.csv"),
        index = False,
        encoding = 'utf-8'
    )
    return jsonify(success = True),200


if __name__ == "__main__":
    app.run(port = 5000,debug = True)