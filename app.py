import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from sklearn.preprocessing import StandardScaler

from src.utils import get_cabin_desk, get_cabin_side 

application=Flask(__name__)

app=application

# Route for a home page

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data=CustomData(
            
            age= int(request.form.get("age")),
            roomService= float(request.form.get("roomService")),
            foodCourt= float(request.form.get("foodCourt")),
            shoppingMall= float(request.form.get("shoppingMall")),
            spa= float(request.form.get("spa")),
            vRDeck= float(request.form.get("vRDeck")),
            
            homePlanet=request.form.get("homePlanet") ,
            cryoSleep=bool(request.form.get("cryoSleep")) ,
            destination=request.form.get("destination") ,
            vIP=bool(request.form.get("vIP")) ,
            cabin=request.form.get("cabin")

            )

        pred_df=data.get_data_as_data_frame()
        pred_df['cabin_desk'] = pred_df['Cabin'].apply(get_cabin_desk)
        pred_df['cabin_side'] = pred_df['Cabin'].apply(get_cabin_side)

        pred_df.drop(columns=['Cabin'],  inplace=True)

        print(pred_df)
        

        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("index.html", results= int(results[0]))


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)