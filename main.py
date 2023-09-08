from fastapi import FastAPI
from flask_cors import CORS
from flask import Flask, jsonify, request
import Forcast
import Reconfig_models
from datetime import date, timedelta

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET', 'POST'])
def home():
    if (request.method == 'GET'):
        data = "hello world"
        return jsonify({'data': data})



@app.route('/forecast',methods=['GET'])
def ref_Input():
    if request.method == 'GET':
        today = date.today()
        day = today.day
        print("day ",day)
        if day== 30:
            Reconfig_models.Reconfig()

        my_array = Forcast.sendData()

        return jsonify( data_array = my_array)

    else:
        return jsonify(data_array=None)



if __name__ == '__main__':
    app.run(debug=True)
