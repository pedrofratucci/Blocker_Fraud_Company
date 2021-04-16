import pickle
import pandas as pd
from flask import Flask, request, Response
from classes.fraud_detection import Fraud_Detection
import xgboost


# loading model trained from pickle file
with open('models/xgb_model_tuned.pkl', 'rb') as file:
    model = pickle.load(file)

# initialized API
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def fraud_predict():
    test_json = request.get_json()

    if test_json:

        # Unique Example
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])

        # multiple Example
        else:  # multiple Example
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())


        # Instantiate BF class
        pipeline = Fraud_Detection()

        # manipulate features
        test_raw_features = pipeline.features_engineering(test_raw)

        # rescale and encode features to predict
        test_raw_prepared = pipeline.data_preparation(test_raw_features)

        # prediction
        df_response = pipeline.get_predict(model, test_raw_prepared)

        return df_response

    else:

        return Response("{}", status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='192.168.0.9', port='5000')