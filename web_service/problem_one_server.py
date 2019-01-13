from flask import Flask, jsonify, request
import numpy as np
from sklearn.externals import joblib
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


one_model = joblib.load('knn_one_v1.pkl')

# loading the encoders
gender_label = LabelEncoder()
nationality_encoder = LabelEncoder()
gender_label.classes_ = np.load('gender_classes.npy')
nationality_encoder.classes_ = np.load('nationality_classes.npy')

# loading the scaler
scaler = pickle.load(open("scaler_one.sav", 'rb'))

@app.route('/test', methods=['GET'])
def profil_prediction2():
    response = {
        'prediction': "Hello"
    }
    return jsonify(response), 200


@app.route('/profile/<gender>/<nationality>/<bac_wilaya>/<bac_average>/<age>', methods=['GET'])
def profil_prediction( gender , nationality , bac_wilaya , bac_average , age):


    profile = np.array([gender_label.transform([gender])[0],
                       nationality_encoder.transform([bool(int(nationality))])[0],
                       int(bac_wilaya),float(bac_average),int(age)])
    new_profile = scaler.transform([profile])
    response = {
        'prediction':  int(one_model.predict(new_profile)[0])
    }
    
    return jsonify(response), 200


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=4000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    app.run(host='0.0.0.0', port=5000)