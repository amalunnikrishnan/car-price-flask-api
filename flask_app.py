from flask import Flask, request, jsonify
import joblib


app = Flask(__name__)

def load_object(filename):
    with open(filename ,'rb') as f:
        loaded = joblib.load(f)
    return loaded

# prediction function route
@app.route('/predict', methods=['POST'])
def predict():
    
    # load the model
    model, _ = load_object('model.pkl')
    
    # get the data from the POST request
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
