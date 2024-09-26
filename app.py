from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the models and scaler
logistic_model = joblib.load('models/logistic_regression_model.pkl')
random_forest_model = joblib.load('models/random_forest_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load expected feature columns
with open('encoded_columns.txt', 'r') as f:
    expected_columns = [line.strip() for line in f.readlines()]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form
        duration = float(request.form['duration'])
        src_bytes = float(request.form['src_bytes'])
        dst_bytes = float(request.form['dst_bytes'])
        logged_in = int(request.form['logged_in'])
        wrong_fragment = int(request.form['wrong_fragment'])
        same_srv_rate = float(request.form['same_srv_rate'])
        srv_count = int(request.form['srv_count'])
        protocol_type = request.form['protocol_type']
        service = request.form['service']
        flag = request.form['flag']
        
        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'duration': [duration],
            'src_bytes': [src_bytes],
            'dst_bytes': [dst_bytes],
            'logged_in': [logged_in],
            'wrong_fragment': [wrong_fragment],
            'same_srv_rate': [same_srv_rate],
            'srv_count': [srv_count],
            'protocol_type': [protocol_type],
            'service': [service],
            'flag': [flag]
        })

        # One-hot encode the input data
        input_data_encoded = pd.get_dummies(input_data, columns=['protocol_type', 'service', 'flag'], drop_first=True)
        input_data_encoded = input_data_encoded.reindex(columns=expected_columns, fill_value=0)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data_encoded)

        # Make predictions with each model
        logistic_prediction = logistic_model.predict(input_data_scaled)
        random_forest_prediction = random_forest_model.predict(input_data_encoded)
        svm_prediction = svm_model.predict(input_data_scaled)

        # Prepare the result
        result = {
            'logistic_regression': logistic_prediction[0],
            'random_forest': random_forest_prediction[0],
            'svm': svm_prediction[0]
        }
        
        return render_template('index.html', prediction=result)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
