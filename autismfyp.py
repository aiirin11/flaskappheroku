from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np


model_path = "autismfyp_model.sav"  
model = pickle.load(open(model_path, 'rb'))

# Assuming you have a scaler saved as well
scaler_path = "autismscaler.sav"
scaler = pickle.load(open(scaler_path, 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index2.html')  

@app.route('/index1', methods=['POST'])
def medical_info():
    
    age = request.form['age']
    gender = request.form['gender']
    jaundice = request.form['jaundice']
    relation = request.form['relation']
    
    return render_template('index1.html', age=age, gender=gender, jaundice=jaundice, relation=relation)
    
@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        responses = []
        for i in range(1, 11):
            responses.append(int(request.form.get(f'q{i}', 0)))
        
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        jaundice = int(request.form['jaundice'])
        relation = int(request.form['relation'])
        
        input_data = responses + [age, gender, jaundice, relation]
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        # standardize the input data
        std_data = scaler.transform(input_data_reshaped)
        
        prediction = model.predict(std_data)[0]

    except Exception as e:
        result = f"Prediction error: {e}"

    return render_template('index3.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True)

    
