from flask import Flask, request, jsonify
import pickle5 as pickle
import pandas as pd
from preprocess import all_preprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, from main route!'

@app.route('/prediction', methods=['POST'])
def prediction():
    try:        
        data = request.get_json()
        df_single_person = pd.DataFrame([data])
        df_single_person = all_preprocess(df_single_person)

    
        # Get the current directory
        current_directory = os.path.dirname(__file__)

        # Path to your model file
        model_path = os.path.join(current_directory, 'model.pkl')

        model = None
        prediction = None
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            prediction = model.predict(df_single_person)
        
        return jsonify({'prediction': prediction.tolist()[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
