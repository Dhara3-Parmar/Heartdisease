
import pandas as pd

def validate_input(values, feature_names):
    input_data = pd.DataFrame([values], columns=feature_names)
    return input_data

def predict_result(model, input_data):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    return prediction, probabilities
