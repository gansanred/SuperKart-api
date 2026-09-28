import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask framework app wrapper
sales_prediction_api = Flask("Product Store Sales Predictor")

# Load complete serialized scikit-learn pipeline container
PIPELINE_PATH = "product_sales_pipeline_v1.joblib"
if os.path.exists(PIPELINE_PATH):
    model_pipeline = joblib.load(PIPELINE_PATH)
else:
    raise FileNotFoundError(f"Critical Error: {PIPELINE_PATH} missing from backend folder root.")

@sales_prediction_api.get('/')
def health_check():
    """Liveness check for cloud orchestration load balancers."""
    return "Product Store Sales Prediction Microservice Online."

@sales_prediction_api.post('/v1/predict/single')
def predict_single_item():
    """
    Handles payload scoring operations for a single transaction.
    Expects JSON data corresponding to schema structural format.
    """
    payload = request.get_json()
    if not payload:
        return jsonify({'error': 'Missing JSON entity wrapper inside message body'}), 400

    try:
        # Parse elements directly mapping fields to schema matching format
        transaction_sample = {
            'Product_Id': payload.get('Product_Id'),
            'Product_Weight': float(payload['Product_Weight']),
            'Product_Sugar_Content': payload.get('Product_Sugar_Content'),
            'Product_Allocated_Area': float(payload['Product_Allocated_Area']),
            'Product_Type': payload.get('Product_Type'),
            'Product_MRP': float(payload['Product_MRP']),
            'Store_Id': payload.get('Store_Id'),
            'Store_Establishment_Year': int(payload['Store_Establishment_Year']),
            'Store_Size': payload.get('Store_Size'),
            'Store_Location_City_Type': payload.get('Store_Location_City_Type'),
            'Store_Type': payload.get('Store_Type')
        }
    except KeyError as missing_key:
        return jsonify({'error': f"Required item column schema declaration missing: {missing_key}"}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'Datatype parsing error. Verify numerical attributes formatting.'}), 400

    # Vectorize payload tracking shape structure
    input_dataframe = pd.DataFrame([transaction_sample])

    # Compute prediction tracking inference data matrix array
    raw_prediction = model_pipeline.predict(input_dataframe)[0]
    final_sales_prediction = round(float(raw_prediction), 2)

    return jsonify({
        'Predicted_Store_Sales_Total': max(0.0, final_sales_prediction)
    })

@sales_prediction_api.post('/v1/predict/batch')
def predict_batch_csv():
    """
    Processes high-throughput batch arrays streaming via file uploads.
    Expects multipart form-data payload with a raw CSV attachment.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file attachment identified in request payload data.'}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'Empty target file processing request rejected.'}), 400

    try:
        # Stream parse directly into internal processing frame memory cache
        batch_dataframe = pd.read_csv(uploaded_file)

        # Verify required keys mapping columns exist
        if 'Product_Id' not in batch_dataframe.columns:
            return jsonify({'error': "Key index column 'Product_Id' must be present in batch csv configuration schema"}), 400

        # Execute structural evaluation passes
        computed_predictions = model_pipeline.predict(batch_dataframe)

        # Vectorized assembly array loop optimization
        batch_dataframe['Predicted_Sales'] = np.clip(computed_predictions, 0, None).round(2)

        # Transform structural series targets directly to serialized response dictionary
        response_payload = dict(zip(batch_dataframe['Product_Id'].astype(str), batch_dataframe['Predicted_Sales']))

        return jsonify(response_payload)

    except Exception as operational_failure:
        return jsonify({'error': f"Batch engine initialization failed structural processing loop: {str(operational_failure)}"}), 500

if __name__ == '__main__':
    # Explicitly configured for open subnet bounds mapping inside container networks
    sales_prediction_api.run(host='0.0.0.0', port=7860, debug=True)
