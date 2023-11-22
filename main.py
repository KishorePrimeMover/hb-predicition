from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from io import StringIO
import numpy as np
from fastapi.staticfiles import StaticFiles
from sklearn.utils import resample

app = FastAPI(debug=True)
# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return {"message": "Go to /static/index.html to view the HTML file"}

def train_models(data: pd.DataFrame):
    X = data[['reading']]
    y = data['label']

     # Train classification models
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    svc_classifier = SVC(probability=True, random_state=42)
    gbc_classifier = GradientBoostingClassifier(random_state=42)

    rf_classifier.fit(X, y)
    svc_classifier.fit(X, y)
    gbc_classifier.fit(X, y)

    # Train regression models
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    svr_regressor = SVR()
    gbr_regressor = GradientBoostingRegressor(random_state=42)

    
    rf_regressor.fit(X, y.replace('/',''))
    svr_regressor.fit(X, y.replace('/',''))
    gbr_regressor.fit(X, y.replace('/',''))

    return rf_classifier, svc_classifier, gbc_classifier, rf_regressor, svr_regressor, gbr_regressor


def predict_with_confidence(data: pd.DataFrame,models, samples, n_bootstraps=100):
    X = data[['reading']]
    y = data['label']
    results = []

    for sample in samples:
        if np.isscalar(sample):
            sample = np.array([[sample]])

        sample_results = {}
        
        for model_name, model in models.items():
            if 'Regressor' in model_name:
                # Directly predict the continuous value
                predicted_value = model.predict(sample)
                
                # Implementing bootstrap for confidence interval estimation
                bootstrap_predictions = []
                for _ in range(n_bootstraps):
                    # Resample the training data
                    X_resampled, y_resampled = resample(X, y)
                    model.fit(X_resampled, y_resampled)
                    bootstrap_predictions.append(model.predict(sample))

                # Calculate confidence interval (e.g., 95%)
                lower_bound = np.percentile(bootstrap_predictions, 2.5)
                upper_bound = np.percentile(bootstrap_predictions, 97.5)
                confidence = (upper_bound - lower_bound) / 2  # Simplified confidence measure

                # Convert NumPy types to Python native types for JSON serialization
                sample_results[model_name] = [sample[0][0].item(), predicted_value[0].item(), confidence.item()]

            else:
                # For classification models, predict class probabilities
                probabilities = model.predict_proba(sample)
                max_prob_index = np.argmax(probabilities, axis=1)
                predicted_class = model.classes_[max_prob_index]
                confidence = probabilities[0, max_prob_index]
                # Convert NumPy types to Python native types for JSON serialization
                sample_results[model_name] = [sample[0][0].item(), predicted_class[0].item(), confidence[0].item()]

        results.append(sample_results)

    return results





@app.post("/predict/")
async def predict(file1: UploadFile = File(...),samples: str = Form(...)):
    # Read the CSV files
    try:
        df1 = pd.read_csv(StringIO(str(await file1.read(), 'utf-8')))
        
        samples_array = np.fromstring(samples, sep=',')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading files: {e}")

    # Combine the dataframes (assuming they have the same structure)
    combined_df = pd.concat([df1], ignore_index=True)

    # Train the models
    rf_classifier, svc_classifier, gbc_classifier, rf_regressor, svr_regressor, gbr_regressor = train_models(combined_df)
    # Combine models in a dictionary
    models = {
        "RF_Classifier": rf_classifier,
        "SVC_Classifier": svc_classifier,
        "GBC_Classifier": gbc_classifier,
        "RF_Regressor": rf_regressor,
        "SVR_Regressor": svr_regressor,
        "GBR_Regressor": gbr_regressor
    }
    # models = {
    #     "RandomForest": rf_model,
    #     "LogisticRegression": lr_model,
    #     "SVC": svc_model,
    #     "GradientBoosting": gbc_model
    # }


    # Get predictions
    predictions = predict_with_confidence(combined_df,models, samples_array)

    return JSONResponse(content={"predictions": predictions})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
