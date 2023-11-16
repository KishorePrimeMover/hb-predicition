from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from io import StringIO
import numpy as np
from fastapi.staticfiles import StaticFiles

app = FastAPI(debug=True)
# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return {"message": "Go to /static/index.html to view the HTML file"}

def train_models(data: pd.DataFrame):
    X = data[['reading']]
    y = data['label']

    # Train different models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    svc_model = SVC(probability=True, random_state=42)
    gbc_model = GradientBoostingClassifier(random_state=42)

    rf_model.fit(X, y)
    
    svc_model.fit(X, y)
    gbc_model.fit(X, y)

    return rf_model, svc_model, gbc_model


def predict_with_confidence(models, samples):
    results = []

    for sample in samples:
        if np.isscalar(sample):
            sample = np.array([[sample]])

        sample_results = {}
        
        for model_name, model in models.items():
            # Predict class probabilities
            probabilities = model.predict_proba(sample)

            # Find the class with the highest probability
            max_prob_index = np.argmax(probabilities, axis=1)
            predicted_class = model.classes_[max_prob_index]
            confidence = probabilities[0, max_prob_index]
            
            sample_results[model_name] = (sample[0][0],predicted_class[0], confidence[0])

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
    rf_model, svc_model, gbc_model = train_models(combined_df)
    models = {
        "Method1": rf_model,
        
        "Method2": svc_model,
        "Method3": gbc_model
    }
    # models = {
    #     "RandomForest": rf_model,
    #     "LogisticRegression": lr_model,
    #     "SVC": svc_model,
    #     "GradientBoosting": gbc_model
    # }


    # Get predictions
    predictions = predict_with_confidence(models, samples_array)

    return JSONResponse(content={"predictions": predictions})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
