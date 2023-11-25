from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from io import StringIO
import numpy as np
from fastapi.staticfiles import StaticFiles

app = FastAPI(debug=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return {"message": "Go to /static/index.html to view the HTML file"}

def train_model(data: pd.DataFrame):
    X = data[['reading']]
    y = data['label']

    # Train GradientBoostingRegressor model
    gbr_regressor = GradientBoostingRegressor(random_state=42)
    gbr_regressor.fit(X, y.replace('/',''))
    return gbr_regressor

def predict_label(data: pd.DataFrame, model, samples):
    X = data[['reading']]
    results = []

    for sample in samples:
        if np.isscalar(sample):
            sample = np.array([[sample]])

        # Directly predict the continuous value
        predicted_value = model.predict(sample)
        # Convert NumPy types to Python native types for JSON serialization
        sample_results = {"Method3": predicted_value[0].item()}
        results.append(sample_results)

    return results

@app.post("/predict/")
async def predict(file1: UploadFile = File(...), samples: str = Form(...)):
    # Read the CSV file
    try:
        df1 = pd.read_csv(StringIO(str(await file1.read(), 'utf-8')))
        samples_array = np.fromstring(samples, sep=',')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading files: {e}")

    # Train the model
    gbr_regressor = train_model(df1)

    # Get predictions
    predictions = predict_label(df1, gbr_regressor, samples_array)

    return JSONResponse(content={"predictions": predictions})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
