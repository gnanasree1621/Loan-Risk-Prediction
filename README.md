# LoanLens — Streamlit + MongoDB

## Run locally
1. Install packages: `pip install -r requirements.txt`
2. Put `loan_risk_model.pkl` next to `app.py`.
3. Run: `streamlit run app.py`

## MongoDB Atlas setup
1. Create a free cluster at https://www.mongodb.com/atlas/database.
2. Create a database user and allow network access for your deployment.
3. Copy the application connection URI from Atlas.
4. For local use, set the environment variable in PowerShell before running Streamlit:
   `$env:MONGODB_URI="mongodb+srv://USER:PASSWORD@YOUR_CLUSTER/"`
5. Optional variables:
   `$env:MONGODB_DATABASE="loanlens"`
   `$env:MONGODB_COLLECTION="predictions"`
6. Restart Streamlit. The sidebar should show MongoDB connected.

For Streamlit Community Cloud, open App settings → Secrets and add:
MONGODB_URI = "mongodb+srv://USER:PASSWORD@YOUR_CLUSTER/"
MONGODB_DATABASE = "loanlens"
MONGODB_COLLECTION = "predictions"

Never commit the MongoDB URI, password, or other secrets to GitHub. Keep the database user limited to the required database. For a public deployment, configure network access securely; do not expose database credentials.

## Data stored
The app saves the predicted class, model-estimated default probability, annual income, credit amount, and timestamp. It does not ask for names, phone numbers, or addresses. Consider storing less information if the extra financial values are not required.

## Important
The model input dates are converted to relative-day values. Verify the date conventions and feature order against the model training pipeline. Prediction history is persistent only when MongoDB is successfully configured; otherwise, it uses session-only fallback.
