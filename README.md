# EcoExposure Render Inference v1

Real-time backend for the EcoExposure internal data-collection app.

Pipeline:
Top photo -> V2 beaker detection -> standardized beaker -> ROI60 ->
18 optical metrics -> water-type Random Forest -> prediction JSON.

## 1. Upload these files to GitHub under `inference/`

Keep the directory structure in this ZIP.

## 2. Create the model files locally

From your Windows `inference` folder:

```powershell
python train_models.py
```

This creates:

```text
models/filtered_rf_v1.joblib
models/salt_rf_v1.joblib
```

Upload both files to GitHub under `inference/models/`.

## 3. Render settings

Root Directory:
```text
inference
```

Build Command:
```text
pip install -r requirements.txt
```

Start Command:
```text
uvicorn EcoExposure_Inference_Server:app --host 0.0.0.0 --port $PORT
```

## 4. Render environment variables

Set in Render (do NOT put the service-role secret in GitHub):

```text
SUPABASE_URL=https://qvmgrchkmmezpwhlfxjv.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<your secret service-role key>
ALLOWED_ORIGINS=https://melinda-ecoteraasia.github.io
```

## 5. Test

Open:
```text
https://YOUR-SERVICE.onrender.com/health
```

It should show `supabase_configured: true` and the filtered/salt model files as present.

## 6. Connect index.html

Change:

```javascript
const MODEL_API_URL = "";
```

to:

```javascript
const MODEL_API_URL =
  "https://YOUR-SERVICE.onrender.com/predict";
```

Then the existing Development Model toggle will call the live backend after each sample is saved.
