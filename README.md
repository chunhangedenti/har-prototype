# Phone HAR Classifier (BBMS3023 teacher prototype)

Three-page Streamlit app for smartphone accelerometer Human Activity Recognition:

1. **Collect** — record labeled `(timestamp, ax, ay, az)` on a phone
2. **Train** — window the signal, extract summary features, train RF / Logistic Regression
3. **Infer** — load the `.joblib` model and predict on new live or CSV data

Train and Infer share `har/features.py` so windowing and features stay consistent.

## Local run

```bash
cd Individual-Assignment/har-prototype
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Laptop tip: use files in `sample_data/` on Train / Infer. Phone motion sensors usually need a real device.

## Streamlit Community Cloud (recommended for Collect)

Phone `DeviceMotion` APIs generally require **HTTPS**. Community Cloud provides that.

1. Push this folder (or the course repo) to GitHub
2. At [share.streamlit.io](https://share.streamlit.io), create an app
3. Set the main file path to `Individual-Assignment/har-prototype/app.py`
   (adjust if your repo root differs)
4. Python version is pinned via `runtime.txt` (`python-3.12`) for Cloud
5. Open the deployed HTTPS URL **on the phone**
6. On Collect: Enable sensors → Record → Stop & send → download CSV

### Classroom notes

- iOS Safari will prompt for motion permission; that must happen from a user tap (handled in the recorder)
- Keep the phone on the same activity for 30–60 seconds per label
- Desktop Chrome/Safari often have no usable accelerometer — that is expected

## CSV format

```text
timestamp,ax,ay,az,label
0.0,0.12,0.05,9.81,sitting
20.0,0.10,0.08,9.79,sitting
```

`timestamp` may be milliseconds or seconds from the start of the recording.

## Model file

The downloaded `.joblib` is a dict bundle:

- `model` — fitted scikit-learn estimator
- `feature_names`, `window_sec`, `overlap`, `sample_rate_hz`, `class_labels`

## Project layout

```text
app.py
pages/1_Collect.py
pages/2_Train.py
pages/3_Infer.py
har/features.py      # shared windowing + features
har/model.py         # train / evaluate / save / predict
har/sensor_bridge.py # phone DeviceMotion → Python batch
har/frontend/        # sealed HTML/JS recorder component
sample_data/         # synthetic CSVs for desktop demos
```
