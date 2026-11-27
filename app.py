"""
BBMS3023 Digital Health — Phone HAR teacher prototype.

Open the sidebar pages:
  1. Collect — record labeled accelerometer CSV from a phone
  2. Train   — window, extract features, train and download a model
  3. Infer   — load the model and predict on new data
"""

import streamlit as st

st.set_page_config(
    page_title="Phone HAR Classifier",
    page_icon="📱",
    layout="centered",
)

st.title("Phone Human Activity Recognition")
st.write(
    """
This prototype classifies **sitting / walking / running** from phone
accelerometer data.

Use the sidebar pages in order:

1. **Collect** — record labeled sensor data on a phone (HTTPS / Streamlit Cloud)
2. **Train** — window the signal, extract summary features, train a classifier
3. **Infer** — load the saved model and predict on a new recording

Laptop tip: if you do not have phone access yet, use the bundled files in
`sample_data/` on the Train and Infer pages.
"""
)

st.info(
    "Teaching focus: Train and Infer must use the **same** windowing and "
    "feature code (`har/features.py`). That train–serve consistency is the "
    "main applied-ML lesson."
)
