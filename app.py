
import streamlit as st
import pandas as pd
import pytesseract
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Allergy Detector", layout="centered")

st.title("🥛 AI-Based Allergy Detection App")

# Upload dataset
uploaded_dataset = st.file_uploader("📄 Upload Allergen Dataset (CSV)", type=["csv"])

if uploaded_dataset:
    df = pd.read_csv(uploaded_dataset)
    if "Food_Name" not in df.columns or "Allergic" not in df.columns:
        st.error("❌ Please upload a dataset with 'Food_Name' and 'Allergic' columns.")
    else:
        df["Food_Label"] = df["Food_Name"].astype("category").cat.codes
        X = df[["Food_Label"]]
        y = df["Allergic"]
        if len(df["Allergic"].unique()) < 2:
            st.error("❌ Dataset must have both allergic (1) and non-allergic (0) samples.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            st.success("✅ AI Model Trained Successfully!")

            uploaded_image = st.file_uploader("🖼️ Upload Food Label Image", type=["png", "jpg", "jpeg"])
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption='Uploaded Image', use_column_width=True)

                # Convert image for OpenCV
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                kernel = np.ones((1, 1), np.uint8)
                gray = cv2.dilate(gray, kernel, iterations=1)
                gray = cv2.erode(gray, kernel, iterations=1)

                # OCR
                text = pytesseract.image_to_string(gray, config='--psm 6')
                st.subheader("🔍 Extracted Text:")
                st.write(text)

                # Match allergens
                allergen_list = df["Food_Name"].tolist()
                matched = [word for word in text.split() if word.lower() in [a.lower() for a in allergen_list]]

                if matched:
                    st.warning(f"⚠️ Allergens Detected: {', '.join(matched)}")
                    st.subheader("🧪 Final Result: **Allergic**")
                else:
                    st.success("✅ No allergens found. Safe to consume.")
                    st.subheader("🧪 Final Result: **Not Allergic**")
