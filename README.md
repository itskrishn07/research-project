# 🩻 Chest X-Ray Pneumonia Classifier

A Streamlit web application that uses a trained CNN to classify pediatric chest X-ray images as **Normal** or **Pneumonia**.

---

## 🗂️ Project Structure

```
research_project/
├── app.py                                  # Streamlit application
├── requirements.txt                        # Python dependencies
├── pneumonia_classifier_weighted.keras     # Trained Keras model
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Create & activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🧠 Model Details

| Property        | Value                                          |
|-----------------|------------------------------------------------|
| Input shape     | `(224, 224, 3)` – RGB                          |
| Architecture    | 3× Conv2D → Dense (128) → Dropout → Sigmoid   |
| Optimizer       | Adam                                           |
| Loss            | Binary Cross-Entropy                           |
| Class weighting | NORMAL × 1.94 / PNEUMONIA × 0.67              |
| Test accuracy   | **81%**                                        |
| Pneumonia recall| **98%**                                        |
| Normal recall   | **52%**                                        |

---

## 📊 Dataset

- **Source:** Pediatric Chest X-Ray Images (Kaggle)
- **Classes:** NORMAL (1,341 images) · PNEUMONIA (3,875 images)
- **Imbalance ratio:** ~3 : 1 (handled with class weights during training)

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**.  
It is **not** a substitute for professional medical diagnosis.  
Always consult a qualified radiologist or physician for clinical decisions.
