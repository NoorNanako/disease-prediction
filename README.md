# 🩺 AI Disease Prediction System

A desktop application that predicts diseases from symptoms using a **Random Forest** machine learning model, built with Python and Tkinter.

## ✨ Features
- 130+ symptoms to choose from with live search filter
- Trains its own ML model on first run (no external dataset download needed)
- Displays top-5 disease predictions with confidence percentages
- Shows severity level and medical advice for each disease
- 39 diseases covered

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app (auto-trains model on first launch)
python app.py
```

## 📁 File Structure
```
disease_prediction/
├── app.py              # Main GUI application
├── train_model.py      # ML model trainer (called automatically)
├── requirements.txt
└── model/              # Created after first run
    ├── disease_model.pkl
    ├── label_encoder.pkl
    ├── symptoms_list.pkl
    └── disease_symptoms_map.pkl
```

## 🧠 ML Model Details
| Property | Value |
|---|---|
| Algorithm | Random Forest |
| Estimators | 200 trees |
| Features | 130+ binary symptom flags |
| Classes | 39 diseases |
| Typical Accuracy | ~97% on test split |

## 📋 How to Use
1. **Search** for symptoms using the search bar on the left
2. **Check** the symptoms you are experiencing
3. Click **Predict** to get instant AI predictions
4. Review the primary diagnosis and alternative possibilities
5. Click **Reset** to start over

> ⚠️ This is an educational AI tool. Always consult a licensed medical professional.
#
