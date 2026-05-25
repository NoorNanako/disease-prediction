"""
Disease Prediction Model Trainer
Trains a Random Forest classifier on symptom data and saves the model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# ─── Expanded symptom-disease dataset ────────────────────────────────────────
DISEASES = {
    "Fungal infection":     ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"],
    "Allergy":              ["continuous_sneezing", "shivering", "chills", "watering_from_eyes", "runny_nose"],
    "GERD":                 ["acidity", "ulcers_on_tongue", "vomiting", "indigestion", "stomach_pain"],
    "Chronic cholestasis":  ["itching", "vomiting", "yellowish_skin", "nausea", "abdominal_pain"],
    "Drug Reaction":        ["itching", "skin_rash", "stomach_pain", "burning_micturition", "fatigue"],
    "Peptic ulcer disease": ["vomiting", "indigestion", "headache", "back_pain", "loss_of_appetite"],
    "AIDS":                 ["muscle_wasting", "patches_in_throat", "high_fever", "extra_marital_contacts"],
    "Diabetes":             ["fatigue", "weight_loss", "restlessness", "polyuria", "family_history", "excessive_hunger"],
    "Gastroenteritis":      ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"],
    "Bronchial Asthma":     ["fatigue", "cough", "high_fever", "breathlessness", "mucoid_sputum"],
    "Hypertension":         ["headache", "chest_pain", "dizziness", "loss_of_balance", "lack_of_concentration"],
    "Migraine":             ["headache", "nausea", "vomiting", "visual_disturbances", "blurred_and_distorted_vision"],
    "Cervical spondylosis": ["neck_pain", "dizziness", "back_pain", "weakness_in_limbs"],
    "Paralysis":            ["vomiting", "headache", "weakness_of_one_body_side", "altered_sensorium"],
    "Jaundice":             ["itching", "vomiting", "fatigue", "weight_loss", "high_fever", "yellowish_skin", "dark_urine"],
    "Malaria":              ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "muscle_pain"],
    "Chicken pox":          ["itching", "skin_rash", "fatigue", "lethargy", "high_fever", "headache", "loss_of_appetite"],
    "Dengue":               ["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "pain_behind_the_eyes", "nausea"],
    "Typhoid":              ["chills", "vomiting", "fatigue", "high_fever", "headache", "nausea", "constipation", "abdominal_pain"],
    "Hepatitis A":          ["joint_pain", "vomiting", "fatigue", "high_fever", "nausea", "loss_of_appetite", "abdominal_pain", "diarrhoea", "mild_fever", "yellowing_of_eyes"],
    "Hepatitis B":          ["itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine", "abdominal_pain", "loss_of_appetite"],
    "Hepatitis C":          ["fatigue", "yellowish_skin", "nausea", "loss_of_appetite", "abdominal_pain"],
    "Hepatitis D":          ["fatigue", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "joint_pain"],
    "Hepatitis E":          ["fatigue", "yellowish_skin", "dark_urine", "vomiting", "joint_pain"],
    "Alcoholic hepatitis":  ["vomiting", "yellowish_skin", "abdominal_pain", "swelling_of_stomach", "distention_of_abdomen", "history_of_alcohol_consumption"],
    "Tuberculosis":         ["cough", "high_fever", "fatigue", "weight_loss", "breathlessness", "sweating", "chest_pain", "loss_of_appetite", "phlegm"],
    "Common Cold":          ["continuous_sneezing", "chills", "fatigue", "cough", "headache", "runny_nose", "congestion", "phlegm", "mild_fever"],
    "Pneumonia":            ["cough", "fatigue", "high_fever", "breathlessness", "sweating", "chills", "nausea", "chest_pain"],
    "Heart attack":         ["vomiting", "breathlessness", "sweating", "chest_pain", "fast_heart_rate"],
    "Varicose veins":       ["fatigue", "cramps", "bruising", "swollen_legs", "swollen_blood_vessels", "prominent_veins_on_calf", "painful_walking"],
    "Hypothyroidism":       ["fatigue", "weight_gain", "cold_hands_and_feets", "mood_swings", "lethargy", "depression", "constipation"],
    "Hyperthyroidism":      ["fatigue", "mood_swings", "weight_loss", "restlessness", "sweating", "diarrhoea", "fast_heart_rate", "excessive_hunger"],
    "Hypoglycemia":         ["fatigue", "vomiting", "sweating", "headache", "nausea", "blurred_and_distorted_vision", "excessive_hunger"],
    "Osteoarthritis":       ["joint_pain", "knee_pain", "hip_joint_pain", "swelling_joints", "movement_stiffness"],
    "Arthritis":            ["muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "loss_of_balance"],
    "Urinary tract infection": ["burning_micturition", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine", "fatigue"],
    "Psoriasis":            ["skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails"],
    "Acne":                 ["skin_rash", "pus_filled_pimples", "blackheads", "scurring", "fatigue"],
    "Impetigo":             ["skin_rash", "blister", "red_sore_around_nose", "yellow_crust_ooze", "itching"],
}

ALL_SYMPTOMS = sorted(set(s for symptoms in DISEASES.values() for s in symptoms))


def build_dataset(samples_per_disease: int = 80) -> pd.DataFrame:
    """Generate a synthetic balanced dataset from the disease-symptom map."""
    rows, labels = [], []
    rng = np.random.default_rng(42)
    for disease, core_symptoms in DISEASES.items():
        for _ in range(samples_per_disease):
            row = {s: 0 for s in ALL_SYMPTOMS}
            # Always include core symptoms (occasionally drop 1-2 for realism)
            for sym in core_symptoms:
                if sym in row:
                    row[sym] = 1 if rng.random() > 0.15 else 0
            # Add a small amount of noise
            for sym in ALL_SYMPTOMS:
                if row[sym] == 0 and rng.random() < 0.05:
                    row[sym] = 1
            rows.append(row)
            labels.append(disease)
    df = pd.DataFrame(rows, columns=ALL_SYMPTOMS)
    df["prognosis"] = labels
    return df


def train_and_save():
    print("🔬 Building training dataset …")
    df = build_dataset(samples_per_disease=100)
    print(f"   Dataset shape: {df.shape}")

    X = df[ALL_SYMPTOMS]
    le = LabelEncoder()
    y = le.fit_transform(df["prognosis"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🌲 Training Random Forest …")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc*100:.1f}%")

    os.makedirs("model", exist_ok=True)
    with open("model/disease_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("model/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open("model/symptoms_list.pkl", "wb") as f:
        pickle.dump(ALL_SYMPTOMS, f)
    with open("model/disease_symptoms_map.pkl", "wb") as f:
        pickle.dump(DISEASES, f)

    print("💾 Model saved to model/")
    return acc


if __name__ == "__main__":
    train_and_save()
