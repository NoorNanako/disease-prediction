"""
Disease Prediction App  –  Main GUI
Run: python app.py
"""

import os
import sys
import pickle
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import numpy as np

# ─── Colours & fonts ─────────────────────────────────────────────────────────
BG          = "#0f1117"
CARD_BG     = "#1a1d2e"
ACCENT      = "#6c63ff"
ACCENT2     = "#00d4aa"
TEXT_MAIN   = "#ffffff"
TEXT_SUB    = "#8b8fa8"
DANGER      = "#ff4757"
SUCCESS     = "#2ed573"
WARNING     = "#ffa502"
BORDER      = "#2a2d3e"

DISEASE_INFO = {
    "Fungal infection":          ("🍄", "Moderate", "See a dermatologist. Keep affected area clean & dry."),
    "Allergy":                   ("🤧", "Mild",     "Antihistamines may help. Identify and avoid triggers."),
    "GERD":                      ("🔥", "Moderate", "Avoid spicy/fatty food. Consult a gastroenterologist."),
    "Chronic cholestasis":       ("⚠️",  "Serious",  "Needs liver function tests. See a hepatologist."),
    "Drug Reaction":             ("💊", "Serious",  "Stop suspected medication. See a doctor immediately."),
    "Peptic ulcer disease":      ("🩺", "Moderate", "Antacids & proton-pump inhibitors. Avoid NSAIDs."),
    "AIDS":                      ("🔴", "Serious",  "Antiretroviral therapy. Regular specialist follow-up."),
    "Diabetes":                  ("🩸", "Serious",  "Blood glucose monitoring. Diet, exercise & medication."),
    "Gastroenteritis":           ("🤢", "Moderate", "Stay hydrated. ORS. Rest."),
    "Bronchial Asthma":          ("💨", "Serious",  "Inhalers/bronchodilators. Avoid triggers & cold air."),
    "Hypertension":              ("❤️", "Serious",  "Low-sodium diet, exercise & antihypertensives."),
    "Migraine":                  ("🧠", "Moderate", "Dark quiet room, analgesics, triptans on prescription."),
    "Cervical spondylosis":      ("🦴", "Moderate", "Physiotherapy, neck exercises, pain relief."),
    "Paralysis":                 ("🚨", "Critical", "Emergency medical care needed immediately."),
    "Jaundice":                  ("🌕", "Serious",  "Liver function tests. Avoid alcohol. Rest."),
    "Malaria":                   ("🦟", "Serious",  "Antimalarial drugs. Bed rest. Hydration."),
    "Chicken pox":               ("🔴", "Mild",     "Calamine lotion, rest, avoid scratching."),
    "Dengue":                    ("🦟", "Serious",  "Hospitalisation if severe. Hydration. No aspirin."),
    "Typhoid":                   ("🦠", "Serious",  "Antibiotics. Liquid diet. Strict hygiene."),
    "Hepatitis A":               ("🟡", "Moderate", "Rest, fluids, avoid alcohol. Usually self-limiting."),
    "Hepatitis B":               ("🟠", "Serious",  "Antiviral therapy. Regular liver monitoring."),
    "Hepatitis C":               ("🟠", "Serious",  "Direct-acting antivirals. Specialist consultation."),
    "Hepatitis D":               ("🟠", "Serious",  "Interferon therapy. Avoid alcohol."),
    "Hepatitis E":               ("🟡", "Moderate", "Rest, fluids. Usually resolves in 4–6 weeks."),
    "Alcoholic hepatitis":       ("🍺", "Serious",  "Stop alcohol immediately. Seek medical help."),
    "Tuberculosis":              ("🫁", "Serious",  "6-month antibiotic course. Isolation initially."),
    "Common Cold":               ("🤒", "Mild",     "Rest, fluids, OTC decongestants."),
    "Pneumonia":                 ("🫁", "Serious",  "Antibiotics or antivirals. Rest. Fluids."),
    "Heart attack":              ("💔", "Critical", "Call emergency services NOW. Every minute counts."),
    "Varicose veins":            ("🦵", "Mild",     "Compression stockings, elevation, avoid prolonged standing."),
    "Hypothyroidism":            ("🦋", "Moderate", "Levothyroxine therapy. Regular thyroid monitoring."),
    "Hyperthyroidism":           ("🦋", "Moderate", "Anti-thyroid drugs or radioiodine. Specialist care."),
    "Hypoglycemia":              ("🍬", "Serious",  "Immediate glucose intake. Frequent small meals."),
    "Osteoarthritis":            ("🦴", "Moderate", "Pain relief, physiotherapy, weight management."),
    "Arthritis":                 ("🦴", "Moderate", "Anti-inflammatories, physiotherapy."),
    "Urinary tract infection":   ("🚽", "Moderate", "Antibiotics, increased fluid intake."),
    "Psoriasis":                 ("🧴", "Moderate", "Topical corticosteroids, phototherapy."),
    "Acne":                      ("🫧", "Mild",     "Topical retinoids, antibiotics, keep skin clean."),
    "Impetigo":                  ("🩹", "Mild",     "Topical or oral antibiotics. Keep area clean."),
}

SEVERITY_COLOR = {
    "Mild":     SUCCESS,
    "Moderate": WARNING,
    "Serious":  "#ff6b6b",
    "Critical": DANGER,
}


# ─── Load model ──────────────────────────────────────────────────────────────
def load_model():
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    if not os.path.exists(os.path.join(model_dir, "disease_model.pkl")):
        return None, None, None, None
    with open(os.path.join(model_dir, "disease_model.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(model_dir, "symptoms_list.pkl"), "rb") as f:
        symptoms = pickle.load(f)
    with open(os.path.join(model_dir, "disease_symptoms_map.pkl"), "rb") as f:
        disease_map = pickle.load(f)
    return clf, le, symptoms, disease_map


# ─── Main App ─────────────────────────────────────────────────────────────────
class DiseasePredictionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🩺 AI Disease Prediction System")
        self.geometry("1100x750")
        self.minsize(900, 650)
        self.configure(bg=BG)
        self.resizable(True, True)

        self.clf, self.le, self.all_symptoms, self.disease_map = load_model()
        self.selected_symptoms: set = set()
        self.symptom_vars: dict = {}

        self._build_ui()

        if self.clf is None:
            self._show_train_prompt()

    # ── UI Construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=ACCENT, height=70)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="🩺  AI Disease Prediction System",
                 font=("Segoe UI", 20, "bold"), bg=ACCENT, fg=TEXT_MAIN
                 ).pack(side="left", padx=24, pady=16)
        tk.Label(header, text="Powered by Machine Learning",
                 font=("Segoe UI", 10), bg=ACCENT, fg="#c9c6ff"
                 ).pack(side="right", padx=24)

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=16)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_right_panel(body)

    def _build_left_panel(self, parent):
        frame = tk.Frame(parent, bg=CARD_BG, bd=0, highlightthickness=1,
                         highlightbackground=BORDER)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # Search
        tk.Label(frame, text="🔍 Search Symptoms",
                 font=("Segoe UI", 13, "bold"), bg=CARD_BG, fg=TEXT_MAIN
                 ).pack(anchor="w", padx=16, pady=(14, 4))

        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._filter_symptoms)
        search_entry = tk.Entry(frame, textvariable=self.search_var,
                                font=("Segoe UI", 11), bg="#252838",
                                fg=TEXT_MAIN, insertbackground=TEXT_MAIN,
                                relief="flat", bd=8)
        search_entry.pack(fill="x", padx=16, pady=(0, 10))

        # Symptom list with scrollbar
        list_frame = tk.Frame(frame, bg=CARD_BG)
        list_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        style = ttk.Style()
        style.configure("Symptoms.TFrame", background=CARD_BG)

        self.symptom_canvas = tk.Canvas(list_frame, bg=CARD_BG,
                                        highlightthickness=0,
                                        yscrollcommand=scrollbar.set)
        self.symptom_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.symptom_canvas.yview)

        self.symptom_frame = tk.Frame(self.symptom_canvas, bg=CARD_BG)
        self.canvas_window = self.symptom_canvas.create_window(
            (0, 0), window=self.symptom_frame, anchor="nw"
        )
        self.symptom_frame.bind("<Configure>", self._on_frame_configure)
        self.symptom_canvas.bind("<Configure>", self._on_canvas_configure)
        self.symptom_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Selected count
        self.count_label = tk.Label(frame, text="0 symptoms selected",
                                    font=("Segoe UI", 9), bg=CARD_BG, fg=TEXT_SUB)
        self.count_label.pack(padx=16, pady=(0, 4))

        # Buttons
        btn_row = tk.Frame(frame, bg=CARD_BG)
        btn_row.pack(fill="x", padx=16, pady=(0, 14))

        self._btn(btn_row, "🔍 Predict", self._predict, ACCENT).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        self._btn(btn_row, "🔄 Reset", self._reset, "#3a3d52").pack(
            side="left", fill="x", expand=True)

        if self.all_symptoms:
            self._populate_symptoms(self.all_symptoms)

    def _build_right_panel(self, parent):
        frame = tk.Frame(parent, bg=CARD_BG, bd=0, highlightthickness=1,
                         highlightbackground=BORDER)
        frame.grid(row=0, column=1, sticky="nsew")

        tk.Label(frame, text="📊 Prediction Results",
                 font=("Segoe UI", 13, "bold"), bg=CARD_BG, fg=TEXT_MAIN
                 ).pack(anchor="w", padx=16, pady=(14, 0))

        # Result card (hidden initially)
        self.result_frame = tk.Frame(frame, bg=CARD_BG)
        self.result_frame.pack(fill="both", expand=True, padx=16, pady=10)

        self._show_placeholder()

    def _show_placeholder(self):
        for w in self.result_frame.winfo_children():
            w.destroy()
        tk.Label(self.result_frame,
                 text="Select symptoms on the left\nand click Predict",
                 font=("Segoe UI", 14), bg=CARD_BG, fg=TEXT_SUB,
                 justify="center"
                 ).place(relx=0.5, rely=0.45, anchor="center")

    # ── Symptom Widgets ──────────────────────────────────────────────────────
    def _populate_symptoms(self, symptoms):
        for w in self.symptom_frame.winfo_children():
            w.destroy()
        self.symptom_vars.clear()
        for sym in sorted(symptoms):
            var = tk.BooleanVar(value=(sym in self.selected_symptoms))
            self.symptom_vars[sym] = var
            label = sym.replace("_", " ").title()
            cb = tk.Checkbutton(
                self.symptom_frame, text=f"  {label}",
                variable=var, bg=CARD_BG, fg=TEXT_MAIN, selectcolor="#252838",
                activebackground=CARD_BG, activeforeground=ACCENT,
                font=("Segoe UI", 10), anchor="w",
                command=lambda s=sym, v=var: self._toggle_symptom(s, v)
            )
            cb.pack(fill="x", pady=1)

    def _toggle_symptom(self, sym, var):
        if var.get():
            self.selected_symptoms.add(sym)
        else:
            self.selected_symptoms.discard(sym)
        self.count_label.config(text=f"{len(self.selected_symptoms)} symptom(s) selected")

    def _filter_symptoms(self, *_):
        q = self.search_var.get().lower()
        filtered = [s for s in self.all_symptoms if q in s] if self.all_symptoms else []
        self._populate_symptoms(filtered if q else self.all_symptoms)

    # ── Canvas helpers ───────────────────────────────────────────────────────
    def _on_frame_configure(self, _):
        self.symptom_canvas.configure(
            scrollregion=self.symptom_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.symptom_canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.symptom_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ── Prediction ───────────────────────────────────────────────────────────
    def _predict(self):
        if self.clf is None:
            messagebox.showerror("Model not found",
                                 "Please train the model first (train_model.py).")
            return
        if len(self.selected_symptoms) < 2:
            messagebox.showwarning("Too few symptoms",
                                   "Please select at least 2 symptoms.")
            return

        X = np.zeros(len(self.all_symptoms))
        for i, sym in enumerate(self.all_symptoms):
            if sym in self.selected_symptoms:
                X[i] = 1

        proba = self.clf.predict_proba([X])[0]
        top_idx = np.argsort(proba)[::-1][:5]

        self._show_results(top_idx, proba)

    def _show_results(self, top_idx, proba):
        for w in self.result_frame.winfo_children():
            w.destroy()

        top_disease = self.le.classes_[top_idx[0]]
        icon, severity, advice = DISEASE_INFO.get(
            top_disease, ("🩺", "Unknown", "Consult a doctor."))
        sev_color = SEVERITY_COLOR.get(severity, TEXT_SUB)

        # Primary result card
        card = tk.Frame(self.result_frame, bg="#252838",
                        highlightthickness=2, highlightbackground=ACCENT)
        card.pack(fill="x", pady=(0, 12))

        top_row = tk.Frame(card, bg="#252838")
        top_row.pack(fill="x", padx=16, pady=(14, 4))

        tk.Label(top_row, text=icon, font=("Segoe UI", 36),
                 bg="#252838", fg=TEXT_MAIN).pack(side="left")

        info = tk.Frame(top_row, bg="#252838")
        info.pack(side="left", padx=12)
        tk.Label(info, text=top_disease, font=("Segoe UI", 16, "bold"),
                 bg="#252838", fg=TEXT_MAIN).pack(anchor="w")
        tk.Label(info, text=f"Severity: {severity}",
                 font=("Segoe UI", 11), bg="#252838", fg=sev_color
                 ).pack(anchor="w")

        pct = proba[top_idx[0]] * 100
        tk.Label(card, text=f"Confidence: {pct:.1f}%",
                 font=("Segoe UI", 11), bg="#252838", fg=ACCENT2
                 ).pack(anchor="w", padx=16, pady=(0, 4))

        # Progress bar
        pb_frame = tk.Frame(card, bg="#3a3d52", height=8)
        pb_frame.pack(fill="x", padx=16, pady=(0, 8))
        pb_frame.pack_propagate(False)
        filled = tk.Frame(pb_frame, bg=ACCENT, height=8,
                          width=int(pct * 3.5))
        filled.place(x=0, y=0, relheight=1)

        # Advice
        tk.Label(card, text=f"💡 {advice}", font=("Segoe UI", 10),
                 bg="#252838", fg=TEXT_SUB, wraplength=380, justify="left"
                 ).pack(anchor="w", padx=16, pady=(0, 14))

        # Other possibilities
        tk.Label(self.result_frame, text="Other Possibilities",
                 font=("Segoe UI", 11, "bold"), bg=CARD_BG, fg=TEXT_SUB
                 ).pack(anchor="w", pady=(0, 6))

        for idx in top_idx[1:]:
            d = self.le.classes_[idx]
            p = proba[idx] * 100
            if p < 1:
                continue
            ico2, sev2, _ = DISEASE_INFO.get(d, ("🩺", "Unknown", ""))
            row = tk.Frame(self.result_frame, bg="#1e2133")
            row.pack(fill="x", pady=2, ipady=6)
            tk.Label(row, text=ico2, font=("Segoe UI", 14),
                     bg="#1e2133", fg=TEXT_MAIN).pack(side="left", padx=10)
            tk.Label(row, text=d, font=("Segoe UI", 10, "bold"),
                     bg="#1e2133", fg=TEXT_MAIN).pack(side="left")
            tk.Label(row, text=f"{p:.1f}%", font=("Segoe UI", 10),
                     bg="#1e2133", fg=ACCENT2).pack(side="right", padx=10)

        # Disclaimer
        tk.Label(self.result_frame,
                 text="⚠️  This is an AI prediction tool. Always consult a qualified doctor.",
                 font=("Segoe UI", 8), bg=CARD_BG, fg=TEXT_SUB, wraplength=450
                 ).pack(pady=(10, 0))

    # ── Reset ────────────────────────────────────────────────────────────────
    def _reset(self):
        self.selected_symptoms.clear()
        self.search_var.set("")
        for var in self.symptom_vars.values():
            var.set(False)
        self.count_label.config(text="0 symptoms selected")
        self._show_placeholder()

    # ── Train prompt ─────────────────────────────────────────────────────────
    def _show_train_prompt(self):
        popup = tk.Toplevel(self)
        popup.title("First Run")
        popup.geometry("400x220")
        popup.configure(bg=CARD_BG)
        popup.grab_set()

        tk.Label(popup, text="🤖 Model Not Found",
                 font=("Segoe UI", 14, "bold"), bg=CARD_BG, fg=TEXT_MAIN
                 ).pack(pady=(20, 6))
        tk.Label(popup,
                 text="The ML model hasn't been trained yet.\nClick below to train it now (takes ~10 seconds).",
                 font=("Segoe UI", 10), bg=CARD_BG, fg=TEXT_SUB
                 ).pack(pady=6)

        def do_train():
            btn.config(state="disabled", text="Training …")
            popup.update()
            try:
                subprocess.run([sys.executable, "train_model.py"], check=True,
                               cwd=os.path.dirname(__file__))
                self.clf, self.le, self.all_symptoms, self.disease_map = load_model()
                self._populate_symptoms(self.all_symptoms)
                popup.destroy()
                messagebox.showinfo("Done", "Model trained! Start predicting.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                popup.destroy()

        btn = self._btn(popup, "🚀 Train Model Now", do_train, ACCENT)
        btn.pack(pady=14, ipadx=16, ipady=6)

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _btn(parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd,
                         bg=color, fg=TEXT_MAIN, activebackground=color,
                         activeforeground=TEXT_MAIN, relief="flat", bd=0,
                         font=("Segoe UI", 10, "bold"), cursor="hand2")


if __name__ == "__main__":
    app = DiseasePredictionApp()
    app.mainloop()
