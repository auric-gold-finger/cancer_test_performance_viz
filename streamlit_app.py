import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# CSS for clean look
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        color: #2c3e50;
        text-align: center;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# App config
st.set_page_config(page_title="Cancer Screening Outcomes", page_icon="📊", layout="wide")

st.markdown("<h1 class='main-header'>Cancer Screening Outcomes Visualizer</h1>", unsafe_allow_html=True)
st.markdown("Enter your details to see a personalized Sankey diagram of screening test outcomes for 100 people like you. Data as of July 2025. For education only—consult a doctor.")

# Data structures
CANCER_INCIDENCE = {
    "lung": {"male": {40: 10, 50: 30, 60: 95, 70: 195, 80: 235}, "female": {40: 15, 50: 38, 60: 80, 70: 145, 80: 175}},
    "breast": {"male": {40: 1, 50: 2, 60: 3, 70: 4, 80: 5}, "female": {40: 48, 50: 130, 60: 200, 70: 250, 80: 275}},
    "colorectal": {"male": {40: 13, 50: 32, 60: 75, 70: 145, 80: 205}, "female": {40: 10, 50: 24, 60: 55, 70: 105, 80: 155}},
    "prostate": {"male": {40: 3, 50: 28, 60: 125, 70: 310, 80: 460}, "female": {40: 0, 50: 0, 60: 0, 70: 0, 80: 0}},
    "liver": {"male": {40: 3, 50: 9, 60: 20, 70: 30, 80: 38}, "female": {40: 1, 50: 4, 60: 8, 70: 13, 80: 16}},
    "pancreatic": {"male": {40: 3, 50: 8, 60: 17, 70: 30, 80: 40}, "female": {40: 2, 50: 7, 60: 14, 70: 25, 80: 33}},
    "ovarian": {"male": {40: 0, 50: 0, 60: 0, 70: 0, 80: 0}, "female": {40: 6, 50: 13, 60: 19, 70: 23, 80: 25}},
    "kidney": {"male": {40: 7, 50: 16, 60: 30, 70: 45, 80: 52}, "female": {40: 4, 50: 9, 60: 17, 70: 25, 80: 32}},
    "bladder": {"male": {40: 4, 50: 9, 60: 24, 70: 58, 80: 88}, "female": {40: 1, 50: 3, 60: 7, 70: 16, 80: 26}},
    "brain": {"male": {40: 5, 50: 7, 60: 9, 70: 13, 80: 16}, "female": {40: 4, 50: 5, 60: 7, 70: 9, 80: 11}},
    "cervical": {"male": {40: 0, 50: 0, 60: 0, 70: 0, 80: 0}, "female": {40: 9, 50: 8, 60: 7, 70: 6, 80: 5}},
    "endometrial": {"male": {40: 0, 50: 0, 60: 0, 70: 0, 80: 0}, "female": {40: 9, 50: 27, 60: 52, 70: 68, 80: 72}},
    "esophageal": {"male": {40: 1, 50: 4, 60: 9, 70: 16, 80: 22}, "female": {40: 0.4, 50: 1, 60: 3, 70: 5, 80: 7}},
    "gastric": {"male": {40: 3, 50: 5, 60: 9, 70: 16, 80: 23}, "female": {40: 2, 50: 3, 60: 5, 70: 9, 80: 13}},
    "head_neck": {"male": {40: 5, 50: 9, 60: 16, 70: 24, 80: 27}, "female": {40: 2, 50: 3, 60: 5, 70: 7, 80: 9}},
    "hodgkin_lymphoma": {"male": {40: 2, 50: 2, 60: 3, 70: 3, 80: 4}, "female": {40: 2, 50: 2, 60: 2, 70: 3, 80: 3}},
    "non_hodgkin_lymphoma": {"male": {40: 5, 50: 9, 60: 16, 70: 30, 80: 42}, "female": {40: 4, 50: 7, 60: 13, 70: 23, 80: 32}},
    "leukemia": {"male": {40: 4, 50: 6, 60: 11, 70: 19, 80: 30}, "female": {40: 3, 50: 4, 60: 7, 70: 13, 80: 19}},
    "melanoma": {"male": {40: 9, 50: 16, 60: 27, 70: 37, 80: 42}, "female": {40: 7, 50: 11, 60: 16, 70: 21, 80: 23}},
    "myeloma": {"male": {40: 1, 50: 3, 60: 6, 70: 11, 80: 16}, "female": {40: 1, 50: 2, 60: 4, 70: 8, 80: 11}},
    "sarcoma": {"male": {40: 1, 50: 2, 60: 2, 70: 3, 80: 4}, "female": {40: 1, 50: 1, 60: 2, 70: 2, 80: 3}},
    "testicular": {"male": {40: 3, 50: 2, 60: 1, 70: 0.5, 80: 0.3}, "female": {40: 0, 50: 0, 60: 0, 70: 0, 80: 0}},
    "thyroid": {"male": {40: 6, 50: 7, 60: 8, 70: 9, 80: 10}, "female": {40: 16, 50: 19, 60: 21, 70: 23, 80: 25}},
    "uterine": {"male": {40: 0, 50: 0, 60: 0, 70: 0, 80: 0}, "female": {40: 9, 50: 27, 60: 52, 70: 68, 80: 72}}
}

TEST_PERFORMANCE = {
    "Whole-body MRI": {
        "lung": {"sensitivity": 0.92, "specificity": 0.94},
        "breast": {"sensitivity": 0.95, "specificity": 0.75},
        "colorectal": {"sensitivity": 0.67, "specificity": 0.95},
        "prostate": {"sensitivity": 0.85, "specificity": 0.90},
        "liver": {"sensitivity": 0.84, "specificity": 0.94},
        "pancreatic": {"sensitivity": 0.75, "specificity": 0.85},
        "ovarian": {"sensitivity": 0.98, "specificity": 0.90},
        "kidney": {"sensitivity": 0.85, "specificity": 0.90},
        "bladder": {"sensitivity": 0.78, "specificity": 0.92},
        "brain": {"sensitivity": 0.92, "specificity": 0.95},
        "cervical": {"sensitivity": 0.88, "specificity": 0.89},
        "endometrial": {"sensitivity": 0.91, "specificity": 0.87},
        "esophageal": {"sensitivity": 0.82, "specificity": 0.88},
        "gastric": {"sensitivity": 0.79, "specificity": 0.85},
        "head_neck": {"sensitivity": 0.86, "specificity": 0.91},
        "hodgkin_lymphoma": {"sensitivity": 0.94, "specificity": 0.96},
        "non_hodgkin_lymphoma": {"sensitivity": 0.89, "specificity": 0.93},
        "leukemia": {"sensitivity": 0.72, "specificity": 0.88},
        "melanoma": {"sensitivity": 0.83, "specificity": 0.91},
        "myeloma": {"sensitivity": 0.85, "specificity": 0.92},
        "sarcoma": {"sensitivity": 0.87, "specificity": 0.89},
        "testicular": {"sensitivity": 0.91, "specificity": 0.95},
        "thyroid": {"sensitivity": 0.86, "specificity": 0.84},
        "uterine": {"sensitivity": 0.89, "specificity": 0.86}
    },
    "Galleri Blood Test": {
        "lung": {"sensitivity": 0.59, "specificity": 0.995},
        "breast": {"sensitivity": 0.25, "specificity": 0.995},
        "colorectal": {"sensitivity": 0.74, "specificity": 0.995},
        "prostate": {"sensitivity": 0.23, "specificity": 0.995},
        "liver": {"sensitivity": 0.93, "specificity": 0.995},
        "pancreatic": {"sensitivity": 0.83, "specificity": 0.995},
        "ovarian": {"sensitivity": 0.83, "specificity": 0.995},
        "kidney": {"sensitivity": 0.58, "specificity": 0.995},
        "bladder": {"sensitivity": 0.43, "specificity": 0.995},
        "brain": {"sensitivity": 0.95, "specificity": 0.995},
        "cervical": {"sensitivity": 0.65, "specificity": 0.995},
        "endometrial": {"sensitivity": 0.68, "specificity": 0.995},
        "esophageal": {"sensitivity": 0.80, "specificity": 0.995},
        "gastric": {"sensitivity": 0.85, "specificity": 0.995},
        "head_neck": {"sensitivity": 0.81, "specificity": 0.995},
        "hodgkin_lymphoma": {"sensitivity": 0.92, "specificity": 0.995},
        "non_hodgkin_lymphoma": {"sensitivity": 0.77, "specificity": 0.995},
        "leukemia": {"sensitivity": 0.89, "specificity": 0.995},
        "melanoma": {"sensitivity": 0.71, "specificity": 0.995},
        "myeloma": {"sensitivity": 0.85, "specificity": 0.995},
        "sarcoma": {"sensitivity": 0.84, "specificity": 0.995},
        "testicular": {"sensitivity": 0.88, "specificity": 0.995},
        "thyroid": {"sensitivity": 0.32, "specificity": 0.995},
        "uterine": {"sensitivity": 0.78, "specificity": 0.995}
    },
    "Low-dose CT Scan": {
        "lung": {"sensitivity": 0.97, "specificity": 0.952},
        "breast": {"sensitivity": 0.70, "specificity": 0.85},
        "colorectal": {"sensitivity": 0.96, "specificity": 0.80},
        "prostate": {"sensitivity": 0.75, "specificity": 0.80},
        "liver": {"sensitivity": 0.68, "specificity": 0.93},
        "pancreatic": {"sensitivity": 0.84, "specificity": 0.67},
        "ovarian": {"sensitivity": 0.83, "specificity": 0.87},
        "kidney": {"sensitivity": 0.85, "specificity": 0.89}
    },
    "Colonoscopy": {
        "colorectal": {"sensitivity": 0.95, "specificity": 0.90}
    },
    "Upper Endoscopy": {
        "esophageal": {"sensitivity": 0.85, "specificity": 0.90},
        "gastric": {"sensitivity": 0.80, "specificity": 0.85}
    },
    "Dermoscopy": {
        "melanoma": {"sensitivity": 0.94, "specificity": 0.85}
    },
    "Mammography": {
        "breast": {"sensitivity": 0.85, "specificity": 0.90}
    },
    "HPV Test": {
        "cervical": {"sensitivity": 0.95, "specificity": 0.94}
    },
    "PSA Test": {
        "prostate": {"sensitivity": 0.90, "specificity": 0.30}
    },
    "Skin Exam": {
        "melanoma": {"sensitivity": 0.77, "specificity": 0.89}
    }
}

# Sidebar inputs
with st.sidebar:
    st.header("Your Details")
    age = st.slider("Age", 30, 90, 55)
    sex = st.selectbox("Sex", ["male", "female"])
    smoking_status = st.selectbox("Smoking Status", ["Never smoked", "Former smoker", "Current smoker"])
    pack_years = st.slider("Pack-years", 0, 80, 20) if smoking_status != "Never smoked" else 0
    family_history = st.multiselect("Family Cancer History", ["Breast cancer", "Colorectal cancer", "Prostate cancer", "Ovarian cancer", "Lung cancer", "Pancreatic cancer"])
    genetic_mutations = st.multiselect("Genetic Mutations", ["BRCA1", "BRCA2", "Lynch syndrome", "TP53 (Li-Fraumeni)"])
    personal_history = st.checkbox("Personal Cancer History")
    tests = st.multiselect("Screening Tests", list(TEST_PERFORMANCE.keys()), default=["Whole-body MRI"])
    if len(tests) > 1:
        mode = st.selectbox("Combination Mode", ["Parallel (Any positive)", "Sequential (All positive)"])
    else:
        mode = None

# Calculations
cancer_types = set()
for test in tests:
    cancer_types.update(TEST_PERFORMANCE[test].keys())
cancer_types = list(cancer_types)

risk_multipliers = {}
for cancer_type in cancer_types:
    if (cancer_type in ["prostate", "testicular"] and sex == "female") or (cancer_type in ["ovarian", "cervical", "endometrial", "uterine"] and sex == "male"):
        continue
    risk_multipliers[cancer_type] = get_risk_multiplier(cancer_type, smoking_status, pack_years, family_history, genetic_mutations, personal_history)

overall_prevalence = calculate_overall_prevalence(age, sex, risk_multipliers)

if len(tests) > 1:
    sens, spec = combine_tests(tests, mode)
    fp_rate = np.mean([DOWNSTREAM_RISKS[test]["false_positive_rate"] for test in tests]) / 100
    biopsy_rate = np.mean([DOWNSTREAM_RISKS[test]["biopsy_rate_fp"] for test in tests])
    comp_rate = np.mean([DOWNSTREAM_RISKS[test]["comp_rate_biopsy"] for test in tests])
else:
    test = tests[0]
    sens = np.mean([TEST_PERFORMANCE[test][ct]["sensitivity"] for ct in TEST_PERFORMANCE[test]])
    spec = np.mean([TEST_PERFORMANCE[test][ct]["specificity"] for ct in TEST_PERFORMANCE[test]])
    fp_rate = DOWNSTREAM_RISKS[test]["false_positive_rate"] / 100
    biopsy_rate = DOWNSTREAM_RISKS[test]["biopsy_rate_fp"]
    comp_rate = DOWNSTREAM_RISKS[test]["comp_rate_biopsy"]

# Outcomes per 100 people
population = 100
has_cancer = overall_prevalence * population
no_cancer = population - has_cancer

tp = sens * has_cancer
fn = (1 - sens) * has_cancer
fp = (1 - spec) * no_cancer
tn = spec * no_cancer

positive = tp + fp
negative = fn + tn

biopsy = positive * biopsy_rate
no_biopsy = positive - biopsy
complication = biopsy * comp_rate
no_complication = biopsy - complication

cancer_treated = tp * 0.8  # Assume 80% of positives lead to treatment
benign = tp * 0.2 + fp  # Rest benign
further_monitor = fn  # FN to monitoring
reassured = tn

# Sankey Diagram
fig = go.Figure(data=[go.Sankey(
    orientation = 'h',
    node = dict(
        pad = 20,
        thickness = 30,
        line = dict(color = "gray", width = 0.5),
        label = [
            "100 People", 
            "Test Positive", "Test Negative", 
            "Follow-up Biopsy", "No Biopsy Needed", "Reassured", "Further Monitoring", 
            "Cancer Found & Treated", "Benign (False Alarm)", "Complication from Biopsy"
        ],
        color = [
            "#AED6F1", "#F39C12", "#BDC3C7",
            "#8E44AD", "#95A5A6", "#2ECC71", "#3498DB",
            "#1ABC9C", "#E67E22", "#E74C3C"
        ],
        x = [0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 0.8],
        y = [0.5, 0.3, 0.7, 0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5],
    ),
    link = dict(
        source = [
            0, 0, 
            1, 1, 
            2, 2, 
            3, 3, 3, 
            4
        ],
        target = [
            1, 2,
            3, 4,
            5, 6,
            7, 8, 9,
            8
        ],
        value = [
            positive, negative,
            biopsy, no_biopsy,
            reassured, further_monitor,
            cancer_treated, benign, complication,
            no_biopsy
        ],
        color = 'rgba(189, 195, 199, 0.5)',
        hovercolor = 'blue',
        label = [f"{v:.1f}" for v in [positive, negative, biopsy, no_biopsy, reassured, further_monitor, cancer_treated, benign, complication, no_biopsy]]  # Show numbers on links
    ),
    textfont = dict(size=14, color="black")
)])

fig.update_layout(
    title_text="Patient Pathways in Screening for 100 People",
    font_size=14,
    height=600,
    width=1200
)
st.plotly_chart(fig, use_container_width=True)

# Metrics
st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
st.write(f"Overall Risk: {overall_prevalence*100:.2f}%")
st.write(f"Positive Results: {positive:.1f}")
st.write(f"Negative Results: {negative:.1f}")
st.write(f"Biopsies: {biopsy:.1f}")
st.write(f"Complications: {complication:.1f}")
st.write(f"Cancer Found & Treated: {cancer_treated:.1f}")
st.write(f"Benign Results (False Alarm): {benign:.1f}")
st.markdown("</div>", unsafe_allow_html=True)