import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Updated Custom CSS for a more modern, personalized look
st.markdown("""
<style>
    .stApp {
        background-color: #f0f4f8;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stMetric {
        background-color: white;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stExpander, .stTabs > div {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        background-color: white;
        margin-bottom: 16px;
    }
    h1, h2, h3 {
        color: #1e40af;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    .personalized-header {
        color: #1e40af;
        font-size: 24px;
        font-weight: bold;
    }
    .info-box {
        background-color: #e0f2fe;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Configure the app
st.set_page_config(
    page_title="Personalized Cancer Risk Explorer",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Personalized Cancer Risk Explorer")
st.markdown("""
This interactive tool helps you understand your potential cancer risks and how screening tests might affect them. 
It's tailored to your profile for a more personal experience.
**Important:** This is for educational purposes only. Always consult a healthcare professional for medical advice. Data updated as of July 2025.
""")

# Updated Data Structures with latest estimates from 2025 sources (ACS, SEER projections, studies)
CANCER_INCIDENCE = {
    "lung": {
        "male": {40: 10, 50: 28, 60: 90, 70: 190, 80: 230},
        "female": {40: 14, 50: 35, 60: 75, 70: 140, 80: 170}
    },
    "breast": {
        "male": {40: 1, 50: 2, 60: 3, 70: 4, 80: 5},
        "female": {40: 45, 50: 125, 60: 195, 70: 245, 80: 270}
    },
    "colorectal": {
        "male": {40: 12, 50: 30, 60: 70, 70: 140, 80: 200},
        "female": {40: 9, 50: 22, 60: 50, 70: 100, 80: 150}
    },
    "prostate": {
        "male": {40: 3, 50: 25, 60: 120, 70: 300, 80: 450},
        "female": {40: 0, 50: 0, 60: 0, 70: 0, 80: 0}
    },
    "liver": {
        "male": {40: 3, 50: 8, 60: 18, 70: 28, 80: 35},
        "female": {40: 1, 50: 3, 60: 7, 70: 12, 80: 15}
    },
    "pancreatic": {
        "male": {40: 3, 50: 7, 60: 16, 70: 28, 80: 38},
        "female": {40: 2, 50: 6, 60: 13, 70: 24, 80: 32}
    },
    "ovarian": {
        "male": {40: 0, 50: 0, 60: 0, 70: 0, 80: 0},
        "female": {40: 5, 50: 12, 60: 18, 70: 22, 80: 24}
    },
    "kidney": {
        "male": {40: 6, 50: 15, 60: 28, 70: 42, 80: 50},
        "female": {40: 3, 50: 8, 60: 16, 70: 24, 80: 30}
    },
    "bladder": {
        "male": {40: 3, 50: 8, 60: 22, 70: 55, 80: 85},
        "female": {40: 1, 50: 2, 60: 6, 70: 15, 80: 25}
    },
    "brain": {
        "male": {40: 4, 50: 6, 60: 8, 70: 12, 80: 15},
        "female": {40: 3, 50: 4, 60: 6, 70: 8, 80: 10}
    },
    "cervical": {
        "male": {40: 0, 50: 0, 60: 0, 70: 0, 80: 0},
        "female": {40: 8, 50: 7, 60: 6, 70: 5, 80: 4}
    },
    "endometrial": {
        "male": {40: 0, 50: 0, 60: 0, 70: 0, 80: 0},
        "female": {40: 8, 50: 25, 60: 50, 70: 65, 80: 70}
    },
    "esophageal": {
        "male": {40: 1, 50: 3, 60: 8, 70: 15, 80: 20},
        "female": {40: 0.3, 50: 0.8, 60: 2, 70: 4, 80: 6}
    },
    "gastric": {
        "male": {40: 2, 50: 4, 60: 8, 70: 15, 80: 22},
        "female": {40: 1, 50: 2, 60: 4, 70: 8, 80: 12}
    },
    "head_neck": {
        "male": {40: 4, 50: 8, 60: 15, 70: 22, 80: 25},
        "female": {40: 1, 50: 2, 60: 4, 70: 6, 80: 8}
    },
    "hodgkin_lymphoma": {
        "male": {40: 2, 50: 2, 60: 2, 70: 3, 80: 4},
        "female": {40: 2, 50: 2, 60: 2, 70: 2, 80: 3}
    },
    "non_hodgkin_lymphoma": {
        "male": {40: 4, 50: 8, 60: 15, 70: 28, 80: 40},
        "female": {40: 3, 50: 6, 60: 12, 70: 22, 80: 30}
    },
    "leukemia": {
        "male": {40: 3, 50: 5, 60: 10, 70: 18, 80: 28},
        "female": {40: 2, 50: 3, 60: 6, 70: 12, 80: 18}
    },
    "melanoma": {
        "male": {40: 8, 50: 15, 60: 25, 70: 35, 80: 40},
        "female": {40: 6, 50: 10, 60: 15, 70: 20, 80: 22}
    },
    "myeloma": {
        "male": {40: 1, 50: 2, 60: 5, 70: 10, 80: 15},
        "female": {40: 0.5, 50: 1, 60: 3, 70: 7, 80: 10}
    },
    "sarcoma": {
        "male": {40: 1, 50: 1, 60: 2, 70: 3, 80: 4},
        "female": {40: 1, 50: 1, 60: 1, 70: 2, 80: 3}
    },
    "testicular": {
        "male": {40: 3, 50: 2, 60: 1, 70: 0.5, 80: 0.3},
        "female": {40: 0, 50: 0, 60: 0, 70: 0, 80: 0}
    },
    "thyroid": {
        "male": {40: 5, 50: 6, 60: 7, 70: 8, 80: 9},
        "female": {40: 15, 50: 18, 60: 20, 70: 22, 80: 24}
    },
    "uterine": {
        "male": {40: 0, 50: 0, 60: 0, 70: 0, 80: 0},
        "female": {40: 8, 50: 25, 60: 50, 70: 65, 80: 70}
    }
}

# Updated TEST_PERFORMANCE based on latest studies (2025)
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
    "CT Scan": {
        "lung": {"sensitivity": 0.97, "specificity": 0.952},
        "breast": {"sensitivity": 0.70, "specificity": 0.85},
        "colorectal": {"sensitivity": 0.96, "specificity": 0.80},
        "prostate": {"sensitivity": 0.75, "specificity": 0.80},
        "liver": {"sensitivity": 0.68, "specificity": 0.93},
        "pancreatic": {"sensitivity": 0.84, "specificity": 0.67},
        "ovarian": {"sensitivity": 0.83, "specificity": 0.87},
        "kidney": {"sensitivity": 0.85, "specificity": 0.89}
    }
}

DOWNSTREAM_RISKS = {
    "Whole-body MRI": {
        "false_positive_rate": 16.0,
        "biopsy_rate_fp": 0.5,
        "comp_rate_biopsy": 0.03,
        "typical_followup": "Additional MRI with contrast, possible biopsy",
        "followup_complications": 2.1,
        "psychological_impact": "Moderate - incidental findings cause anxiety",
        "radiation_exposure": "None from MRI, possible CT follow-up"
    },
    "Galleri Blood Test": {
        "false_positive_rate": 0.5,
        "biopsy_rate_fp": 0.5,
        "comp_rate_biopsy": 0.03,
        "typical_followup": "Imaging scans (CT, MRI, PET), possible biopsy",
        "followup_complications": 3.8,
        "psychological_impact": "High - positive blood test causes significant anxiety",
        "radiation_exposure": "Moderate to high from follow-up CT/PET scans"
    },
    "CT Scan": {
        "false_positive_rate": 23.0,
        "biopsy_rate_fp": 0.5,
        "comp_rate_biopsy": 0.03,
        "typical_followup": "Repeat CT, additional imaging, possible biopsy",
        "followup_complications": 4.2,
        "psychological_impact": "Moderate to high - abnormal findings cause worry",
        "radiation_exposure": "Additional radiation from repeat scans"
    }
}

# Functions
def calculate_ppv_npv(sensitivity, specificity, prevalence):
    if prevalence == 0:
        return 0, 1
    ppv = (sensitivity * prevalence) / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence)) if (sensitivity * prevalence + (1 - specificity) * (1 - prevalence)) > 0 else 0
    npv = (specificity * (1 - prevalence)) / ((1 - sensitivity) * prevalence + specificity * (1 - prevalence)) if ((1 - sensitivity) * prevalence + specificity * (1 - prevalence)) > 0 else 1
    return ppv, npv

def calculate_post_test_risk_negative(sensitivity, specificity, prevalence):
    if prevalence == 0:
        return 0
    return ((1 - sensitivity) * prevalence) / ((1 - sensitivity) * prevalence + specificity * (1 - prevalence)) if ((1 - sensitivity) * prevalence + specificity * (1 - prevalence)) > 0 else 0

def get_prevalence_from_incidence(incidence_rate):
    return (incidence_rate / 100000) * 5  # Assuming 5-year prevalence approximation

def interpolate_incidence(age, sex, cancer_type):
    if cancer_type not in CANCER_INCIDENCE:
        return 0
    age_points = list(CANCER_INCIDENCE[cancer_type][sex].keys())
    incidence_points = list(CANCER_INCIDENCE[cancer_type][sex].values())
    
    if age <= min(age_points):
        return incidence_points[0]
    elif age >= max(age_points):
        return incidence_points[-1]
    else:
        return np.interp(age, age_points, incidence_points)

def get_risk_multiplier(cancer_type, smoking_status, pack_years, family_history, genetic_mutations, personal_history):
    multiplier = 1.0
    if smoking_status == "Current smoker":
        if cancer_type == "lung":
            if pack_years < 20:
                multiplier *= 15
            elif pack_years < 40:
                multiplier *= 25
            else:
                multiplier *= 35
        elif cancer_type in ["bladder", "kidney", "pancreatic", "cervical", "esophageal", "gastric", "head_neck"]:
            multiplier *= 2.5
        elif cancer_type in ["colorectal", "liver"]:
            multiplier *= 1.8
    elif smoking_status == "Former smoker":
        if cancer_type == "lung":
            if pack_years < 20:
                multiplier *= 8
            elif pack_years < 40:
                multiplier *= 12
            else:
                multiplier *= 18
        elif cancer_type in ["bladder", "kidney", "pancreatic", "cervical", "esophageal", "gastric", "head_neck"]:
            multiplier *= 1.8
        elif cancer_type in ["colorectal", "liver"]:
            multiplier *= 1.4
    cancer_family_map = {
        "breast": "Breast cancer",
        "colorectal": "Colorectal cancer", 
        "prostate": "Prostate cancer",
        "ovarian": "Ovarian cancer",
        "lung": "Lung cancer",
        "pancreatic": "Pancreatic cancer"
    }
    if cancer_family_map.get(cancer_type) in family_history:
        if cancer_type == "breast":
            multiplier *= 2.3
        elif cancer_type == "colorectal":
            multiplier *= 2.2
        elif cancer_type == "prostate":
            multiplier *= 2.5
        elif cancer_type == "ovarian":
            multiplier *= 3.1
        elif cancer_type in ["lung", "pancreatic"]:
            multiplier *= 1.8
    if "BRCA1" in genetic_mutations:
        if cancer_type == "breast":
            multiplier *= 35
        elif cancer_type == "ovarian":
            multiplier *= 20
    if "BRCA2" in genetic_mutations:
        if cancer_type == "breast":
            multiplier *= 20
        elif cancer_type == "ovarian":
            multiplier *= 8
        elif cancer_type == "prostate":
            multiplier *= 4.5
    if "Lynch syndrome" in genetic_mutations:
        if cancer_type == "colorectal":
            multiplier *= 15
        elif cancer_type == "ovarian":
            multiplier *= 6
        elif cancer_type == "endometrial":
            multiplier *= 12
    if "TP53 (Li-Fraumeni)" in genetic_mutations:
        if cancer_type in ["breast", "lung", "colorectal", "liver", "brain", "sarcoma"]:
            multiplier *= 10
    if personal_history:
        multiplier *= 2.5
    return min(multiplier, 100)

def calculate_overall_cancer_prevalence(age, sex, risk_multipliers=None):
    total_prevalence = 0
    for cancer_type in CANCER_INCIDENCE.keys():
        if (cancer_type in ["prostate", "testicular"] and sex == "female") or (cancer_type in ["ovarian", "cervical", "endometrial", "uterine"] and sex == "male"):
            continue
        incidence_rate = interpolate_incidence(age, sex.lower(), cancer_type)
        prevalence = get_prevalence_from_incidence(incidence_rate)
        if risk_multipliers and cancer_type in risk_multipliers:
            prevalence *= risk_multipliers[cancer_type]
        total_prevalence += prevalence
    return total_prevalence

def to_fraction(risk_percent):
    if risk_percent <= 0:
        return "0 in 10,000"
    n = round(100 / risk_percent)
    return f"1 in {n}"

# Sidebar
with st.sidebar:
    st.header("Your Profile")
    name = st.text_input("Your Name (optional)", "")
    age = st.slider("Age", 30, 90, 55)
    sex = st.selectbox("Biological Sex", ["Male", "Female"])
    test_types = st.multiselect("Screening Tests to Compare", ["Whole-body MRI", "Galleri Blood Test", "CT Scan"], default=["Whole-body MRI"])
    st.header("Risk Factors")
    smoking_status = st.selectbox("Smoking", ["Never smoked", "Former smoker", "Current smoker"])
    if smoking_status != "Never smoked":
        pack_years = st.slider("Pack-years", 0, 80, 20, 5)
    else:
        pack_years = 0
    family_history = st.multiselect("Family Cancer History", ["Breast cancer", "Colorectal cancer", "Prostate cancer", "Ovarian cancer", "Lung cancer", "Pancreatic cancer"])
    genetic_mutations = st.multiselect("Genetic Mutations", ["BRCA1", "BRCA2", "Lynch syndrome", "TP53 (Li-Fraumeni)"])
    personal_history = st.checkbox("Personal Cancer History")
    use_custom_probability = st.checkbox("Use Custom Overall Risk (%)")
    if use_custom_probability:
        custom_probability = st.slider("Custom Risk", 0.1, 50.0, 5.0, 0.1)

# Personalized Greeting
if name:
    st.markdown(f"<div class='personalized-header'>Hello, {name}! Here's your personalized cancer risk exploration based on age {age} and {sex} profile.</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='personalized-header'>Your Personalized Cancer Risk Exploration (Age {age}, {sex})</div>", unsafe_allow_html=True)

# Tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔍 Risk Details", "📈 Charts", "⚠️ Risks & Next Steps"])

# Calculations (support multiple tests)
results_dict = {}
overall_metrics = {}
for test_type in test_types:
    cancer_types = list(TEST_PERFORMANCE[test_type].keys())
    risk_multipliers = {}
    for cancer_type in cancer_types:
        if (cancer_type in ["prostate", "testicular"] and sex == "Female") or (cancer_type in ["ovarian", "cervical", "endometrial", "uterine"] and sex == "Male"):
            continue
        risk_multipliers[cancer_type] = get_risk_multiplier(cancer_type, smoking_status, pack_years, family_history, genetic_mutations, personal_history)

    baseline_overall_prevalence = calculate_overall_cancer_prevalence(age, sex.lower())
    personalized_overall_prevalence = calculate_overall_cancer_prevalence(age, sex.lower(), risk_multipliers)

    results = []
    overall_tp = overall_fp = overall_fn = overall_tn = overall_pre_test_risk = overall_post_test_risk = overall_post_positive_risk = 0
    overall_total = 0

    for cancer_type in cancer_types:
        if (cancer_type in ["prostate", "testicular"] and sex == "Female") or (cancer_type in ["ovarian", "cervical", "endometrial", "uterine"] and sex == "Male"):
            continue
        sensitivity = TEST_PERFORMANCE[test_type][cancer_type]["sensitivity"]
        specificity = TEST_PERFORMANCE[test_type][cancer_type]["specificity"]
        if use_custom_probability:
            prevalence = custom_probability / 100 / len(cancer_types)  # Divide custom by types for per cancer
        else:
            incidence_rate = interpolate_incidence(age, sex.lower(), cancer_type)
            baseline_prevalence = get_prevalence_from_incidence(incidence_rate)
            prevalence = baseline_prevalence * risk_multipliers.get(cancer_type, 1)
        ppv, npv = calculate_ppv_npv(sensitivity, specificity, prevalence)
        post_test_risk_neg = calculate_post_test_risk_negative(sensitivity, specificity, prevalence)
        post_test_risk_pos = ppv
        false_positive_risk = (1 - specificity) * (1 - prevalence)
        baseline_incidence = interpolate_incidence(age, sex.lower(), cancer_type)
        baseline_risk = get_prevalence_from_incidence(baseline_incidence)
        abs_risk_reduction = prevalence - post_test_risk_neg
        results.append({
            "Cancer Type": cancer_type.replace("_", " ").title(),
            "Baseline Risk": round(baseline_risk * 100, 3),
            "Your Risk": round(prevalence * 100, 3),
            "Risk Multiplier": round(risk_multipliers.get(cancer_type, 1), 1),
            "Post-test Risk (if negative)": round(post_test_risk_neg * 100, 4),
            "Post-test Risk (if positive)": round(post_test_risk_pos * 100, 4),
            "Abs Risk Reduction": round(abs_risk_reduction * 100, 4),
            "Rel Risk Reduction": round(((prevalence - post_test_risk_neg) / prevalence) * 100, 1) if prevalence > 0 else 0,
            "False Positive Risk": round(false_positive_risk * 100, 2),
            "Detection Rate": round(sensitivity * 100, 1),
            "Accuracy Rate": round(specificity * 100, 1),
            "Positive Accuracy": round(ppv * 100, 1),
            "Negative Accuracy": round(npv * 100, 1)
        })
        tp = sensitivity * prevalence
        fp = (1 - specificity) * (1 - prevalence)
        fn = (1 - sensitivity) * prevalence
        tn = specificity * (1 - prevalence)
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        overall_tn += tn
        overall_pre_test_risk += prevalence
        overall_post_test_risk += post_test_risk_neg
        overall_post_positive_risk += post_test_risk_pos * (tp + fp)
        overall_total += (tp + fp + fn + tn)

    df = pd.DataFrame(results)
    df = df.sort_values("Your Risk", ascending=False).reset_index(drop=True)
    results_dict[test_type] = df

    pie_values = [overall_tp / overall_total * 100, overall_fp / overall_total * 100, overall_fn / overall_total * 100, overall_tn / overall_total * 100] if overall_total > 0 else [0, 0, 0, 100]

    overall_abs_reduction = (overall_pre_test_risk - overall_post_test_risk) * 100
    overall_rel_reduction = ((overall_pre_test_risk - overall_post_test_risk) / overall_pre_test_risk * 100) if overall_pre_test_risk > 0 else 0
    overall_post_positive_risk = overall_post_positive_risk / overall_total if overall_total > 0 else 0

    overall_metrics[test_type] = {
        "current_risk": personalized_overall_prevalence * 100,
        "post_neg_risk": overall_post_test_risk * 100,
        "risk_drop_abs": overall_abs_reduction,
        "risk_drop_rel": overall_rel_reduction,
        "fp_rate": DOWNSTREAM_RISKS[test_type]['false_positive_rate'],
        "biopsy_rate": DOWNSTREAM_RISKS[test_type]['false_positive_rate'] * DOWNSTREAM_RISKS[test_type]['biopsy_rate_fp'],
        "comp_rate": DOWNSTREAM_RISKS[test_type]['false_positive_rate'] * DOWNSTREAM_RISKS[test_type]['biopsy_rate_fp'] * DOWNSTREAM_RISKS[test_type]['comp_rate_biopsy'],
        "pie_values": pie_values
    }

with tab1:
    st.markdown('<div class="info-box">Based on your inputs, your overall risk is adjusted by factors like smoking and family history.</div>', unsafe_allow_html=True)
    cols = st.columns(len(test_types))
    for i, test_type in enumerate(test_types):
        with cols[i]:
            st.subheader(test_type)
            metrics = overall_metrics[test_type]
            st.metric("Current Risk", f"{metrics['current_risk']:.2f}% ({to_fraction(metrics['current_risk'])})")
            st.metric("Risk After Negative Test", f"{metrics['post_neg_risk']:.2f}% ({to_fraction(metrics['post_neg_risk'])})")
            st.metric("Risk Drop", f"{metrics['risk_drop_abs']:.2f}%", f"{metrics['risk_drop_rel']:.1f}% relative")
            st.metric("False Positive Chance", f"{metrics['fp_rate']}% ({to_fraction(metrics['fp_rate'])})")
            st.metric("Biopsy/Complication Chance", f"{metrics['biopsy_rate']:.2f}% / {metrics['comp_rate']:.4f}%")

with tab2:
    for test_type in test_types:
        st.subheader(f"Detailed Risks for {test_type}")
        df = results_dict[test_type]
        simplified_df = df[["Cancer Type", "Your Risk", "Post-test Risk (if negative)", "Post-test Risk (if positive)", "Abs Risk Reduction", "Rel Risk Reduction", "False Positive Risk"]]
        formatter = {
            "Your Risk": lambda x: f"{x:.3f}% ({to_fraction(x)})",
            "Post-test Risk (if negative)": lambda x: f"{x:.4f}% ({to_fraction(x)})",
            "Post-test Risk (if positive)": lambda x: f"{x:.4f}% ({to_fraction(x)})",
            "Abs Risk Reduction": lambda x: f"{x:.4f}% ({to_fraction(x)})",
            "Rel Risk Reduction": "{:.1f}%",
            "False Positive Risk": lambda x: f"{x:.2f}% ({to_fraction(x)})"
        }
        st.dataframe(simplified_df.style.format(formatter))

with tab3:
    for test_type in test_types:
        st.subheader(f"Charts for {test_type}")
        df = results_dict[test_type]
        # Risk Change Chart
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(
            name='Current Risk',
            x=df["Cancer Type"],
            y=df["Your Risk"],
            marker_color='#3b82f6',
            textposition='outside',
            texttemplate='%{y:.2f}%',
            opacity=0.8,
            customdata=[to_fraction(y) for y in df["Your Risk"]],
            hovertemplate='%{x}<br>Risk: %{y:.2f}% (%{customdata})'
        ))
        fig_comparison.add_trace(go.Bar(
            name='After Negative',
            x=df["Cancer Type"],
            y=df["Post-test Risk (if negative)"],
            marker_color='#ef4444',
            textposition='outside',
            texttemplate='%{y:.2f}%',
            opacity=0.8,
            customdata=[to_fraction(y) for y in df["Post-test Risk (if negative)"]],
            hovertemplate='%{x}<br>Risk: %{y:.2f}% (%{customdata})'
        ))
        fig_comparison.add_trace(go.Bar(
            name='After Positive',
            x=df["Cancer Type"],
            y=df["Post-test Risk (if positive)"],
            marker_color='#f59e0b',
            textposition='outside',
            texttemplate='%{y:.2f}%',
            opacity=0.8,
            customdata=[to_fraction(y) for y in df["Post-test Risk (if positive)"]],
            hovertemplate='%{x}<br>Risk: %{y:.2f}% (%{customdata})'
        ))
        max_risk = max(df["Your Risk"].max(), df["Post-test Risk (if positive)"].max())
        y_max = max(1, max_risk * 1.3)
        fig_comparison.update_layout(
            barmode='group',
            height=600,
            yaxis_title='Risk (%)',
            yaxis=dict(range=[0, y_max]),
            legend=dict(orientation='h', yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='white',
            bargap=0.15,
            font=dict(family="Arial", size=12, color="#1f2937")
        )
        fig_comparison.update_xaxes(tickangle=45, tickfont_size=12)
        st.plotly_chart(fig_comparison, use_container_width=True)

        # Pie Chart
        pie_values = overall_metrics[test_type]["pie_values"]
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Correct Detection', 'False Alarm', 'Missed Cancer', 'Correct Clear'],
            values=pie_values,
            hole=.4,
            marker_colors=['#22c55e', '#ef4444', '#f59e0b', '#3b82f6']
        )])
        fig_pie.update_layout(height=400, font=dict(family="Arial", size=12, color="#1f2937"))
        st.plotly_chart(fig_pie, use_container_width=True)

with tab4:
    for test_type in test_types:
        st.subheader(f"Risks & Next Steps for {test_type}")
        risk_data = DOWNSTREAM_RISKS[test_type]
        st.markdown(f"**Typical Follow-up:** {risk_data['typical_followup']}")
        st.markdown(f"**Follow-up Complications Rate:** {risk_data['followup_complications']}%")
        st.markdown(f"**Psychological Impact:** {risk_data['psychological_impact']}")
        st.markdown(f"**Radiation Exposure:** {risk_data['radiation_exposure']}")

        # Bar for risks
        fig_risks = go.Figure(go.Bar(
            x=['False Positive', 'Biopsy', 'Complication'],
            y=[risk_data['false_positive_rate'], risk_data['false_positive_rate'] * risk_data['biopsy_rate_fp'], risk_data['false_positive_rate'] * risk_data['biopsy_rate_fp'] * risk_data['comp_rate_biopsy']],
            marker_color='#ef4444',
            textposition='outside',
            texttemplate='%{y:.2f}%'
        ))
        fig_risks.update_layout(height=300, yaxis_title='Chance (%)', plot_bgcolor='white', font=dict(family="Arial", size=12, color="#1f2937"))
        st.plotly_chart(fig_risks, use_container_width=True)

# Download Report
def generate_report():
    report = f"Personalized Cancer Risk Report for {name or 'User'}\nAge: {age}, Sex: {sex}\n\n"
    for test_type in test_types:
        report += f"--- {test_type} ---\n"
        metrics = overall_metrics[test_type]
        report += f"Current Risk: {metrics['current_risk']:.2f}%\n"
        report += f"Risk After Negative Test: {metrics['post_neg_risk']:.2f}%\n"
        report += f"Risk Drop: {metrics['risk_drop_abs']:.2f}% ({metrics['risk_drop_rel']:.1f}% relative)\n"
        report += f"False Positive Chance: {metrics['fp_rate']}%\n"
        report += f"Biopsy/Complication Chance: {metrics['biopsy_rate']:.2f}% / {metrics['comp_rate']:.4f}%\n\n"
        df = results_dict[test_type]
        report += "Detailed Risks:\n" + df.to_string() + "\n\n"
    return report

report_text = generate_report()
st.download_button("Download Your Personalized Report", report_text, "cancer_risk_report.txt")