import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import math

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
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# App config
st.set_page_config(page_title="Cancer Screening Outcomes Visualizer", page_icon="📊", layout="wide")

st.markdown("<h1 class='main-header'>Cancer Screening Outcomes Visualizer</h1>", unsafe_allow_html=True)
st.markdown("""
<div class='warning-box'>
⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational purposes only. The risk calculations are simplified models and should not replace professional medical advice. Always consult with a healthcare provider for medical decisions.
</div>
""", unsafe_allow_html=True)

st.markdown("Enter your details to see a personalized Sankey diagram of screening test outcomes for 100 or 1000 people like you. Data as of July 2025.")

# Data structures (unchanged from original)
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

DOWNSTREAM_RISKS = {
    "Whole-body MRI": {"false_positive_rate": 8.0, "biopsy_rate_fp": 0.5, "comp_rate_biopsy": 0.03, "psychological_impact": "Moderate", "radiation_exposure": "None"},
    "Galleri Blood Test": {"false_positive_rate": 0.5, "biopsy_rate_fp": 0.5, "comp_rate_biopsy": 0.03, "psychological_impact": "High", "radiation_exposure": "From follow-ups"},
    "Low-dose CT Scan": {"false_positive_rate": 4.8, "biopsy_rate_fp": 0.5, "comp_rate_biopsy": 0.03, "psychological_impact": "Moderate", "radiation_exposure": "Low"},
    "Colonoscopy": {"false_positive_rate": 5.0, "biopsy_rate_fp": 0.8, "comp_rate_biopsy": 0.01, "psychological_impact": "Low", "radiation_exposure": "None"},
    "Upper Endoscopy": {"false_positive_rate": 10.0, "biopsy_rate_fp": 0.6, "comp_rate_biopsy": 0.02, "psychological_impact": "Moderate", "radiation_exposure": "None"},
    "Dermoscopy": {"false_positive_rate": 15.0, "biopsy_rate_fp": 0.3, "comp_rate_biopsy": 0.005, "psychological_impact": "Low", "radiation_exposure": "None"},
    "Mammography": {"false_positive_rate": 10.0, "biopsy_rate_fp": 0.4, "comp_rate_biopsy": 0.01, "psychological_impact": "Moderate", "radiation_exposure": "Low"},
    "HPV Test": {"false_positive_rate": 6.0, "biopsy_rate_fp": 0.5, "comp_rate_biopsy": 0.01, "psychological_impact": "Low", "radiation_exposure": "None"},
    "PSA Test": {"false_positive_rate": 70.0, "biopsy_rate_fp": 0.6, "comp_rate_biopsy": 0.02, "psychological_impact": "Moderate", "radiation_exposure": "None"},
    "Skin Exam": {"false_positive_rate": 11.0, "biopsy_rate_fp": 0.2, "comp_rate_biopsy": 0.005, "psychological_impact": "Low", "radiation_exposure": "None"}
}

# ============================================================================
# FIXED FUNCTIONS
# ============================================================================

@st.cache_data
def interpolate_incidence(age, sex, cancer_type):
    """Interpolate cancer incidence rate for given age/sex/cancer type"""
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

def get_risk_multiplier_fixed(cancer_type, smoking_status, pack_years, family_history, family_ages, genetic_mutations, personal_history):
    """
    Calculate risk multiplier using logarithmic combination to prevent extreme values
    """
    risk_factors = []  # Log-scale risk factors to be summed
    
    # Smoking risk factors (log scale)
    if smoking_status == "Current smoker":
        if cancer_type == "lung":
            if pack_years < 20:
                risk_factors.append(math.log(10))  # 10x
            elif pack_years < 40:
                risk_factors.append(math.log(15))  # 15x
            else:
                risk_factors.append(math.log(20))  # 20x (reduced from 35x)
        elif cancer_type in ["bladder", "kidney", "pancreatic", "cervical", "esophageal", "gastric", "head_neck"]:
            risk_factors.append(math.log(2.0))  # 2x (reduced from 2.5x)
        elif cancer_type in ["colorectal", "liver"]:
            risk_factors.append(math.log(1.6))  # 1.6x (reduced from 1.8x)
    
    elif smoking_status == "Former smoker":
        if cancer_type == "lung":
            if pack_years < 20:
                risk_factors.append(math.log(5))   # 5x (reduced from 8x)
            elif pack_years < 40:
                risk_factors.append(math.log(8))   # 8x (reduced from 12x)
            else:
                risk_factors.append(math.log(12))  # 12x (reduced from 18x)
        elif cancer_type in ["bladder", "kidney", "pancreatic", "cervical", "esophageal", "gastric", "head_neck"]:
            risk_factors.append(math.log(1.5))  # 1.5x (reduced from 1.8x)
        elif cancer_type in ["colorectal", "liver"]:
            risk_factors.append(math.log(1.3))  # 1.3x (reduced from 1.4x)
    
    # Family history with diminishing returns
    cancer_family_map = {
        "breast": "Breast cancer",
        "colorectal": "Colorectal cancer", 
        "prostate": "Prostate cancer",
        "ovarian": "Ovarian cancer",
        "lung": "Lung cancer",
        "pancreatic": "Pancreatic cancer"
    }
    
    family_cancer = cancer_family_map.get(cancer_type)
    if family_cancer in family_history:
        min_age = min(family_ages.get(family_cancer, [60])) if family_ages.get(family_cancer) else 60
        family_count = len(family_ages.get(family_cancer, []))
        
        # Base family history multiplier (reduced from original)
        if cancer_type == "colorectal":
            base_multiplier = 2.5 if min_age < 60 else 1.8  # Reduced from 3.5/2.2
        elif cancer_type == "breast":
            base_multiplier = 2.2 if min_age < 50 else 1.8  # Reduced from 3.0/2.3
        elif cancer_type == "prostate":
            base_multiplier = 2.0  # Reduced from 2.5
        elif cancer_type == "ovarian":
            base_multiplier = 2.5  # Reduced from 3.1
        elif cancer_type in ["lung", "pancreatic"]:
            base_multiplier = 1.5  # Reduced from 1.8
        else:
            base_multiplier = 1.3
        
        # Diminishing returns for multiple relatives
        family_factor = 1.0 + 0.2 * min(family_count - 1, 2)  # Max 1.4x for multiple relatives
        
        total_family_multiplier = base_multiplier * family_factor
        risk_factors.append(math.log(total_family_multiplier))
    
    # Genetic mutations (significantly reduced)
    if "BRCA1" in genetic_mutations:
        if cancer_type == "breast":
            risk_factors.append(math.log(8))   # 8x (reduced from 35x)
        elif cancer_type == "ovarian":
            risk_factors.append(math.log(6))   # 6x (reduced from 20x)
    
    if "BRCA2" in genetic_mutations:
        if cancer_type == "breast":
            risk_factors.append(math.log(5))   # 5x (reduced from 20x)
        elif cancer_type == "ovarian":
            risk_factors.append(math.log(3))   # 3x (reduced from 8x)
        elif cancer_type == "prostate":
            risk_factors.append(math.log(3))   # 3x (reduced from 4.5x)
    
    if "Lynch syndrome" in genetic_mutations:
        if cancer_type == "colorectal":
            risk_factors.append(math.log(6))   # 6x (reduced from 15x)
        elif cancer_type == "ovarian":
            risk_factors.append(math.log(3))   # 3x (reduced from 6x)
        elif cancer_type == "endometrial":
            risk_factors.append(math.log(5))   # 5x (reduced from 12x)
    
    if "TP53 (Li-Fraumeni)" in genetic_mutations:
        if cancer_type in ["breast", "lung", "colorectal", "liver", "brain", "sarcoma"]:
            risk_factors.append(math.log(4))   # 4x (reduced from 10x)
    
    # Personal history
    if personal_history:
        risk_factors.append(math.log(2.0))  # 2x (reduced from 2.5x)
    
    # Combine on log scale and convert back
    if not risk_factors:
        return 1.0
    
    total_log_risk = sum(risk_factors)
    final_multiplier = math.exp(total_log_risk)
    
    # Cap at 25x maximum (much more reasonable)
    return min(final_multiplier, 25.0)

def calculate_overall_prevalence_fixed(age, sex, risk_multipliers=None):
    """
    Calculate overall cancer prevalence using proper probability combination
    """
    cancer_probabilities = []
    individual_risks = {}
    
    for cancer_type in CANCER_INCIDENCE.keys():
        # Skip sex-inappropriate cancers
        if ((cancer_type in ["prostate", "testicular"] and sex == "female") or 
            (cancer_type in ["ovarian", "cervical", "endometrial", "uterine"] and sex == "male")):
            continue
            
        incidence_rate = interpolate_incidence(age, sex, cancer_type)
        prevalence = (incidence_rate / 100000) * 10  # 10-year risk
        
        if risk_multipliers and cancer_type in risk_multipliers:
            prevalence *= risk_multipliers[cancer_type]
        
        # Cap individual cancer risk at 60% (very high but realistic)
        prevalence = min(prevalence, 0.60)
        cancer_probabilities.append(prevalence)
        individual_risks[cancer_type] = prevalence
    
    # Use proper probability combination for independent events
    # P(any cancer) = 1 - P(no cancer) = 1 - ∏(1 - P(cancer_i))
    prob_no_cancer = 1.0
    for prob in cancer_probabilities:
        prob_no_cancer *= (1 - prob)
    
    overall_prevalence = 1 - prob_no_cancer
    
    # Additional safety cap at 85%
    return min(overall_prevalence, 0.85), individual_risks

def calculate_per_cancer_prevalence_fixed(age, sex, risk_multipliers=None):
    """Calculate personalized risk for each cancer type"""
    results = []
    for cancer_type in CANCER_INCIDENCE.keys():
        if ((cancer_type in ["prostate", "testicular"] and sex == "female") or 
            (cancer_type in ["ovarian", "cervical", "endometrial", "uterine"] and sex == "male")):
            continue
            
        incidence_rate = interpolate_incidence(age, sex, cancer_type)
        prevalence = (incidence_rate / 100000) * 10
        
        if risk_multipliers and cancer_type in risk_multipliers:
            prevalence *= risk_multipliers[cancer_type]
        
        # Cap individual risk
        prevalence = min(prevalence, 0.60)
        
        results.append({
            "Cancer Type": cancer_type.replace("_", " ").title(), 
            "Base Risk (%)": (incidence_rate / 100000) * 10 * 100,
            "Risk Multiplier": risk_multipliers.get(cancer_type, 1.0) if risk_multipliers else 1.0,
            "Personalized Risk (%)": prevalence * 100
        })
    
    return pd.DataFrame(results)

def combine_tests(tests, mode, age, sex, risk_multipliers):
    """Combine multiple tests with proper probability handling"""
    cancer_outcomes = {}
    for cancer_type in CANCER_INCIDENCE.keys():
        if ((cancer_type in ["prostate", "testicular"] and sex == "female") or 
            (cancer_type in ["ovarian", "cervical", "endometrial", "uterine"] and sex == "male")):
            continue
            
        sens_combined = 0 if mode == "Parallel" else 1
        spec_combined = 1 if mode == "Parallel" else 0
        
        for test in tests:
            if cancer_type in TEST_PERFORMANCE.get(test, {}):
                sens = TEST_PERFORMANCE[test][cancer_type]["sensitivity"]
                spec = TEST_PERFORMANCE[test][cancer_type]["specificity"]
                
                if mode == "Parallel":
                    sens_combined = 1 - (1 - sens_combined) * (1 - sens)
                    spec_combined *= spec
                else:
                    sens_combined *= sens
                    spec_combined = 1 - (1 - spec_combined) * (1 - spec)
        
        incidence_rate = interpolate_incidence(age, sex, cancer_type)
        prevalence = (incidence_rate / 100000) * 10
        
        if risk_multipliers and cancer_type in risk_multipliers:
            prevalence *= risk_multipliers[cancer_type]
        
        prevalence = min(prevalence, 0.60)  # Cap individual cancer risk
        cancer_outcomes[cancer_type] = {
            "sensitivity": sens_combined, 
            "specificity": spec_combined, 
            "prevalence": prevalence
        }

    # Aggregate outcomes
    overall_sens = np.mean([co["sensitivity"] for co in cancer_outcomes.values()])
    overall_spec = np.mean([co["specificity"] for co in cancer_outcomes.values()])
    return overall_sens, overall_spec, cancer_outcomes

def calculate_y_positions(values, total_height=0.9, min_spacing=0.05):
    """Calculate y-positions for Sankey nodes to avoid overlap"""
    total_value = sum(values)
    if total_value == 0:
        return [0.5] * len(values)
    
    normalized = [v / total_value for v in values]
    y_positions = []
    current_y = 0.05
    
    for norm in normalized:
        y_positions.append(current_y + norm * total_height / 2)
        current_y += norm * total_height + min_spacing
    
    return y_positions

def validate_inputs(age, smoking_status, pack_years, family_ages):
    """Validate user inputs and return error messages"""
    errors = []
    warnings = []
    
    if not 30 <= age <= 90:
        errors.append("Age must be between 30 and 90")
    
    if smoking_status != "Never smoked" and pack_years == 0:
        warnings.append("Please enter pack-years for smokers (pack-years = packs per day × years smoked)")
    
    if smoking_status == "Never smoked" and pack_years > 0:
        warnings.append("Pack-years should be 0 for never smokers")
    
    # Validate family history ages
    for cancer, ages in family_ages.items():
        for age_val in ages:
            if age_val < 20 or age_val > 100:
                errors.append(f"Invalid age {age_val} for {cancer}. Ages should be between 20-100.")
    
    return errors, warnings

# ============================================================================
# STREAMLIT UI
# ============================================================================

# Sidebar inputs
with st.sidebar:
    st.header("Your Details")
    
    # Basic demographics
    age = st.slider("Age", 30, 90, 55)
    sex = st.selectbox("Sex", ["male", "female"])
    
    # Smoking history
    smoking_status = st.selectbox("Smoking Status", ["Never smoked", "Former smoker", "Current smoker"])
    pack_years = 0
    if smoking_status != "Never smoked":
        pack_years = st.slider("Pack-years (packs per day × years smoked)", 0, 80, 20)
        st.caption("Example: 1 pack/day for 20 years = 20 pack-years")
    
    # Family history
    family_history = st.multiselect(
        "Family Cancer History (1st/2nd degree relatives)", 
        ["Breast cancer", "Colorectal cancer", "Prostate cancer", "Ovarian cancer", "Lung cancer", "Pancreatic cancer"]
    )
    
    family_ages = {}
    for fh in family_history:
        ages_input = st.text_input(
            f"Ages of Diagnosis for {fh} (comma-separated)", 
            value="60", 
            key=fh,
            help="Enter ages when family members were diagnosed, separated by commas"
        )
        try:
            family_ages[fh] = [int(x.strip()) for x in ages_input.split(",") if x.strip().isdigit()]
        except:
            family_ages[fh] = [60]  # Default fallback
    
    # Genetic testing
    genetic_mutations = st.multiselect(
        "Known Genetic Mutations", 
        ["BRCA1", "BRCA2", "Lynch syndrome", "TP53 (Li-Fraumeni)"],
        help="Only include mutations confirmed by genetic testing"
    )
    
    # Personal history
    personal_history = st.checkbox("Personal Cancer History", help="Have you been diagnosed with cancer before?")
    
    # Screening tests
    st.subheader("Screening Tests to Evaluate")
    tests = st.multiselect(
        "Select Tests", 
        list(TEST_PERFORMANCE.keys()),
        help="Choose which screening tests to include in the analysis"
    )
    
    # Display options
    st.subheader("Display Options")
    per_thousand = st.checkbox("Show per 1000 people", value=False)
    population = 1000 if per_thousand else 100
    
    # Input validation
    errors, warnings = validate_inputs(age, smoking_status, pack_years, family_ages)
    
    if errors:
        for error in errors:
            st.error(f"❌ {error}")
    
    if warnings:
        for warning in warnings:
            st.warning(f"⚠️ {warning}")

# Main calculations
if not errors:  # Only proceed if no validation errors
    
    # Calculate risk multipliers for all cancer types
    cancer_types = set(CANCER_INCIDENCE.keys())
    risk_multipliers = {}
    
    for ct in cancer_types:
        if not ((ct in ["prostate", "testicular"] and sex == "female") or 
                (ct in ["ovarian", "cervical", "endometrial", "uterine"] and sex == "male")):
            risk_multipliers[ct] = get_risk_multiplier_fixed(
                ct, smoking_status, pack_years, family_history, 
                family_ages, genetic_mutations, personal_history
            )
    
    # Calculate overall risk
    overall_prevalence, individual_risks = calculate_overall_prevalence_fixed(age, sex, risk_multipliers)
    
    # Display risk summary
    st.header("Your Personalized Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="10-Year Cancer Risk",
            value=f"{overall_prevalence*100:.1f}%",
            help="Estimated probability of developing any cancer in the next 10 years"
        )
    
    with col2:
        affected = int(overall_prevalence * population)
        st.metric(
            label="Natural Frequency",
            value=f"{affected} of {population}",
            help=f"Out of {population} people with your risk profile, about {affected} would develop cancer"
        )
    
    with col3:
        if overall_prevalence < 0.05:
            risk_level = "Low"
            risk_color = "🟢"
        elif overall_prevalence < 0.15:
            risk_level = "Moderate" 
            risk_color = "🟡"
        else:
            risk_level = "High"
            risk_color = "🔴"
        
        st.metric(
            label="Risk Level",
            value=f"{risk_color} {risk_level}",
            help="Relative risk category"
        )
    
    # Show highest risk cancers
    if individual_risks:
        top_risks = sorted(individual_risks.items(), key=lambda x: x[1], reverse=True)[:3]
        st.subheader("Your Highest Risk Cancers")
        
        for cancer, risk in top_risks:
            if risk > 0.001:  # Only show risks > 0.1%
                st.write(f"• **{cancer.replace('_', ' ').title()}**: {risk*100:.1f}% (10-year risk)")
    
    # Screening test analysis
    if tests:
        st.header("Screening Test Analysis")
        
        if len(tests) > 1:
            mode = st.selectbox(
                "How to combine multiple tests:",
                ["Parallel (Any positive)", "Sequential (All positive)"],
                help="Parallel: positive if ANY test is positive. Sequential: positive only if ALL tests are positive."
            )
            sens, spec, cancer_outcomes = combine_tests(tests, mode, age, sex, risk_multipliers)
            
            # Calculate combined downstream risks
            fp_rate = np.mean([DOWNSTREAM_RISKS[t]["false_positive_rate"] for t in tests]) / 100
            biopsy_rates = [DOWNSTREAM_RISKS[t]["biopsy_rate_fp"] for t in tests]
            base_biopsy_rate = np.mean(biopsy_rates)
            adjusted_biopsy_rate = max(0.1, min(0.9, base_biopsy_rate * (1 - spec)))
            comp_rate = np.mean([DOWNSTREAM_RISKS[t]["comp_rate_biopsy"] for t in tests])
            
        else:
            test = tests[0]
            sens = np.mean([TEST_PERFORMANCE[test].get(ct, {"sensitivity": 0})["sensitivity"] 
                          for ct in cancer_types if ct in TEST_PERFORMANCE[test]])
            spec = np.mean([TEST_PERFORMANCE[test].get(ct, {"specificity": 1})["specificity"] 
                          for ct in cancer_types if ct in TEST_PERFORMANCE[test]])
            fp_rate = DOWNSTREAM_RISKS[test]["false_positive_rate"] / 100
            adjusted_biopsy_rate = DOWNSTREAM_RISKS[test]["biopsy_rate_fp"]
            comp_rate = DOWNSTREAM_RISKS[test]["comp_rate_biopsy"]

        # Calculate outcomes for the population
        has_cancer = overall_prevalence * population
        no_cancer = population - has_cancer

        tp = sens * has_cancer  # True positives
        fn = (1 - sens) * has_cancer  # False negatives
        fp = (1 - spec) * no_cancer  # False positives
        tn = spec * no_cancer  # True negatives

        positive = tp + fp  # Total positive tests
        negative = fn + tn  # Total negative tests

        biopsy = positive * adjusted_biopsy_rate
        no_biopsy = positive - biopsy
        complication = biopsy * comp_rate
        no_complication = biopsy - complication

        cancer_treated = tp * 0.8  # Assume 80% of detected cancers are successfully treated
        benign = tp * 0.2 + fp  # Benign findings include some true positives + false positives
        further_monitor = fn  # False negatives need further monitoring
        reassured = tn  # True negatives are reassured

        # Create Sankey diagram
        st.subheader("Patient Flow Diagram")
        
        # Calculate dynamic y-positions for each stage
        stage1_values = [positive, negative]
        stage1_y = calculate_y_positions(stage1_values)
        
        stage2_values = [biopsy, no_biopsy, reassured, further_monitor]
        stage2_y = calculate_y_positions(stage2_values)
        
        stage3_values = [cancer_treated, benign, complication]
        stage3_y = calculate_y_positions(stage3_values)

        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            orientation='h',
            node=dict(
                pad=20,
                thickness=30,
                line=dict(color="gray", width=0.5),
                label=[
                    f"{population} People Screened",
                    f"Test Positive ({positive:.1f})", 
                    f"Test Negative ({negative:.1f})",
                    f"Biopsy Performed ({biopsy:.1f})", 
                    f"No Biopsy ({no_biopsy:.1f})",
                    f"Reassured ({reassured:.1f})", 
                    f"Monitoring Needed ({further_monitor:.1f})",
                    f"Cancer Detected & Treated ({cancer_treated:.1f})", 
                    f"Benign Finding ({benign:.1f})",
                    f"Biopsy Complication ({complication:.1f})"
                ],
                color=[
                    "#2E86AB", "#A23B72", "#F18F01", 
                    "#C73E1D", "#86A873", "#4B9DE8", "#FF6B6B",
                    "#51CF66", "#FFD93D", "#FF8E53"
                ],
                x=[0, 0.2, 0.2, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9, 0.9],
                y=[
                    0.5,  # Root node centered
                    stage1_y[0], stage1_y[1],  # Stage 1
                    stage2_y[0], stage2_y[1], stage2_y[2], stage2_y[3],  # Stage 2
                    stage3_y[0], stage3_y[1], stage3_y[2]  # Stage 3
                ],
            ),
            link=dict(
                source=[0, 0, 1, 1, 2, 2, 3, 3, 3, 4],
                target=[1, 2, 3, 4, 5, 6, 7, 8, 9, 8],
                value=[positive, negative, biopsy, no_biopsy, reassured, further_monitor, 
                      cancer_treated, benign, complication, no_biopsy],
                color='rgba(100, 150, 200, 0.3)',
                hovertemplate='%{source.label} → %{target.label}<br>Count: %{value:.1f}<extra></extra>'
            ),
            textfont=dict(size=12, color="black")
        )])

        fig.update_layout(
            title=f"Screening Outcomes for {population} People Like You",
            font_size=12,
            height=700,
            width=1200,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Key metrics
        with st.expander("📊 Detailed Results", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Test Performance:**")
                st.write(f"• Sensitivity: {sens:.1%}")
                st.write(f"• Specificity: {spec:.1%}")
                st.write(f"• Positive Predictive Value: {(tp/positive if positive > 0 else 0):.1%}")
                st.write(f"• Negative Predictive Value: {(tn/negative if negative > 0 else 0):.1%}")
            
            with col2:
                st.markdown("**Expected Outcomes:**")
                st.write(f"• Cancers detected: {cancer_treated:.1f}")
                st.write(f"• Cancers missed: {further_monitor:.1f}")
                st.write(f"• False alarms: {fp:.1f}")
                st.write(f"• Biopsy complications: {complication:.1f}")

    else:
        # No tests selected - show baseline risk
        st.header("Baseline Risk (No Screening)")
        
        has_cancer = overall_prevalence * population
        no_cancer = population - has_cancer

        # Simple baseline Sankey
        baseline_y = calculate_y_positions([has_cancer, no_cancer])

        fig = go.Figure(data=[go.Sankey(
            orientation='h',
            node=dict(
                pad=20,
                thickness=30,
                line=dict(color="gray", width=0.5),
                label=[
                    f"{population} People",
                    f"Will Develop Cancer ({has_cancer:.1f})", 
                    f"Will Not Develop Cancer ({no_cancer:.1f})"
                ],
                color=["#2E86AB", "#FF6B6B", "#51CF66"],
                x=[0, 0.5, 0.5],
                y=[0.5, baseline_y[0], baseline_y[1]],
            ),
            link=dict(
                source=[0, 0],
                target=[1, 2],
                value=[has_cancer, no_cancer],
                color='rgba(100, 150, 200, 0.3)',
                hovertemplate='%{source.label} → %{target.label}<br>Count: %{value:.1f}<extra></extra>'
            ),
            textfont=dict(size=14, color="black")
        )])

        fig.update_layout(
            title=f"10-Year Cancer Risk for {population} People Like You (No Screening)",
            font_size=14,
            height=400,
            width=1000
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Per-cancer risk breakdown
    st.header("Risk Breakdown by Cancer Type")
    per_cancer_df = calculate_per_cancer_prevalence_fixed(age, sex, risk_multipliers)
    
    # Sort by personalized risk and show top 10
    per_cancer_df_sorted = per_cancer_df.sort_values("Personalized Risk (%)", ascending=False).head(10)
    
    # Format the dataframe for better display
    per_cancer_df_sorted["Base Risk (%)"] = per_cancer_df_sorted["Base Risk (%)"].round(3)
    per_cancer_df_sorted["Risk Multiplier"] = per_cancer_df_sorted["Risk Multiplier"].round(1)
    per_cancer_df_sorted["Personalized Risk (%)"] = per_cancer_df_sorted["Personalized Risk (%)"].round(2)
    
    st.dataframe(
        per_cancer_df_sorted,
        use_container_width=True,
        hide_index=True
    )
    
    # Risk factors summary
    st.header("Your Risk Factors")
    active_factors = []
    
    if smoking_status != "Never smoked":
        active_factors.append(f"Smoking ({smoking_status.lower()}, {pack_years} pack-years)")
    
    if family_history:
        family_summary = ", ".join([f"{fh} (ages: {', '.join(map(str, family_ages.get(fh, [])))})" 
                                   for fh in family_history])
        active_factors.append(f"Family history: {family_summary}")
    
    if genetic_mutations:
        active_factors.append(f"Genetic mutations: {', '.join(genetic_mutations)}")
    
    if personal_history:
        active_factors.append("Personal cancer history")
    
    if active_factors:
        for factor in active_factors:
            st.write(f"• {factor}")
    else:
        st.write("• No major risk factors identified")
    
    # Limitations and disclaimers
    with st.expander("⚠️ Important Limitations"):
        st.markdown("""
        **This tool has important limitations:**
        
        - Risk estimates are based on population data and may not reflect your individual risk
        - Not all risk factors are included (diet, exercise, environmental exposures, etc.)
        - Risk multipliers are simplified approximations
        - Screening test performance varies by individual characteristics
        - Does not account for competing causes of death
        - Should not be used for medical decision-making
        
        **Always consult with a healthcare provider** who can assess your complete medical history and provide personalized recommendations.
        """)

else:
    st.error("Please correct the input errors above before proceeding.")