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
    
    if sex not in CANCER_INCIDENCE[cancer_type]:
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
                risk_factors.append(math.log(20))  # 20x
        elif cancer_type in ["bladder", "kidney", "pancreatic", "cervical", "esophageal", "gastric", "head_neck"]:
            risk_factors.append(math.log(2.0))  # 2x
        elif cancer_type in ["colorectal", "liver"]:
            risk_factors.append(math.log(1.6))  # 1.6x
    
    elif smoking_status == "Former smoker":
        if cancer_type == "lung":
            if pack_years < 20:
                risk_factors.append(math.log(5))   # 5x
            elif pack_years < 40:
                risk_factors.append(math.log(8))   # 8x
            else:
                risk_factors.append(math.log(12))  # 12x
        elif cancer_type in ["bladder", "kidney", "pancreatic", "cervical", "esophageal", "gastric", "head_neck"]:
            risk_factors.append(math.log(1.5))  # 1.5x
        elif cancer_type in ["colorectal", "liver"]:
            risk_factors.append(math.log(1.3))  # 1.3x
    
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
    if family_cancer and family_cancer in family_history:
        family_ages_list = family_ages.get(family_cancer, [60])
        if family_ages_list:  # Check if list is not empty
            min_age = min(family_ages_list)
            family_count = len(family_ages_list)
            
            # Base family history multiplier
            if cancer_type == "colorectal":
                base_multiplier = 2.5 if min_age < 60 else 1.8
            elif cancer_type == "breast":
                base_multiplier = 2.2 if min_age < 50 else 1.8
            elif cancer_type == "prostate":
                base_multiplier = 2.0
            elif cancer_type == "ovarian":
                base_multiplier = 2.5
            elif cancer_type in ["lung", "pancreatic"]:
                base_multiplier = 1.5
            else:
                base_multiplier = 1.3
            
            # Diminishing returns for multiple relatives
            family_factor = 1.0 + 0.2 * min(family_count - 1, 2)  # Max 1.4x for multiple relatives
            
            total_family_multiplier = base_multiplier * family_factor
            risk_factors.append(math.log(total_family_multiplier))
    
    # Genetic mutations
    if "BRCA1" in genetic_mutations:
        if cancer_type == "breast":
            risk_factors.append(math.log(8))   # 8x
        elif cancer_type == "ovarian":
            risk_factors.append(math.log(6))   # 6x
    
    if "BRCA2" in genetic_mutations:
        if cancer_type == "breast":
            risk_factors.append(math.log(5))   # 5x
        elif cancer_type == "ovarian":
            risk_factors.append(math.log(3))   # 3x
        elif cancer_type == "prostate":
            risk_factors.append(math.log(3))   # 3x
    
    if "Lynch syndrome" in genetic_mutations:
        if cancer_type == "colorectal":
            risk_factors.append(math.log(6))   # 6x
        elif cancer_type == "ovarian":
            risk_factors.append(math.log(3))   # 3x
        elif cancer_type == "endometrial":
            risk_factors.append(math.log(5))   # 5x
    
    if "TP53 (Li-Fraumeni)" in genetic_mutations:
        if cancer_type in ["breast", "lung", "colorectal", "liver", "brain", "sarcoma"]:
            risk_factors.append(math.log(4))   # 4x
    
    # Personal history
    if personal_history:
        risk_factors.append(math.log(2.0))  # 2x
    
    # Combine on log scale and convert back
    if not risk_factors:
        return 1.0
    
    total_log_risk = sum(risk_factors)
    final_multiplier = math.exp(total_log_risk)
    
    # Cap at 25x maximum
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
        
        # Cap individual cancer risk at 60%
        prevalence = min(prevalence, 0.60)
        cancer_probabilities.append(prevalence)
        individual_risks[cancer_type] = prevalence
    
    # Use proper probability combination for independent events
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
        base_prevalence = (incidence_rate / 100000) * 10
        
        multiplier = risk_multipliers.get(cancer_type, 1.0) if risk_multipliers else 1.0
        prevalence = base_prevalence * multiplier
        
        # Cap individual risk
        prevalence = min(prevalence, 0.60)
        
        results.append({
            "Cancer Type": cancer_type.replace("_", " ").title(), 
            "Base Risk (%)": base_prevalence * 100,
            "Risk Multiplier": multiplier,
            "Personalized Risk (%)": prevalence * 100
        })
    
    return pd.DataFrame(results)

def combine_tests(tests, mode, age, sex, risk_multipliers):
    """Combine multiple tests with proper probability handling"""
    cancer_outcomes = {}
    
    # Get all applicable cancer types for this person
    applicable_cancers = []
    for cancer_type in CANCER_INCIDENCE.keys():
        if not ((cancer_type in ["prostate", "testicular"] and sex == "female") or 
                (cancer_type in ["ovarian", "cervical", "endometrial", "uterine"] and sex == "male")):
            applicable_cancers.append(cancer_type)
    
    for cancer_type in applicable_cancers:
        sens_combined = 0 if mode == "Parallel" else 1
        spec_combined = 1 if mode == "Parallel" else 0
        
        # Check if any of the selected tests can detect this cancer
        applicable_tests = []
        for test in tests:
            if test in TEST_PERFORMANCE and cancer_type in TEST_PERFORMANCE[test]:
                applicable_tests.append(test)
        
        if applicable_tests:  # Only process if we have applicable tests
            for test in applicable_tests:
                sens = TEST_PERFORMANCE[test][cancer_type]["sensitivity"]
                spec = TEST_PERFORMANCE[test][cancer_type]["specificity"]
                
                if mode == "Parallel":
                    sens_combined = 1 - (1 - sens_combined) * (1 - sens)
                    spec_combined *= spec
                else:  # Sequential
                    sens_combined *= sens
                    spec_combined = 1 - (1 - spec_combined) * (1 - spec)
        else:
            # If no tests can detect this cancer, set to no detection
            sens_combined = 0
            spec_combined = 1
        
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

    # Calculate aggregate outcomes only for cancers that can be detected
    detectable_outcomes = {k: v for k, v in cancer_outcomes.items() if v["sensitivity"] > 0}
    
    if detectable_outcomes:
        overall_sens = np.mean([co["sensitivity"] for co in detectable_outcomes.values()])
        overall_spec = np.mean([co["specificity"] for co in detectable_outcomes.values()])
    else:
        overall_sens = 0
        overall_spec = 1
    
    return overall_sens, overall_spec, cancer_outcomes

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

def create_sankey_diagram(population, overall_prevalence, sens, spec, adjusted_biopsy_rate, comp_rate):
    """Create the Sankey diagram with fixed positioning"""
    
    # Calculate outcomes
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

    # Create Sankey diagram with fixed positioning
    fig = go.Figure(data=[go.Sankey(
        orientation='h',
        arrangement='snap',
        node=dict(
            pad=8,
            thickness=12,
            line=dict(color="gray", width=0.3),
            label=[
                f"{population} Screened",                 # 0
                f"Positive ({positive:.1f})",            # 1
                f"Negative ({negative:.1f})",            # 2
                f"Biopsy ({biopsy:.1f})",               # 3
                f"No Biopsy ({no_biopsy:.1f})",         # 4
                f"Reassured ({reassured:.1f})",         # 5
                f"Monitor ({further_monitor:.1f})",      # 6
                f"Cancer Found ({cancer_treated:.1f})",  # 7
                f"Benign ({benign:.1f})",               # 8
                f"Complication ({complication:.1f})"     # 9
            ],
            color=[
                "#2E86AB",  # Blue
                "#E74C3C", "#27AE60",  # Red/Green
                "#F39C12", "#3498DB", "#9B59B6", "#E67E22",  # Various
                "#1ABC9C", "#F1C40F", "#E74C3C"  # Teal/Yellow/Red
            ],
            x=[0.12, 0.31, 0.39, 0.61, 0.69, 0.61, 0.69, 0.88, 0.88, 0.88],
            y=[0.5, 0.3, 0.7, 0.2, 0.4, 0.8, 0.9, 0.25, 0.55, 0.4],
        ),
        link=dict(
            source=[0, 0, 1, 1, 2, 2, 3, 3, 3, 4],
            target=[1, 2, 3, 4, 5, 6, 7, 8, 9, 8],
            value=[positive, negative, biopsy, no_biopsy, reassured, further_monitor, 
                   cancer_treated, benign, complication, no_biopsy],
            color=[
                'rgba(231, 76, 60, 0.4)', 'rgba(39, 174, 96, 0.4)', 
                'rgba(243, 156, 18, 0.4)', 'rgba(52, 152, 219, 0.4)',
                'rgba(155, 89, 182, 0.4)', 'rgba(230, 126, 34, 0.4)',
                'rgba(26, 188, 156, 0.4)', 'rgba(241, 196, 15, 0.4)',
                'rgba(231, 76, 60, 0.4)', 'rgba(241, 196, 15, 0.4)'
            ],
            hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>' +
                         'Count: %{value:.1f} people<br><extra></extra>'
        ),
        textfont=dict(size=8, color="black", family="Arial"),
        domain=dict(x=[0.02, 0.98], y=[0.15, 0.85])
    )])

    fig.update_layout(
        title=dict(
            text=f"Screening Process Flow for {population} People Like You",
            x=0.5,
            font=dict(size=13, family="Arial")
        ),
        font=dict(size=8, family="Arial"),
        height=420,
        margin=dict(l=25, r=25, t=70, b=25),
        showlegend=False,
        plot_bgcolor='white',
        annotations=[
            dict(x=0.12, y=0.95, text="<b>Screening</b>", showarrow=False, 
                 font=dict(size=10, color="#34495e", family="Arial"), xref="paper", yref="paper"),
            dict(x=0.35, y=0.95, text="<b>Test Results</b>", showarrow=False,
                 font=dict(size=10, color="#34495e", family="Arial"), xref="paper", yref="paper"),
            dict(x=0.65, y=0.95, text="<b>Follow-up</b>", showarrow=False,
                 font=dict(size=10, color="#34495e", family="Arial"), xref="paper", yref="paper"),
            dict(x=0.88, y=0.95, text="<b>Outcomes</b>", showarrow=False,
                 font=dict(size=10, color="#34495e", family="Arial"), xref="paper", yref="paper")
        ]
    )
    
    return fig

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
            ages_list = []
            for x in ages_input.split(","):
                x = x.strip()
                if x and x.replace('.', '').isdigit():
                    age_val = int(float(x))
                    if 20 <= age_val <= 100:
                        ages_list.append(age_val)
            family_ages[fh] = ages_list if ages_list else [60]
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
            mode_short = mode.split()[0]  # Extract "Parallel" or "Sequential"
            sens, spec, cancer_outcomes = combine_tests(tests, mode_short, age, sex, risk_multipliers)
            
            # Calculate combined downstream risks
            fp_rates = []
            biopsy_rates = []
            comp_rates = []
            
            for t in tests:
                if t in DOWNSTREAM_RISKS:
                    fp_rates.append(DOWNSTREAM_RISKS[t]["false_positive_rate"])
                    biopsy_rates.append(DOWNSTREAM_RISKS[t]["biopsy_rate_fp"])
                    comp_rates.append(DOWNSTREAM_RISKS[t]["comp_rate_biopsy"])
            
            if fp_rates:  # Only calculate if we have data
                fp_rate = np.mean(fp_rates) / 100
                base_biopsy_rate = np.mean(biopsy_rates)
                comp_rate = np.mean(comp_rates)
            else:
                fp_rate = 0.05  # Default 5%
                base_biopsy_rate = 0.5
                comp_rate = 0.02
            
            adjusted_biopsy_rate = max(0.1, min(0.9, base_biopsy_rate))
            
        else:
            test = tests[0]
            
            # Get all cancer types this test can detect
            detectable_cancers = []
            for cancer_type in cancer_types:
                if not ((cancer_type in ["prostate", "testicular"] and sex == "female") or 
                        (cancer_type in ["ovarian", "cervical", "endometrial", "uterine"] and sex == "male")):
                    if cancer_type in TEST_PERFORMANCE.get(test, {}):
                        detectable_cancers.append(cancer_type)
            
            if detectable_cancers:
                sens = np.mean([TEST_PERFORMANCE[test][ct]["sensitivity"] for ct in detectable_cancers])
                spec = np.mean([TEST_PERFORMANCE[test][ct]["specificity"] for ct in detectable_cancers])
            else:
                sens = 0
                spec = 1
            
            if test in DOWNSTREAM_RISKS:
                fp_rate = DOWNSTREAM_RISKS[test]["false_positive_rate"] / 100
                adjusted_biopsy_rate = DOWNSTREAM_RISKS[test]["biopsy_rate_fp"]
                comp_rate = DOWNSTREAM_RISKS[test]["comp_rate_biopsy"]
            else:
                fp_rate = 0.05
                adjusted_biopsy_rate = 0.5
                comp_rate = 0.02

        # Create and display Sankey diagram
        st.subheader("Patient Flow Diagram")
        
        if sens > 0 or spec < 1:  # Only show if test has some effect
            fig = create_sankey_diagram(population, overall_prevalence, sens, spec, adjusted_biopsy_rate, comp_rate)
            st.plotly_chart(fig, use_container_width=True)

            # Add interpretation guide
            st.markdown("""
            **💡 How to read this diagram:** 
            Follow the flow from left to right to see what happens during screening. 
            Thicker flows represent more people, but all paths are sized for visibility.
            """)
        else:
            st.warning("⚠️ Selected test(s) cannot detect any cancers relevant to your demographic profile.")

        # Calculate individual outcomes for personal impact analysis
        has_cancer = overall_prevalence * population
        no_cancer = population - has_cancer

        tp = sens * has_cancer  # True positives
        fn = (1 - sens) * has_cancer  # False negatives
        fp = (1 - spec) * no_cancer  # False positives
        tn = spec * no_cancer  # True negatives

        positive = tp + fp  # Total positive tests
        negative = fn + tn  # Total negative tests

        biopsy = positive * adjusted_biopsy_rate
        complication = biopsy * comp_rate

        # Personal probability calculations
        your_cancer_risk = overall_prevalence
        your_positive_test_chance = positive / population if population > 0 else 0
        your_cancer_detection_chance = tp / population if population > 0 else 0
        your_false_positive_chance = fp / population if population > 0 else 0
        your_biopsy_chance = biopsy / population if population > 0 else 0
        your_complication_chance = complication / population if population > 0 else 0
        
        # Screening impact comparison
        st.subheader("🔄 Your Personal Screening Impact")
        st.markdown("**How does screening change YOUR outcomes?**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Your Probability Changes**")
            
            # Create individual probability comparison chart
            scenarios = ['Cancer Risk', 'Cancer Detected', 'False Positive', 'Biopsy Needed', 'Complication']
            no_screening = [your_cancer_risk * 100, 0, 0, 0, 0]
            with_screening = [
                your_cancer_risk * 100,
                your_cancer_detection_chance * 100,
                your_false_positive_chance * 100,
                your_biopsy_chance * 100,
                your_complication_chance * 100
            ]
            
            fig_individual = go.Figure()
            
            fig_individual.add_trace(go.Bar(
                name='Without Screening',
                x=scenarios,
                y=no_screening,
                marker_color='#E74C3C',
                text=[f'{y:.1f}%' if y > 0.1 else f'{y:.2f}%' for y in no_screening],
                textposition='auto',
                hovertemplate='Without Screening<br>%{y:.2f}%<extra></extra>'
            ))
            
            fig_individual.add_trace(go.Bar(
                name='With Screening',
                x=scenarios,
                y=with_screening,
                marker_color='#27AE60',
                text=[f'{y:.1f}%' if y > 0.1 else f'{y:.2f}%' for y in with_screening],
                textposition='auto',
                hovertemplate='With Screening<br>%{y:.2f}%<extra></extra>'
            ))
            
            fig_individual.update_layout(
                title='Your Personal Probabilities (%)',
                xaxis_title='',
                yaxis_title='Probability (%)',
                barmode='group',
                height=350,
                showlegend=True,
                margin=dict(l=50, r=50, t=60, b=50),
                font=dict(size=10)
            )
            
            fig_individual.update_xaxis(tickangle=45)
            
            st.plotly_chart(fig_individual, use_container_width=True)
        
        with col2:
            st.markdown("**🎯 Your Personal Benefits & Risks**")
            
            # Individual benefits
            st.markdown("**✅ Your Benefits from Screening:**")
            if your_cancer_detection_chance > 0.001:
                st.write(f"• **{your_cancer_detection_chance*100:.2f}%** chance of early cancer detection")
                
                # Estimated mortality benefit
                mortality_reduction = your_cancer_detection_chance * 0.3  # 30% mortality reduction
                st.write(f"• **{mortality_reduction*100:.2f}%** reduction in cancer death risk")
            else:
                st.write(f"• **Very low** chance of finding cancer ({your_cancer_detection_chance*100:.3f}%)")
            
            detection_efficiency = (your_cancer_detection_chance / your_cancer_risk * 100) if your_cancer_risk > 0 else 0
            st.write(f"• **{detection_efficiency:.0f}%** of your cancers would be caught early")
            
            # Individual risks
            st.markdown("**⚠️ Your Risks from Screening:**")
            st.write(f"• **{your_false_positive_chance*100:.1f}%** chance of false positive")
            st.write(f"• **{your_biopsy_chance*100:.1f}%** chance of needing biopsy")
            if your_complication_chance > 0.001:
                st.write(f"• **{your_complication_chance*100:.2f}%** chance of biopsy complication")
            else:
                st.write(f"• **<0.01%** chance of biopsy complication")
            
            # Personal net benefit
            st.markdown("**📈 Your Net Benefit:**")
            benefit_score = your_cancer_detection_chance * 100
            harm_score = your_false_positive_chance * 10 + your_complication_chance * 50
            net_benefit_individual = benefit_score - harm_score
            
            if net_benefit_individual > 0.1:
                st.success(f"**Positive (+{net_benefit_individual:.2f})** - Likely beneficial for you")
            elif net_benefit_individual > -0.1:
                st.info(f"**Neutral ({net_benefit_individual:.2f})** - Marginal benefit")
            else:
                st.warning(f"**Negative ({net_benefit_individual:.2f})** - Limited benefit for your risk profile")
            
            # Personalized recommendation
            st.markdown("**🎯 For Your Risk Profile:**")
            if your_cancer_risk > 0.1:  # High risk (>10%)
                if net_benefit_individual > 0.1:
                    st.success("✅ **Screening strongly recommended** - High risk with clear benefit")
                else:
                    st.info("🟡 **Discuss with doctor** - High risk but consider test limitations")
            elif your_cancer_risk > 0.05:  # Moderate risk (5-10%)
                if net_benefit_individual > 0.05:
                    st.success("✅ **Screening recommended** - Moderate risk with good benefit")
                else:
                    st.info("🟡 **Consider screening** - Weigh personal preferences")
            else:  # Low risk (<5%)
                if net_benefit_individual > 0.02:
                    st.info("🟡 **Screening may be worthwhile** - Low risk but some benefit")
                else:
                    st.warning("⚠️ **Limited benefit** - Consider if convenience/cost worth it")

        # Detailed personal metrics
        with st.expander("📊 Your Detailed Personal Metrics", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Your Risk Profile:**")
                st.write(f"• 10-year cancer risk: **{your_cancer_risk*100:.1f}%**")
                st.write(f"• Risk level: **{risk_level}**")
                baseline_risk = calculate_overall_prevalence_fixed(age, sex, None)[0]
                risk_multiplier_overall = your_cancer_risk / baseline_risk if baseline_risk > 0 else 1
                st.write(f"• Risk vs average: **{risk_multiplier_overall:.1f}x**")
            
            with col2:
                st.markdown("**Your Test Chances:**")
                st.write(f"• Positive test: **{your_positive_test_chance*100:.1f}%**")
                if your_positive_test_chance > 0:
                    your_ppv = your_cancer_detection_chance / your_positive_test_chance if your_positive_test_chance > 0 else 0
                    st.write(f"• If positive, cancer probability: **{your_ppv*100:.0f}%**")
                
                your_negative_test_chance = negative / population if population > 0 else 0
                if your_negative_test_chance > 0:
                    your_npv = (tn/population) / your_negative_test_chance if your_negative_test_chance > 0 else 0
                    st.write(f"• If negative, no cancer probability: **{your_npv*100:.1f}%**")
            
            with col3:
                st.markdown("**Your Outcomes:**")
                if your_cancer_risk > 0:
                    detection_rate = (your_cancer_detection_chance / your_cancer_risk * 100)
                    st.write(f"• Cancer detection rate: **{detection_rate:.0f}%**")
                    
                    miss_rate = 100 - detection_rate
                    st.write(f"• Cancer miss rate: **{miss_rate:.0f}%**")
                
                # Cost estimate
                cost_per_test = {"Mammography": 150, "Colonoscopy": 800, "Low-dose CT Scan": 300, 
                               "PSA Test": 50, "Whole-body MRI": 2000, "Galleri Blood Test": 1000}
                avg_test_cost = np.mean([cost_per_test.get(test, 500) for test in tests])
                expected_cost = avg_test_cost + (your_biopsy_chance * 1500)
                st.write(f"• Expected cost: **${expected_cost:.0f}**")

        st.markdown("---")

    else:
        # No tests selected - show baseline risk
        st.header("Baseline Risk (No Screening)")
        
        has_cancer = overall_prevalence * population
        no_cancer = population - has_cancer

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
                y=[0.5, 0.3, 0.7],
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
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Per-cancer risk breakdown
    st.header("Risk Breakdown by Cancer Type")
    per_cancer_df = calculate_per_cancer_prevalence_fixed(age, sex, risk_multipliers)
    
    if not per_cancer_df.empty:
        # Sort by personalized risk and show top 10
        per_cancer_df_sorted = per_cancer_df.sort_values("Personalized Risk (%)", ascending=False).head(10)
        
        # Format the dataframe for better display
        per_cancer_df_sorted = per_cancer_df_sorted.copy()
        per_cancer_df_sorted["Base Risk (%)"] = per_cancer_df_sorted["Base Risk (%)"].round(3)
        per_cancer_df_sorted["Risk Multiplier"] = per_cancer_df_sorted["Risk Multiplier"].round(1)
        per_cancer_df_sorted["Personalized Risk (%)"] = per_cancer_df_sorted["Personalized Risk (%)"].round(2)
        
        st.dataframe(
            per_cancer_df_sorted,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No cancer risk data available for your demographic profile.")
    
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