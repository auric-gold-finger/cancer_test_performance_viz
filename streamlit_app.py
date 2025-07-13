import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configure the app
st.set_page_config(
    page_title="Cancer Screening Test Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("Cancer Screening Test Analysis")
st.markdown("""
This tool helps you understand how cancer screening tests might affect your personal cancer risk. 
Enter your information on the left, and we'll show simple takeaways about benefits and risks.
**Important:** This is for educational purposes only. Talk to your doctor for medical advice.
Data updated as of July 2025.
""")

# Updated with latest available data as of July 2025
TEST_PERFORMANCE = {
    "Whole-body MRI": {
        "lung": {"sensitivity": 0.50, "specificity": 0.93},
        "breast": {"sensitivity": 0.95, "specificity": 0.74},
        "colorectal": {"sensitivity": 0.67, "specificity": 0.95},
        "prostate": {"sensitivity": 0.84, "specificity": 0.89},
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
    "Grail Blood Test": {
        "lung": {"sensitivity": 0.74, "specificity": 0.995},
        "breast": {"sensitivity": 0.34, "specificity": 0.995},
        "colorectal": {"sensitivity": 0.83, "specificity": 0.995},
        "prostate": {"sensitivity": 0.16, "specificity": 0.995},
        "liver": {"sensitivity": 0.88, "specificity": 0.995},
        "pancreatic": {"sensitivity": 0.75, "specificity": 0.995},
        "ovarian": {"sensitivity": 0.90, "specificity": 0.995},
        "kidney": {"sensitivity": 0.46, "specificity": 0.995},
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

# Downstream testing and complication risks - Updated with latest estimates
DOWNSTREAM_RISKS = {
    "Whole-body MRI": {
        "false_positive_rate": 8.0,  # Updated from recent studies (e.g., 8% FPR in CPS cohorts)
        "biopsy_rate_fp": 0.5,  # Approx 50% of FPs lead to biopsy
        "comp_rate_biopsy": 0.03,  # 3% complication rate per biopsy
        "typical_followup": "Additional MRI with contrast, possible biopsy",
        "followup_complications": 2.1,  # Contrast reactions, biopsy complications
        "psychological_impact": "Moderate - incidental findings cause anxiety",
        "radiation_exposure": "None from MRI, possible CT follow-up"
    },
    "Grail Blood Test": {
        "false_positive_rate": 0.5,
        "biopsy_rate_fp": 0.5,
        "comp_rate_biopsy": 0.03,
        "typical_followup": "Imaging scans (CT, MRI, PET), possible biopsy",
        "followup_complications": 3.8,  # Multiple imaging, biopsy risks
        "psychological_impact": "High - positive blood test causes significant anxiety",
        "radiation_exposure": "Moderate to high from follow-up CT/PET scans"
    },
    "CT Scan": {
        "false_positive_rate": 4.8,  # Updated from recent lung screening studies
        "biopsy_rate_fp": 0.5,
        "comp_rate_biopsy": 0.03,
        "typical_followup": "Repeat CT, additional imaging, possible biopsy",
        "followup_complications": 4.2,  # Additional radiation, biopsy complications
        "psychological_impact": "Moderate to high - abnormal findings cause worry",
        "radiation_exposure": "Additional radiation from repeat scans"
    }
}

# Real US cancer incidence rates per 100,000 (SEER/CDC 2018-2022 data, no major 2025 changes) - EXPANDED
CANCER_INCIDENCE = {
    "lung": {
        "male": {40: 8, 50: 25, 60: 85, 70: 180, 80: 220},
        "female": {40: 12, 50: 30, 60: 70, 70: 130, 80: 160}
    },
    "breast": {
        "male": {40: 1, 50: 2, 60: 3, 70: 4, 80: 5},
        "female": {40: 50, 50: 130, 60: 200, 70: 250, 80: 280}
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

def calculate_ppv_npv(sensitivity, specificity, prevalence):
    """Calculate test accuracy metrics"""
    if prevalence == 0:
        return 0, 1
    ppv = (sensitivity * prevalence) / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence)) if (sensitivity * prevalence + (1 - specificity) * (1 - prevalence)) > 0 else 0
    npv = (specificity * (1 - prevalence)) / ((1 - sensitivity) * prevalence + specificity * (1 - prevalence)) if ((1 - sensitivity) * prevalence + specificity * (1 - prevalence)) > 0 else 1
    return ppv, npv

def calculate_post_test_risk_negative(sensitivity, specificity, prevalence):
    """Calculate probability you still have cancer after a negative test - FIXED"""
    if prevalence == 0:
        return 0
    return ((1 - sensitivity) * prevalence) / ((1 - sensitivity) * prevalence + specificity * (1 - prevalence)) if ((1 - sensitivity) * prevalence + specificity * (1 - prevalence)) > 0 else 0

def get_prevalence_from_incidence(incidence_rate):
    """Convert yearly cancer rate to current prevalence"""
    return (incidence_rate / 100000) * 5

def interpolate_incidence(age, sex, cancer_type):
    """Get cancer risk for specific age"""
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
    """Calculate risk multiplier based on risk factors"""
    multiplier = 1.0
    
    # Smoking effects (based on epidemiological studies)
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
    
    # Family history effects
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
            multiplier *= 2.3  # First-degree relative
        elif cancer_type == "colorectal":
            multiplier *= 2.2
        elif cancer_type == "prostate":
            multiplier *= 2.5
        elif cancer_type == "ovarian":
            multiplier *= 3.1
        elif cancer_type in ["lung", "pancreatic"]:
            multiplier *= 1.8
    
    # Genetic mutation effects
    if "BRCA1" in genetic_mutations:
        if cancer_type == "breast":
            multiplier *= 35  # Lifetime risk ~72%
        elif cancer_type == "ovarian":
            multiplier *= 20  # Lifetime risk ~44%
    
    if "BRCA2" in genetic_mutations:
        if cancer_type == "breast":
            multiplier *= 20  # Lifetime risk ~69%
        elif cancer_type == "ovarian":
            multiplier *= 8   # Lifetime risk ~17%
        elif cancer_type == "prostate":
            multiplier *= 4.5
    
    if "Lynch syndrome" in genetic_mutations:
        if cancer_type == "colorectal":
            multiplier *= 15  # Lifetime risk ~80%
        elif cancer_type == "ovarian":
            multiplier *= 6
        elif cancer_type == "endometrial":
            multiplier *= 12
    
    if "TP53 (Li-Fraumeni)" in genetic_mutations:
        # Li-Fraumeni increases risk for many cancers
        if cancer_type in ["breast", "lung", "colorectal", "liver", "brain", "sarcoma"]:
            multiplier *= 10
    
    # Personal cancer history (increases risk of second cancers)
    if personal_history:
        multiplier *= 2.5  # General increased risk for second cancers
    
    return min(multiplier, 100)  # Cap at 100x to avoid unrealistic values

def calculate_overall_cancer_prevalence(age, sex, risk_multipliers=None):
    """Calculate overall cancer prevalence for age group"""
    total_prevalence = 0
    
    for cancer_type in CANCER_INCIDENCE.keys():
        if cancer_type in ["prostate", "testicular"] and sex == "female":
            continue
        if cancer_type in ["ovarian", "cervical", "endometrial", "uterine"] and sex == "male":
            continue
            
        incidence_rate = interpolate_incidence(age, sex, cancer_type)
        prevalence = get_prevalence_from_incidence(incidence_rate)
        
        if risk_multipliers and cancer_type in risk_multipliers:
            prevalence *= risk_multipliers[cancer_type]
        
        total_prevalence += prevalence
    
    return total_prevalence

# Sidebar inputs
with st.sidebar:
    st.header("Your Information")
    st.markdown("Fill in these details to personalize your results.")

    age = st.slider("Age", min_value=30, max_value=90, value=55, step=1, help="Your current age.")
    sex = st.selectbox("Sex", ["male", "female"], help="Biological sex for risk calculation.")

    test_type = st.selectbox(
        "Screening Test",
        ["Whole-body MRI", "Grail Blood Test", "CT Scan"],
        help="Choose the test you're interested in."
    )

    st.header("Risk Factors")
    st.markdown("These can increase your cancer risk. Select what applies to you.")

    smoking_status = st.selectbox(
        "Smoking Status",
        ["Never smoked", "Former smoker", "Current smoker"],
        help="Smoking raises risk for many cancers, especially lung."
    )

    if smoking_status in ["Former smoker", "Current smoker"]:
        pack_years = st.slider(
            "Pack-years of smoking",
            min_value=0, max_value=80, value=20, step=5,
            help="Packs per day × years smoked (e.g., 1 pack/day for 20 years = 20)."
        )
    else:
        pack_years = 0

    family_history = st.multiselect(
        "Family History (parents, siblings, children)",
        ["Breast cancer", "Colorectal cancer", "Prostate cancer", "Ovarian cancer", 
         "Lung cancer", "Pancreatic cancer"],
        help="Close relatives with these cancers may increase your risk."
    )

    genetic_mutations = st.multiselect(
        "Known Genetic Mutations (from testing)",
        ["BRCA1", "BRCA2", "Lynch syndrome", "TP53 (Li-Fraumeni)"],
        help="Only select if confirmed by a genetic test."
    )

    personal_history = st.checkbox(
        "I have had cancer before",
        help="Previous cancer can raise risk of new ones."
    )

    use_custom_probability = st.checkbox("Use my own risk estimate", help="Override calculations with your doctor's estimate.")

    if use_custom_probability:
        custom_probability = st.slider(
            "My estimated cancer risk (%)", 
            min_value=0.1, max_value=50.0, value=5.0, step=0.1,
            help="Enter a percentage if you have a custom estimate."
        )

# Calculate personalized risk multipliers
cancer_types = list(TEST_PERFORMANCE[test_type].keys())

risk_multipliers = {}
for cancer_type in cancer_types:
    if cancer_type in ["prostate", "testicular"] and sex == "female":
        continue
    if cancer_type in ["ovarian", "cervical", "endometrial", "uterine"] and sex == "male":
        continue
    risk_multipliers[cancer_type] = get_risk_multiplier(
        cancer_type, smoking_status, pack_years, family_history, 
        genetic_mutations, personal_history
    )

# Calculate overall cancer prevalence for context
baseline_overall_prevalence = calculate_overall_cancer_prevalence(age, sex)
personalized_overall_prevalence = calculate_overall_cancer_prevalence(age, sex, risk_multipliers)

# Calculate results
results = []
overall_tp = 0
overall_fp = 0
overall_fn = 0
overall_tn = 0
overall_pre_test_risk = 0
overall_post_test_risk = 0

for cancer_type in cancer_types:
    if cancer_type in ["prostate", "testicular"] and sex == "female":
        continue
    if cancer_type in ["ovarian", "cervical", "endometrial", "uterine"] and sex == "male":
        continue
        
    sensitivity = TEST_PERFORMANCE[test_type][cancer_type]["sensitivity"]
    specificity = TEST_PERFORMANCE[test_type][cancer_type]["specificity"]
    
    if use_custom_probability:
        prevalence = custom_probability / 100
    else:
        incidence_rate = interpolate_incidence(age, sex, cancer_type)
        baseline_prevalence = get_prevalence_from_incidence(incidence_rate)
        prevalence = baseline_prevalence * risk_multipliers[cancer_type]
    
    ppv, npv = calculate_ppv_npv(sensitivity, specificity, prevalence)
    post_test_risk = calculate_post_test_risk_negative(sensitivity, specificity, prevalence)
    false_positive_risk = (1 - specificity) * (1 - prevalence)
    
    # Calculate baseline risk for comparison
    baseline_incidence = interpolate_incidence(age, sex, cancer_type)
    baseline_risk = get_prevalence_from_incidence(baseline_incidence)
    
    abs_risk_reduction = prevalence - post_test_risk
    
    results.append({
        "Cancer Type": cancer_type.replace("_", " ").title(),
        "Baseline Risk": round(baseline_risk * 100, 3),
        "Your Risk": round(prevalence * 100, 3),
        "Risk Multiplier": round(risk_multipliers[cancer_type], 1),
        "Post-test Risk (if negative)": round(post_test_risk * 100, 4),
        "Abs Risk Reduction": round(abs_risk_reduction * 100, 4),
        "Rel Risk Reduction": round(((prevalence - post_test_risk) / prevalence) * 100, 1) if prevalence > 0 else 0,
        "False Positive Risk": round(false_positive_risk * 100, 2),
        "Detection Rate": round(sensitivity * 100, 1),
        "Accuracy Rate": round(specificity * 100, 1),
        "Positive Accuracy": round(ppv * 100, 1),
        "Negative Accuracy": round(npv * 100, 1)
    })
    
    # Accumulate for overall
    tp = sensitivity * prevalence
    fp = (1 - specificity) * (1 - prevalence)
    fn = (1 - sensitivity) * prevalence
    tn = specificity * (1 - prevalence)
    overall_tp += tp
    overall_fp += fp
    overall_fn += fn
    overall_tn += tn
    overall_pre_test_risk += prevalence
    overall_post_test_risk += post_test_risk

df = pd.DataFrame(results)

# Overall outcomes for pie
overall_total = overall_tp + overall_fp + overall_fn + overall_tn
if overall_total > 0:
    pie_values = [overall_tp / overall_total * 100, overall_fp / overall_total * 100, overall_fn / overall_total * 100, overall_tn / overall_total * 100]
else:
    pie_values = [0, 0, 0, 100]

# Calculate overall metrics
overall_abs_reduction = (overall_pre_test_risk - overall_post_test_risk) * 100
overall_rel_reduction = ((overall_pre_test_risk - overall_post_test_risk) / overall_pre_test_risk * 100) if overall_pre_test_risk > 0 else 0
overall_fp_risk = overall_fp * 100  # Approximate overall FP %

# Key Takeaways Section
st.header("Key Takeaways")
st.markdown("Here's a quick summary of what this means for you.")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Your Overall Cancer Risk", f"{personalized_overall_prevalence*100:.2f}%", help="This is your estimated chance of having cancer now, based on your info.")
    st.metric("Risk After Negative Test", f"{overall_post_test_risk*100:.2f}%", help="If the test is negative, your risk drops to this.")

with col2:
    st.metric("Benefit: Risk Reduction", f"{overall_abs_reduction:.2f}% points", delta=f"{overall_rel_reduction:.1f}% relative", help="How much a negative test lowers your risk.")
    
with col3:
    risk_data = DOWNSTREAM_RISKS[test_type]
    expected_biopsies = risk_data['false_positive_rate'] * risk_data['biopsy_rate_fp']
    expected_comps = expected_biopsies * risk_data['comp_rate_biopsy']
    st.metric("Risk: False Positive Chance", f"{risk_data['false_positive_rate']}%", help="Chance of a wrong 'positive' result leading to more tests.")
    st.metric("Possible Biopsies/Complications", f"{expected_biopsies:.2f}% / {expected_comps:.4f}%", help="Estimated chance of needing a biopsy or having a complication from follow-up.")

st.markdown("""
- **Benefit:** A negative test can give you peace of mind by lowering your estimated cancer risk.
- **Tradeoff:** Positive results might lead to extra tests or biopsies, even if you don't have cancer (false positive).
- **Next Steps:** If your risk is high, discuss screening with your doctor. Tests aren't perfect and can sometimes find slow-growing cancers that don't need treatment.
""")

# Risk factors summary
if any([smoking_status != "Never smoked", family_history, genetic_mutations, personal_history]):
    with st.expander("Your Risk Factors (Click to expand)"):
        risk_factors_list = []
        
        if smoking_status == "Current smoker":
            risk_factors_list.append(f"Current smoker ({pack_years} pack-years)")
        elif smoking_status == "Former smoker":
            risk_factors_list.append(f"Former smoker ({pack_years} pack-years)")
        
        if family_history:
            risk_factors_list.append(f"Family history: {', '.join(family_history)}")
        
        if genetic_mutations:
            risk_factors_list.append(f"Genetic mutations: {', '.join(genetic_mutations)}")
        
        if personal_history:
            risk_factors_list.append("Personal cancer history")
        
        for factor in risk_factors_list:
            st.write(f"• {factor}")
else:
    st.info("You have no major risk factors selected. Your risks are based on average for your age and sex.")

# Visual Sections with Expanders
with st.expander("See How Your Risk Changes with a Negative Test"):
    st.subheader("Cancer Risk Before vs. After Negative Test")
    st.markdown("This chart shows your current risk (light red) and risk after a negative test (dark red) for different cancers.")

    fig_comparison = go.Figure()

    fig_comparison.add_trace(go.Bar(
        name='Current Risk',
        x=df["Cancer Type"],
        y=df["Your Risk"],
        marker_color='lightcoral',
        opacity=0.8,
        hovertemplate="<b>%{x}</b><br>Current risk: %{y}%"
    ))

    fig_comparison.add_trace(go.Bar(
        name='After Negative Test',
        x=df["Cancer Type"],
        y=df["Post-test Risk (if negative)"],
        marker_color='darkred',
        opacity=0.8,
        hovertemplate="<b>%{x}</b><br>After negative: %{y}%"
    ))

    fig_comparison.update_layout(
        xaxis_title='Cancer Type',
        yaxis_title='Risk (%)',
        yaxis=dict(range=[0, max(df["Your Risk"].max(), 100)]),
        barmode='group',
        height=400
    )

    st.plotly_chart(fig_comparison, use_container_width=True)

with st.expander("See Expected Test Outcomes"):
    st.subheader("What Might Happen with the Test")
    st.markdown("This pie shows the likely results: correct detections, false alarms, missed cancers, or correct all-clear.")

    fig_pie = go.Figure(data=[go.Pie(
        labels=['Correct Detection (True Positive)', 'False Alarm (False Positive)', 'Missed Cancer (False Negative)', 'Correct All-Clear (True Negative)'],
        values=pie_values,
        hole=.3,
        marker_colors=['#FF9999', '#FF6666', '#CC0000', '#66CC66']
    )])
    st.plotly_chart(fig_pie, use_container_width=True)

with st.expander("Detailed Results Table"):
    st.subheader("Breakdown by Cancer Type")
    st.markdown("""
    - **Your Risk:** Estimated chance of this cancer now.
    - **After Negative:** Risk if test is negative.
    - **Risk Reduction:** How much risk drops (absolute points and relative %).
    - **False Positive Risk:** Chance of wrong positive for this cancer.
    """)
    simplified_df = df[["Cancer Type", "Your Risk", "Post-test Risk (if negative)", "Abs Risk Reduction", "Rel Risk Reduction", "False Positive Risk"]]
    st.dataframe(
        simplified_df.style.format({
            "Your Risk": "{:.3f}%",
            "Post-test Risk (if negative)": "{:.4f}%",
            "Abs Risk Reduction": "{:.4f}%",
            "Rel Risk Reduction": "{:.1f}%",
            "False Positive Risk": "{:.2f}%"
        })
    )

with st.expander("Risks of Follow-Up Tests"):
    st.subheader("What If the Test is Positive?")
    st.markdown("Positive results often lead to more tests. Here's what could happen.")

    risk_data = DOWNSTREAM_RISKS[test_type]
    expected_biopsies = risk_data['false_positive_rate'] * risk_data['biopsy_rate_fp']
    expected_comps = expected_biopsies * risk_data['comp_rate_biopsy']

    st.markdown(f"**Chance of False Positive:** {risk_data['false_positive_rate']}% (test says positive, but no cancer)")
    st.markdown(f"**Chance of Needing a Biopsy:** {expected_biopsies:.2f}% overall")
    st.markdown(f"**Chance of Complications (e.g., infection):** {expected_comps:.4f}% overall")
    st.markdown(f"**Typical Next Steps:** {risk_data['typical_followup']}")
    st.markdown(f"**Other Risks:** {risk_data['psychological_impact']}; {risk_data['radiation_exposure']}")

    fig_risks = go.Figure()
    risk_types = ['False Positive', 'Biopsy Needed', 'Complications']
    risk_values = [risk_data['false_positive_rate'], expected_biopsies, expected_comps]

    fig_risks.add_trace(go.Bar(
        x=risk_types,
        y=risk_values,
        marker_color='crimson',
        text=[f"{v:.2f}%" for v in risk_values],
        textposition='auto'
    ))

    fig_risks.update_layout(
        yaxis_title='Chance (%)',
        height=300
    )

    st.plotly_chart(fig_risks, use_container_width=True)