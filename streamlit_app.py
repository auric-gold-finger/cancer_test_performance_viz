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
st.markdown("Compare test performance and understand what results mean for your cancer risk")

# Real clinical data from recent studies and trials
TEST_PERFORMANCE = {
    "Whole-body MRI": {
        "lung": {"sensitivity": 0.50, "specificity": 0.93},
        "breast": {"sensitivity": 0.95, "specificity": 0.74},
        "colorectal": {"sensitivity": 0.67, "specificity": 0.95},
        "prostate": {"sensitivity": 0.84, "specificity": 0.89},
        "liver": {"sensitivity": 0.84, "specificity": 0.94},
        "pancreatic": {"sensitivity": 0.75, "specificity": 0.85},
        "ovarian": {"sensitivity": 0.98, "specificity": 0.90},
        "kidney": {"sensitivity": 0.85, "specificity": 0.90}
    },
    "Grail Blood Test": {
        "lung": {"sensitivity": 0.74, "specificity": 0.995},
        "breast": {"sensitivity": 0.34, "specificity": 0.995},
        "colorectal": {"sensitivity": 0.83, "specificity": 0.995},
        "prostate": {"sensitivity": 0.16, "specificity": 0.995},
        "liver": {"sensitivity": 0.88, "specificity": 0.995},
        "pancreatic": {"sensitivity": 0.75, "specificity": 0.995},
        "ovarian": {"sensitivity": 0.90, "specificity": 0.995},
        "kidney": {"sensitivity": 0.46, "specificity": 0.995}
    },
    "CT Scan": {
        "lung": {"sensitivity": 0.93, "specificity": 0.77},
        "breast": {"sensitivity": 0.70, "specificity": 0.85},
        "colorectal": {"sensitivity": 0.96, "specificity": 0.80},
        "prostate": {"sensitivity": 0.75, "specificity": 0.80},
        "liver": {"sensitivity": 0.68, "specificity": 0.93},
        "pancreatic": {"sensitivity": 0.84, "specificity": 0.67},
        "ovarian": {"sensitivity": 0.83, "specificity": 0.87},
        "kidney": {"sensitivity": 0.85, "specificity": 0.89}
    }
}

# Downstream testing and complication risks
DOWNSTREAM_RISKS = {
    "Whole-body MRI": {
        "false_positive_rate": 8.5,  # Average across cancer types
        "typical_followup": "Additional MRI with contrast, possible biopsy",
        "followup_complications": 2.1,  # Contrast reactions, biopsy complications
        "psychological_impact": "Moderate - incidental findings cause anxiety",
        "radiation_exposure": "None from MRI, possible CT follow-up"
    },
    "Grail Blood Test": {
        "false_positive_rate": 0.5,
        "typical_followup": "Imaging scans (CT, MRI, PET), possible biopsy",
        "followup_complications": 3.8,  # Multiple imaging, biopsy risks
        "psychological_impact": "High - positive blood test causes significant anxiety",
        "radiation_exposure": "Moderate to high from follow-up CT/PET scans"
    },
    "CT Scan": {
        "false_positive_rate": 23.3,  # Average across applications
        "typical_followup": "Repeat CT, additional imaging, possible biopsy",
        "followup_complications": 4.2,  # Additional radiation, biopsy complications
        "psychological_impact": "Moderate to high - abnormal findings cause worry",
        "radiation_exposure": "Additional radiation from repeat scans"
    }
}

# Real US cancer incidence rates per 100,000 (SEER/CDC 2018-2022 data)
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
    }
}

def calculate_ppv_npv(sensitivity, specificity, prevalence):
    """Calculate test accuracy metrics"""
    ppv = (sensitivity * prevalence) / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence))
    npv = (specificity * (1 - prevalence)) / ((1 - sensitivity) * prevalence + specificity * (1 - prevalence))
    return ppv, npv

def calculate_post_test_risk_negative(sensitivity, prevalence):
    """Calculate probability you still have cancer after a negative test"""
    return ((1 - sensitivity) * prevalence) / ((1 - sensitivity) * prevalence + 1 - prevalence)

def get_prevalence_from_incidence(incidence_rate):
    """Convert yearly cancer rate to current prevalence"""
    return (incidence_rate / 100000) * 5

def interpolate_incidence(age, sex, cancer_type):
    """Get cancer risk for specific age"""
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
        elif cancer_type in ["bladder", "kidney", "pancreatic"]:
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
        elif cancer_type in ["bladder", "kidney", "pancreatic"]:
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
    
    if "TP53 (Li-Fraumeni)" in genetic_mutations:
        # Li-Fraumeni increases risk for many cancers
        if cancer_type in ["breast", "lung", "colorectal", "liver"]:
            multiplier *= 10
    
    # Personal cancer history (increases risk of second cancers)
    if personal_history:
        multiplier *= 2.5  # General increased risk for second cancers
    
    return min(multiplier, 100)  # Cap at 100x to avoid unrealistic values

def calculate_overall_cancer_prevalence(age, sex, risk_multipliers=None):
    """Calculate overall cancer prevalence for age group"""
    total_prevalence = 0
    
    for cancer_type in CANCER_INCIDENCE.keys():
        if cancer_type == "prostate" and sex == "female":
            continue
        if cancer_type == "ovarian" and sex == "male":
            continue
            
        incidence_rate = interpolate_incidence(age, sex, cancer_type)
        prevalence = get_prevalence_from_incidence(incidence_rate)
        
        if risk_multipliers and cancer_type in risk_multipliers:
            prevalence *= risk_multipliers[cancer_type]
        
        total_prevalence += prevalence
    
    return total_prevalence

# Sidebar inputs
st.sidebar.header("Basic Information")

age = st.sidebar.slider("Age", min_value=30, max_value=90, value=55, step=1)
sex = st.sidebar.selectbox("Sex", ["male", "female"])

test_type = st.sidebar.selectbox(
    "Screening Test",
    ["Whole-body MRI", "Grail Blood Test", "CT Scan"]
)

# Risk factors section
st.sidebar.header("Risk Factors")
st.sidebar.markdown("*These significantly affect your cancer risk*")

# Smoking history
smoking_status = st.sidebar.selectbox(
    "Smoking Status",
    ["Never smoked", "Former smoker", "Current smoker"],
    help="Smoking significantly increases risk for lung, bladder, and other cancers"
)

if smoking_status in ["Former smoker", "Current smoker"]:
    pack_years = st.sidebar.slider(
        "Pack-years of smoking",
        min_value=0, max_value=80, value=20, step=5,
        help="Packs per day × years smoked (e.g., 1 pack/day for 20 years = 20 pack-years)"
    )
else:
    pack_years = 0

# Family history
family_history = st.sidebar.multiselect(
    "Family History (first-degree relatives)",
    ["Breast cancer", "Colorectal cancer", "Prostate cancer", "Ovarian cancer", 
     "Lung cancer", "Pancreatic cancer"],
    help="Parents, siblings, or children with these cancers"
)

# Genetic mutations
genetic_mutations = st.sidebar.multiselect(
    "Known Genetic Mutations",
    ["BRCA1", "BRCA2", "Lynch syndrome", "TP53 (Li-Fraumeni)"],
    help="Only select if confirmed by genetic testing"
)

# Personal cancer history
personal_history = st.sidebar.checkbox(
    "Personal history of cancer",
    help="Previous cancer diagnosis increases risk of recurrence and second cancers"
)

use_custom_probability = st.sidebar.checkbox("Override with custom risk estimate")

if use_custom_probability:
    custom_probability = st.sidebar.slider(
        "Custom cancer risk (%)", 
        min_value=0.1, max_value=50.0, value=5.0, step=0.1
    )

# Calculate personalized risk multipliers
cancer_types = list(TEST_PERFORMANCE[test_type].keys())

risk_multipliers = {}
for cancer_type in cancer_types:
    if cancer_type == "prostate" and sex == "female":
        continue
    if cancer_type == "ovarian" and sex == "male":
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

for cancer_type in cancer_types:
    if cancer_type == "prostate" and sex == "female":
        continue
    if cancer_type == "ovarian" and sex == "male":
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
    post_test_risk = calculate_post_test_risk_negative(sensitivity, prevalence)
    false_positive_risk = (1 - specificity) * (1 - prevalence)
    
    # Calculate baseline risk for comparison
    baseline_incidence = interpolate_incidence(age, sex, cancer_type)
    baseline_risk = get_prevalence_from_incidence(baseline_incidence)
    
    results.append({
        "Cancer Type": cancer_type.replace("_", " ").title(),
        "Baseline Risk": round(baseline_risk * 100, 3),
        "Your Risk": round(prevalence * 100, 3),
        "Risk Multiplier": round(risk_multipliers[cancer_type], 1),
        "Post-test Risk (if negative)": round(post_test_risk * 100, 4),
        "Risk Reduction": round(((prevalence - post_test_risk) / prevalence) * 100, 1),
        "False Positive Risk": round(false_positive_risk * 100, 2),
        "Detection Rate": round(sensitivity * 100, 1),
        "Accuracy Rate": round(specificity * 100, 1),
        "Positive Accuracy": round(ppv * 100, 1),
        "Negative Accuracy": round(npv * 100, 1)
    })

df = pd.DataFrame(results)

# Main content
st.subheader(f"Test Performance: {test_type}")

# Overall cancer risk context
st.subheader("Your Overall Cancer Risk Profile")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Average Risk (Your Age/Sex)", 
        f"{baseline_overall_prevalence*100:.2f}%",
        help=f"Typical cancer risk for {sex}s aged {age}"
    )

with col2:
    st.metric(
        "Your Personalized Risk", 
        f"{personalized_overall_prevalence*100:.2f}%",
        delta=f"{((personalized_overall_prevalence/baseline_overall_prevalence)-1)*100:+.0f}%",
        help="Your risk considering smoking, family history, and genetic factors"
    )

with col3:
    risk_category = "Low" if personalized_overall_prevalence < 0.02 else "Moderate" if personalized_overall_prevalence < 0.05 else "High"
    st.metric(
        "Risk Category", 
        risk_category,
        help="Based on your personalized risk factors"
    )

# Risk factors summary
if any([smoking_status != "Never smoked", family_history, genetic_mutations, personal_history]):
    st.markdown("**Your Risk Factors:**")
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
    st.info("You have reported no major cancer risk factors")

# Risk comparison section
st.subheader("Cancer Risk: Before Testing vs After Negative Test")
st.markdown("*This shows how much a negative test reduces your cancer probability*")

# Create side-by-side comparison chart
fig_comparison = go.Figure()

fig_comparison.add_trace(go.Bar(
    name='Your Current Cancer Risk',
    x=df["Cancer Type"],
    y=df["Your Risk"],
    marker_color='lightcoral',
    opacity=0.8,
    hovertemplate="<b>%{x}</b><br>Your current risk: %{y}%<br>Risk multiplier: " + df["Risk Multiplier"].astype(str) + "x<extra></extra>"
))

fig_comparison.add_trace(go.Bar(
    name='Risk After Negative Test',
    x=df["Cancer Type"],
    y=df["Post-test Risk (if negative)"],
    marker_color='darkred',
    opacity=0.8,
    hovertemplate="<b>%{x}</b><br>Risk after negative test: %{y}%<extra></extra>"
))

fig_comparison.update_layout(
    title='Your Cancer Risk: Current vs After Negative Test',
    xaxis_title='Cancer Type',
    yaxis_title='Probability of Having Cancer (%)',
    barmode='group',
    height=500,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig_comparison, use_container_width=True)

# False positive and downstream risks
st.subheader("False Positive and Downstream Testing Risks")

downstream_info = DOWNSTREAM_RISKS[test_type]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "False Positive Rate", 
        f"{downstream_info['false_positive_rate']}%",
        help="Percentage of people without cancer who get incorrect positive results"
    )

with col2:
    st.metric(
        "Downstream Testing",
        f"{downstream_info['typical_followup']}"
    )

with col3:
    st.metric(
        "Complication Rates from follow-up",
        f"{downstream_info['followup_complications']}"
    )

# False positive risk by cancer type
fig_fp = go.Figure()

fig_fp.add_trace(go.Bar(
    x=df["Cancer Type"],
    y=df["False Positive Risk"],
    marker_color='orange',
    hovertemplate="<b>%{x}</b><br>False positive risk: %{y}%<br>This is your chance of getting an incorrect positive result<extra></extra>"
))

fig_fp.update_layout(
    title='False Positive Risk by Cancer Type',
    xaxis_title='Cancer Type',
    yaxis_title='Probability of False Positive Result (%)',
    height=400
)

st.plotly_chart(fig_fp, use_container_width=True)

# Test performance visualization - simplified and clearer
st.subheader("Test Performance Analysis")

# Create two meaningful charts instead of four confusing ones
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        'Test Trade-offs: Detection vs False Positives', 
        'Personal Risk Assessment'
    ),
    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
)

# Chart 1: Detection rate vs False positive rate (more meaningful comparison)
fig.add_trace(
    go.Scatter(
        x=df["False Positive Risk"], 
        y=df["Detection Rate"],
        mode='markers+text',
        text=df["Cancer Type"],
        textposition="top center",
        marker=dict(size=12, color='steelblue'),
        showlegend=False,
        hovertemplate="<b>%{text}</b><br>Detection Rate: %{y}%<br>False Positive Risk: %{x}%<br><extra></extra>"
    ),
    row=1, col=1
)

# Chart 2: Only show cancers where user has meaningful risk (>0.01%)
meaningful_risk_df = df[df["Your Risk"] > 0.01].copy()
if len(meaningful_risk_df) == 0:
    meaningful_risk_df = df.nlargest(3, "Your Risk")  # Show top 3 if all risks are very low

# Risk reduction chart (more actionable than absolute risk)
fig.add_trace(
    go.Bar(
        x=meaningful_risk_df["Cancer Type"], 
        y=meaningful_risk_df["Risk Reduction"], 
        marker_color='forestgreen', 
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Negative test reduces your cancer probability by %{y}%<extra></extra>"
    ),
    row=1, col=2
)

fig.update_layout(
    height=500, 
    showlegend=False
)

fig.update_xaxes(title_text="False Positive Risk (%)", row=1, col=1)
fig.update_yaxes(title_text="Cancer Detection Rate (%)", row=1, col=1)
fig.update_xaxes(title_text="Cancer Type", row=1, col=2)
fig.update_yaxes(title_text="Risk Reduction from Negative Test (%)", row=1, col=2)
fig.update_xaxes(tickangle=45, row=1, col=2)

st.plotly_chart(fig, use_container_width=True)

# Add explanatory text for the charts
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Test Trade-offs Chart:**
    - **Upper left** = Good detection, low false positives (ideal)
    - **Lower right** = Poor detection, high false positives (worst)
    - Each point represents a different cancer type
    """)

with col2:
    highest_risk_cancers = df.nlargest(3, "Your Risk")["Cancer Type"].tolist()
    st.markdown(f"""
    **Personal Risk Chart:**
    - Shows how much confidence a negative test gives you
    - Only displays cancers where you have meaningful risk
    - Your top risk areas: {', '.join(highest_risk_cancers[:2])}
    """)

# Risk reduction summary table
st.subheader("Personalized Risk Analysis")
st.markdown("*How your risk factors affect cancer probability and how much confidence a negative test provides*")

risk_summary = df[["Cancer Type", "Baseline Risk", "Your Risk", "Risk Multiplier",
                  "Post-test Risk (if negative)", "Risk Reduction", "False Positive Risk"]].copy()

st.dataframe(
    risk_summary,
    use_container_width=True,
    column_config={
        "Cancer Type": "Cancer Type",
        "Baseline Risk": st.column_config.NumberColumn(
            "Average Risk (%)", 
            format="%.3f",
            help="Average risk for someone your age and sex"
        ),
        "Your Risk": st.column_config.NumberColumn(
            "Your Risk (%)", 
            format="%.3f",
            help="Your personalized risk considering all risk factors"
        ),
        "Risk Multiplier": st.column_config.NumberColumn(
            "Risk Multiplier", 
            format="%.1f",
            help="How much your risk factors increase cancer probability"
        ),
        "Post-test Risk (if negative)": st.column_config.NumberColumn(
            "Risk After Negative Test (%)", 
            format="%.4f",
            help="Your probability of still having cancer even after negative test"
        ),
        "Risk Reduction": st.column_config.NumberColumn(
            "Risk Reduction (%)", 
            format="%.1f",
            help="Percentage reduction in cancer probability from negative test"
        ),
        "False Positive Risk": st.column_config.NumberColumn(
            "False Positive Risk (%)", 
            format="%.2f",
            help="Your probability of getting incorrect positive result"
        )
    }
)

# Key insights with clear explanations
st.subheader("Key Insights")

best_risk_reduction = df.loc[df["Risk Reduction"].idxmax()]
highest_false_positive = df.loc[df["False Positive Risk"].idxmax()]
highest_current_risk = df.loc[df["Your Risk"].idxmax()]
highest_multiplier = df.loc[df["Risk Multiplier"].idxmax()]

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    **Most Reassuring Negative Test:**
    {best_risk_reduction['Cancer Type']} provides {best_risk_reduction['Risk Reduction']}% risk reduction
    
    **Highest Risk Impact:**
    {highest_multiplier['Cancer Type']} - {highest_multiplier['Risk Multiplier']}x higher than average
    """)

with col2:
    st.markdown(f"""
    **Your Highest Current Risk:**
    {highest_current_risk['Cancer Type']} at {highest_current_risk['Your Risk']}% probability
    
    **Highest False Positive Risk:**
    {highest_false_positive['Cancer Type']} has {highest_false_positive['False Positive Risk']}% chance of incorrect positive
    
    **Overall False Positive Rate:**
    {downstream_info['false_positive_rate']}% of people without cancer get positive results
    """)

# Test-specific downstream risks
st.subheader(f"Downstream Risks: {test_type}")

st.markdown(f"""
**If you get a positive result, you will likely need:**
{downstream_info['typical_followup']}

**Risks from follow-up testing:**
- **Complication rate:** {downstream_info['followup_complications']}% chance of complications from additional procedures
- **Psychological impact:** {downstream_info['psychological_impact']}
- **Radiation exposure:** {downstream_info['radiation_exposure']}

**False positive burden:** Out of 1,000 people without cancer who get this test, 
{int(downstream_info['false_positive_rate'] * 10)} will receive incorrect positive results and undergo unnecessary follow-up.
""")

# Complete results table
st.subheader("Complete Test Performance Data")

display_df = df[["Cancer Type", "Baseline Risk", "Your Risk", "Risk Multiplier", 
                "Detection Rate", "Accuracy Rate", "Positive Accuracy", "Negative Accuracy", 
                "False Positive Risk"]].copy()

st.dataframe(
    display_df,
    use_container_width=True,
    column_config={
        "Cancer Type": "Cancer Type",
        "Baseline Risk": st.column_config.NumberColumn("Average Risk (%)", format="%.3f"),
        "Your Risk": st.column_config.NumberColumn("Your Risk (%)", format="%.3f"),
        "Risk Multiplier": st.column_config.NumberColumn("Risk Multiplier", format="%.1f"),
        "Detection Rate": st.column_config.NumberColumn("Detection Rate (%)", format="%.1f"),
        "Accuracy Rate": st.column_config.NumberColumn("Accuracy Rate (%)", format="%.1f"),
        "Positive Accuracy": st.column_config.NumberColumn("Positive Reliability (%)", format="%.1f"),
        "Negative Accuracy": st.column_config.NumberColumn("Negative Reliability (%)", format="%.1f"),
        "False Positive Risk": st.column_config.NumberColumn("False Positive Risk (%)", format="%.2f")
    }
)

# Disclaimers
st.markdown("---")
st.warning("""
**Important:** This tool is for educational purposes only. Always consult healthcare providers 
for medical decisions. Risk calculations are based on population studies and may not reflect 
individual circumstances. Genetic counseling is recommended for high-risk genetic mutations.
""")

st.info("""
**Data Sources:** Clinical trial data from CCGA validation study (Grail), NLST trial (CT), 
systematic reviews (MRI), SEER/CDC cancer statistics (2018-2022), and epidemiological studies 
on cancer risk factors. Risk multipliers derived from meta-analyses and cohort studies.
""")