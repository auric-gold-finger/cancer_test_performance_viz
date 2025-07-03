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

# Sidebar inputs
st.sidebar.header("Input Parameters")

age = st.sidebar.slider("Age", min_value=30, max_value=90, value=55, step=1)
sex = st.sidebar.selectbox("Sex", ["male", "female"])

test_type = st.sidebar.selectbox(
    "Screening Test",
    ["Whole-body MRI", "Grail Blood Test", "CT Scan"]
)

use_custom_probability = st.sidebar.checkbox("Use custom cancer risk")

if use_custom_probability:
    custom_probability = st.sidebar.slider(
        "Cancer risk (%)", 
        min_value=0.1, max_value=20.0, value=5.0, step=0.1
    )

# Calculate results
results = []
cancer_types = list(TEST_PERFORMANCE[test_type].keys())

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
        prevalence = get_prevalence_from_incidence(incidence_rate)
    
    ppv, npv = calculate_ppv_npv(sensitivity, specificity, prevalence)
    post_test_risk = calculate_post_test_risk_negative(sensitivity, prevalence)
    false_positive_risk = (1 - specificity) * (1 - prevalence)
    
    results.append({
        "Cancer Type": cancer_type.replace("_", " ").title(),
        "Pre-test Risk": round(prevalence * 100, 3),
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

# Clear explanation of risk measures
st.markdown("""
**Risk Definitions:**
- **Pre-test Risk**: Your probability of having cancer right now, before any testing
- **Post-test Risk (if negative)**: Your probability of still having cancer even after a negative test result
- **False Positive Risk**: Probability the test will incorrectly say you have cancer when you don't
""")

# Risk comparison section
st.subheader("Cancer Risk: Before Testing vs After Negative Test")
st.markdown("*This shows how much a negative test reduces your cancer probability*")

# Create side-by-side comparison chart
fig_comparison = go.Figure()

fig_comparison.add_trace(go.Bar(
    name='Probability You Have Cancer Now',
    x=df["Cancer Type"],
    y=df["Pre-test Risk"],
    marker_color='lightcoral',
    opacity=0.8,
    hovertemplate="<b>%{x}</b><br>Probability you have cancer now: %{y}%<extra></extra>"
))

fig_comparison.add_trace(go.Bar(
    name='Probability You Still Have Cancer After Negative Test',
    x=df["Cancer Type"],
    y=df["Post-test Risk (if negative)"],
    marker_color='darkred',
    opacity=0.8,
    hovertemplate="<b>%{x}</b><br>Probability you still have cancer after negative test: %{y}%<extra></extra>"
))

fig_comparison.update_layout(
    title='Your Cancer Probability: Current Risk vs Risk After Negative Test',
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

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "False Positive Rate", 
        f"{downstream_info['false_positive_rate']}%",
        help="Percentage of people without cancer who get incorrect positive results"
    )
    
    st.markdown(f"""
    **What happens after a positive test:**
    {downstream_info['typical_followup']}
    
    **Complication rate from follow-up procedures:**
    {downstream_info['followup_complications']}%
    """)

with col2:
    st.markdown(f"""
    **Psychological impact:**
    {downstream_info['psychological_impact']}
    
    **Additional radiation exposure:**
    {downstream_info['radiation_exposure']}
    """)

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

# Test performance visualization with clear labels
st.subheader("Detailed Test Performance")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Cancer Detection Rate (How often test finds cancer when present)', 
        'Test Accuracy Rate (How often test correctly identifies no cancer)',
        'Positive Test Reliability (When test says cancer, probability it\'s correct)', 
        'Your Current Cancer Risk by Type'
    ),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Detection rate (sensitivity)
fig.add_trace(
    go.Bar(
        x=df["Cancer Type"], 
        y=df["Detection Rate"], 
        marker_color='coral', 
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Detects %{y}% of cancers when present<extra></extra>"
    ),
    row=1, col=1
)

# Accuracy rate (specificity)
fig.add_trace(
    go.Bar(
        x=df["Cancer Type"], 
        y=df["Accuracy Rate"], 
        marker_color='lightblue', 
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Correctly identifies %{y}% of people without cancer<extra></extra>"
    ),
    row=1, col=2
)

# Positive accuracy (PPV) with color coding
colors = ['red' if x < 10 else 'orange' if x < 50 else 'green' 
          for x in df["Positive Accuracy"]]

fig.add_trace(
    go.Bar(
        x=df["Cancer Type"], 
        y=df["Positive Accuracy"], 
        marker_color=colors, 
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>When test is positive, %{y}% chance you actually have cancer<extra></extra>"
    ),
    row=2, col=1
)

# Current risk profile
fig.add_trace(
    go.Bar(
        x=df["Cancer Type"], 
        y=df["Pre-test Risk"], 
        marker_color='gold', 
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Your current probability of having this cancer: %{y}%<extra></extra>"
    ),
    row=2, col=2
)

fig.update_layout(height=700, showlegend=False)
fig.update_yaxes(title_text="Detection Rate (%)", row=1, col=1)
fig.update_yaxes(title_text="Accuracy Rate (%)", row=1, col=2)
fig.update_yaxes(title_text="Reliability (%)", row=2, col=1)
fig.update_yaxes(title_text="Current Risk (%)", row=2, col=2)
fig.update_xaxes(tickangle=45)

st.plotly_chart(fig, use_container_width=True)

# Risk reduction summary table
st.subheader("Risk Reduction Summary")
st.markdown("*How much confidence a negative test provides*")

risk_summary = df[["Cancer Type", "Pre-test Risk", "Post-test Risk (if negative)", 
                  "Risk Reduction", "False Positive Risk"]].copy()

st.dataframe(
    risk_summary,
    use_container_width=True,
    column_config={
        "Cancer Type": "Cancer Type",
        "Pre-test Risk": st.column_config.NumberColumn(
            "Current Risk (%)", 
            format="%.3f",
            help="Your probability of having this cancer right now"
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
highest_current_risk = df.loc[df["Pre-test Risk"].idxmax()]

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    **Most Reassuring Negative Test:**
    {best_risk_reduction['Cancer Type']} provides {best_risk_reduction['Risk Reduction']}% risk reduction
    
    **Highest False Positive Risk:**
    {highest_false_positive['Cancer Type']} has {highest_false_positive['False Positive Risk']}% chance of incorrect positive
    """)

with col2:
    st.markdown(f"""
    **Your Highest Current Risk:**
    {highest_current_risk['Cancer Type']} at {highest_current_risk['Pre-test Risk']}% probability
    
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

display_df = df[["Cancer Type", "Detection Rate", "Accuracy Rate", "Pre-test Risk", 
                "Positive Accuracy", "Negative Accuracy", "False Positive Risk"]].copy()

st.dataframe(
    display_df,
    use_container_width=True,
    column_config={
        "Cancer Type": "Cancer Type",
        "Detection Rate": st.column_config.NumberColumn("Detection Rate (%)", format="%.1f"),
        "Accuracy Rate": st.column_config.NumberColumn("Accuracy Rate (%)", format="%.1f"),
        "Pre-test Risk": st.column_config.NumberColumn("Current Risk (%)", format="%.3f"),
        "Positive Accuracy": st.column_config.NumberColumn("Positive Reliability (%)", format="%.1f"),
        "Negative Accuracy": st.column_config.NumberColumn("Negative Reliability (%)", format="%.1f"),
        "False Positive Risk": st.column_config.NumberColumn("False Positive Risk (%)", format="%.2f")
    }
)

# Disclaimers
st.markdown("---")
st.warning("""
**Important:** This tool is for educational purposes only. Always consult healthcare providers 
for medical decisions. Test performance varies based on individual factors not captured here.
Downstream complication rates are estimates based on published literature.
""")

st.info("""
**Data Sources:** Clinical trial data from CCGA validation study (Grail), NLST trial (CT), 
systematic reviews (MRI), SEER/CDC cancer statistics (2018-2022), and published studies on 
screening follow-up procedures and complications.
""")