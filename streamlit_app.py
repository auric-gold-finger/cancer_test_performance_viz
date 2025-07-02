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
    """Calculate cancer risk after a negative test"""
    return ((1 - sensitivity) * prevalence) / ((1 - sensitivity) * prevalence + 1 - prevalence)

def get_prevalence_from_incidence(incidence_rate):
    """Convert yearly cancer rate to prevalence"""
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
    
    results.append({
        "Cancer Type": cancer_type.replace("_", " ").title(),
        "Detection Rate": round(sensitivity * 100, 1),
        "Accuracy Rate": round(specificity * 100, 1),
        "Pre-test Risk": round(prevalence * 100, 3),
        "Post-test Risk (if negative)": round(post_test_risk * 100, 4),
        "Positive Accuracy": round(ppv * 100, 1),
        "Negative Accuracy": round(npv * 100, 1),
        "Risk Reduction": round(((prevalence - post_test_risk) / prevalence) * 100, 1)
    })

df = pd.DataFrame(results)

# Main content
st.subheader(f"Test Performance: {test_type}")

# Summary metrics
col1, col2, col3 = st.columns(3)
with col1:
    avg_detection = df["Detection Rate"].mean()
    st.metric("Average Detection Rate", f"{avg_detection:.1f}%")
with col2:
    avg_accuracy = df["Accuracy Rate"].mean()
    st.metric("Average Accuracy Rate", f"{avg_accuracy:.1f}%")
with col3:
    avg_positive_accuracy = df["Positive Accuracy"].mean()
    st.metric("Average Positive Test Accuracy", f"{avg_positive_accuracy:.1f}%")

# Risk comparison section
st.subheader("Risk Profile: Before vs After Negative Test")

# Create side-by-side comparison
comparison_data = []
for _, row in df.iterrows():
    comparison_data.append({
        "Cancer Type": row["Cancer Type"],
        "Before Test": row["Pre-test Risk"],
        "After Negative Test": row["Post-test Risk (if negative)"],
        "Risk Reduction": row["Risk Reduction"]
    })

comparison_df = pd.DataFrame(comparison_data)

# Visualization of risk reduction
fig_comparison = go.Figure()

fig_comparison.add_trace(go.Bar(
    name='Before Test',
    x=comparison_df["Cancer Type"],
    y=comparison_df["Before Test"],
    marker_color='lightblue',
    opacity=0.8
))

fig_comparison.add_trace(go.Bar(
    name='After Negative Test',
    x=comparison_df["Cancer Type"],
    y=comparison_df["After Negative Test"],
    marker_color='darkblue',
    opacity=0.8
))

fig_comparison.update_layout(
    title='Cancer Risk Before and After Negative Test Result',
    xaxis_title='Cancer Type',
    yaxis_title='Risk (%)',
    barmode='group',
    height=400
)

st.plotly_chart(fig_comparison, use_container_width=True)

# Risk reduction table
st.subheader("Risk Reduction Summary")
st.markdown("How much a negative test reduces your cancer risk:")

formatted_comparison = comparison_df.copy()
formatted_comparison["Before Test"] = formatted_comparison["Before Test"].round(3)
formatted_comparison["After Negative Test"] = formatted_comparison["After Negative Test"].round(4)
formatted_comparison["Risk Reduction"] = formatted_comparison["Risk Reduction"].round(1)

st.dataframe(
    formatted_comparison,
    use_container_width=True,
    column_config={
        "Cancer Type": "Cancer Type",
        "Before Test": st.column_config.NumberColumn("Before Test (%)", format="%.3f"),
        "After Negative Test": st.column_config.NumberColumn("After Negative Test (%)", format="%.4f"),
        "Risk Reduction": st.column_config.NumberColumn("Risk Reduction (%)", format="%.1f")
    }
)

# Test performance visualization
st.subheader("Detailed Test Performance")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Detection Rate by Cancer Type', 'Accuracy Rate by Cancer Type', 
                   'Positive Test Accuracy', 'Your Cancer Risk Profile'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Detection rate
fig.add_trace(
    go.Bar(x=df["Cancer Type"], y=df["Detection Rate"], 
           marker_color='coral', showlegend=False),
    row=1, col=1
)

# Accuracy rate
fig.add_trace(
    go.Bar(x=df["Cancer Type"], y=df["Accuracy Rate"], 
           marker_color='lightblue', showlegend=False),
    row=1, col=2
)

# Positive accuracy with color coding
colors = ['red' if x < 10 else 'orange' if x < 50 else 'green' 
          for x in df["Positive Accuracy"]]

fig.add_trace(
    go.Bar(x=df["Cancer Type"], y=df["Positive Accuracy"], 
           marker_color=colors, showlegend=False),
    row=2, col=1
)

# Risk profile
fig.add_trace(
    go.Bar(x=df["Cancer Type"], y=df["Pre-test Risk"], 
           marker_color='gold', showlegend=False),
    row=2, col=2
)

fig.update_layout(height=600, showlegend=False)
fig.update_yaxes(title_text="Rate (%)", row=1, col=1)
fig.update_yaxes(title_text="Rate (%)", row=1, col=2)
fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
fig.update_yaxes(title_text="Risk (%)", row=2, col=2)
fig.update_xaxes(tickangle=45)

st.plotly_chart(fig, use_container_width=True)

# Complete results table
st.subheader("Complete Results")
display_df = df[["Cancer Type", "Detection Rate", "Accuracy Rate", "Pre-test Risk", 
                "Positive Accuracy", "Negative Accuracy"]].copy()

st.dataframe(
    display_df,
    use_container_width=True,
    column_config={
        "Cancer Type": "Cancer Type",
        "Detection Rate": st.column_config.NumberColumn("Detection Rate (%)", format="%.1f"),
        "Accuracy Rate": st.column_config.NumberColumn("Accuracy Rate (%)", format="%.1f"),
        "Pre-test Risk": st.column_config.NumberColumn("Your Risk (%)", format="%.3f"),
        "Positive Accuracy": st.column_config.NumberColumn("Positive Accuracy (%)", format="%.1f"),
        "Negative Accuracy": st.column_config.NumberColumn("Negative Accuracy (%)", format="%.1f")
    }
)

# Key insights
st.subheader("Key Insights")

best_detection = df.loc[df["Detection Rate"].idxmax()]
best_risk_reduction = df.loc[df["Risk Reduction"].idxmax()]
highest_risk = df.loc[df["Pre-test Risk"].idxmax()]

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    **Test Performance:**
    - Best detection: {best_detection['Cancer Type']} ({best_detection['Detection Rate']}%)
    - Highest risk reduction from negative test: {best_risk_reduction['Cancer Type']} ({best_risk_reduction['Risk Reduction']}%)
    """)

with col2:
    st.markdown(f"""
    **Your Risk Profile:**
    - Highest personal risk: {highest_risk['Cancer Type']} ({highest_risk['Pre-test Risk']}%)
    - Average positive test accuracy: {avg_positive_accuracy:.1f}%
    """)

# Test-specific information
st.subheader(f"About {test_type}")

test_info = {
    "Grail Blood Test": {
        "description": "Blood test that detects cancer DNA fragments",
        "strengths": ["Very low false positive rate (0.5%)", "Simple blood draw", "Multiple cancer types"],
        "limitations": ["Misses many early cancers", "Expensive", "New technology"]
    },
    "Whole-body MRI": {
        "description": "Magnetic resonance imaging of entire body",
        "strengths": ["No radiation", "Detailed images", "Good for soft tissues"],
        "limitations": ["Expensive", "Time consuming", "Many incidental findings"]
    },
    "CT Scan": {
        "description": "X-ray computed tomography scan",
        "strengths": ["Fast", "Widely available", "Good for lung screening"],
        "limitations": ["Radiation exposure", "Higher false positive rates", "Limited to certain cancers"]
    }
}

info = test_info[test_type]
st.write(f"**Description:** {info['description']}")

col1, col2 = st.columns(2)
with col1:
    st.write("**Strengths:**")
    for strength in info['strengths']:
        st.write(f"- {strength}")

with col2:
    st.write("**Limitations:**")
    for limitation in info['limitations']:
        st.write(f"- {limitation}")

# Disclaimers
st.markdown("---")
st.warning("""
**Important:** This tool is for educational purposes only. Always consult healthcare providers 
for medical decisions. Test performance varies based on individual factors not captured here.
""")

st.info("""
**Data Sources:** Clinical trial data from CCGA validation study (Grail), NLST trial (CT), 
systematic reviews (MRI), and SEER/CDC cancer statistics (2018-2022).
""")