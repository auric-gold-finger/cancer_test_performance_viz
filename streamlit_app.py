import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configure the app
st.set_page_config(
    page_title="Cancer Screening Test Performance",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Cancer Screening Test Performance Analyzer")
st.markdown("Visualize Positive Predictive Value (PPV) and Negative Predictive Value (NPV) for different cancer screening tests")

# Test performance data (sensitivity and specificity)
TEST_PERFORMANCE = {
    "Whole-body MRI": {
        "lung": {"sensitivity": 0.85, "specificity": 0.92},
        "breast": {"sensitivity": 0.88, "specificity": 0.94},
        "colorectal": {"sensitivity": 0.82, "specificity": 0.90},
        "prostate": {"sensitivity": 0.90, "specificity": 0.88},
        "liver": {"sensitivity": 0.87, "specificity": 0.93},
        "pancreatic": {"sensitivity": 0.78, "specificity": 0.89},
        "ovarian": {"sensitivity": 0.83, "specificity": 0.91},
        "kidney": {"sensitivity": 0.89, "specificity": 0.95}
    },
    "Grail MCED": {
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
        "lung": {"sensitivity": 0.94, "specificity": 0.73},
        "breast": {"sensitivity": 0.70, "specificity": 0.85},
        "colorectal": {"sensitivity": 0.88, "specificity": 0.82},
        "prostate": {"sensitivity": 0.75, "specificity": 0.80},
        "liver": {"sensitivity": 0.92, "specificity": 0.88},
        "pancreatic": {"sensitivity": 0.85, "specificity": 0.84},
        "ovarian": {"sensitivity": 0.78, "specificity": 0.87},
        "kidney": {"sensitivity": 0.91, "specificity": 0.89}
    }
}

# Age and sex-adjusted cancer incidence rates (per 100,000)
CANCER_INCIDENCE = {
    "lung": {
        "male": {40: 15, 50: 45, 60: 120, 70: 280, 80: 350},
        "female": {40: 18, 50: 38, 60: 95, 70: 180, 80: 220}
    },
    "breast": {
        "male": {40: 1, 50: 2, 60: 3, 70: 4, 80: 5},
        "female": {40: 150, 50: 250, 60: 350, 70: 400, 80: 420}
    },
    "colorectal": {
        "male": {40: 25, 50: 60, 60: 140, 70: 280, 80: 350},
        "female": {40: 20, 50: 45, 60: 110, 70: 220, 80: 280}
    },
    "prostate": {
        "male": {40: 10, 50: 80, 60: 400, 70: 800, 80: 1000},
        "female": {40: 0, 50: 0, 60: 0, 70: 0, 80: 0}
    },
    "liver": {
        "male": {40: 8, 50: 18, 60: 35, 70: 55, 80: 65},
        "female": {40: 3, 50: 8, 60: 18, 70: 28, 80: 35}
    },
    "pancreatic": {
        "male": {40: 5, 50: 12, 60: 28, 70: 55, 80: 70},
        "female": {40: 4, 50: 10, 60: 24, 70: 48, 80: 62}
    },
    "ovarian": {
        "male": {40: 0, 50: 0, 60: 0, 70: 0, 80: 0},
        "female": {40: 15, 50: 25, 60: 35, 70: 40, 80: 42}
    },
    "kidney": {
        "male": {40: 12, 50: 25, 60: 45, 70: 65, 80: 75},
        "female": {40: 6, 50: 14, 60: 28, 70: 42, 80: 52}
    }
}

def calculate_ppv_npv(sensitivity, specificity, prevalence):
    """Calculate PPV and NPV given test characteristics and disease prevalence"""
    ppv = (sensitivity * prevalence) / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence))
    npv = (specificity * (1 - prevalence)) / ((1 - sensitivity) * prevalence + specificity * (1 - prevalence))
    return ppv, npv

def get_prevalence_from_incidence(incidence_rate):
    """Convert annual incidence rate per 100,000 to prevalence"""
    # Rough approximation: prevalence ≈ incidence × average duration
    # Using 5-year prevalence as approximation
    return (incidence_rate / 100000) * 5

def interpolate_incidence(age, sex, cancer_type):
    """Interpolate incidence rate for given age"""
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

# Age input
age = st.sidebar.slider("Age", min_value=30, max_value=90, value=55, step=1)

# Sex input
sex = st.sidebar.selectbox("Sex", ["male", "female"])

# Test type
test_type = st.sidebar.selectbox(
    "Test Type", 
    list(TEST_PERFORMANCE.keys())
)

# Custom pre-test probability option
use_custom_probability = st.sidebar.checkbox("Use custom pre-test probability")

if use_custom_probability:
    custom_probability = st.sidebar.slider(
        "Custom pre-test probability (%)", 
        min_value=0.01, 
        max_value=10.0, 
        value=1.0, 
        step=0.01
    )

# Calculate results for all cancer types
results = []
cancer_types = list(TEST_PERFORMANCE[test_type].keys())

for cancer_type in cancer_types:
    # Skip irrelevant combinations
    if cancer_type == "prostate" and sex == "female":
        continue
    if cancer_type == "ovarian" and sex == "male":
        continue
        
    # Get test performance
    sensitivity = TEST_PERFORMANCE[test_type][cancer_type]["sensitivity"]
    specificity = TEST_PERFORMANCE[test_type][cancer_type]["specificity"]
    
    # Calculate prevalence
    if use_custom_probability:
        prevalence = custom_probability / 100
    else:
        incidence_rate = interpolate_incidence(age, sex, cancer_type)
        prevalence = get_prevalence_from_incidence(incidence_rate)
    
    # Calculate PPV and NPV
    ppv, npv = calculate_ppv_npv(sensitivity, specificity, prevalence)
    
    results.append({
        "Cancer Type": cancer_type.title(),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Prevalence (%)": prevalence * 100,
        "PPV (%)": ppv * 100,
        "NPV (%)": npv * 100
    })

# Create DataFrame
df = pd.DataFrame(results)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Test Performance: {test_type}")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Positive Predictive Value (PPV)', 'Negative Predictive Value (NPV)', 
                       'Sensitivity vs Specificity', 'Prevalence by Cancer Type'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # PPV bar chart
    fig.add_trace(
        go.Bar(x=df["Cancer Type"], y=df["PPV (%)"], name="PPV", 
               marker_color='lightcoral', showlegend=False),
        row=1, col=1
    )
    
    # NPV bar chart
    fig.add_trace(
        go.Bar(x=df["Cancer Type"], y=df["NPV (%)"], name="NPV", 
               marker_color='lightblue', showlegend=False),
        row=1, col=2
    )
    
    # Sensitivity vs Specificity scatter
    fig.add_trace(
        go.Scatter(x=df["Specificity"], y=df["Sensitivity"], 
                  mode='markers+text', text=df["Cancer Type"],
                  textposition="top center", marker=dict(size=10, color='green'),
                  name="Test Performance", showlegend=False),
        row=2, col=1
    )
    
    # Prevalence bar chart
    fig.add_trace(
        go.Bar(x=df["Cancer Type"], y=df["Prevalence (%)"], name="Prevalence", 
               marker_color='orange', showlegend=False),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"Cancer Screening Performance Analysis - {test_type}",
        showlegend=False
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="PPV (%)", row=1, col=1)
    fig.update_yaxes(title_text="NPV (%)", row=1, col=2)
    fig.update_yaxes(title_text="Sensitivity", row=2, col=1)
    fig.update_yaxes(title_text="Prevalence (%)", row=2, col=2)
    fig.update_xaxes(title_text="Specificity", row=2, col=1)
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Parameters")
    st.write(f"**Age:** {age} years")
    st.write(f"**Sex:** {sex.title()}")
    st.write(f"**Test:** {test_type}")
    
    if use_custom_probability:
        st.write(f"**Custom Pre-test Probability:** {custom_probability}%")
    else:
        st.write("**Using age/sex-adjusted prevalence**")
    
    st.subheader("Key Metrics")
    
    # Summary statistics
    avg_ppv = df["PPV (%)"].mean()
    avg_npv = df["NPV (%)"].mean()
    
    st.metric("Average PPV", f"{avg_ppv:.1f}%")
    st.metric("Average NPV", f"{avg_npv:.1f}%")
    
    # Best performing cancer types
    best_ppv = df.loc[df["PPV (%)"].idxmax(), "Cancer Type"]
    best_npv = df.loc[df["NPV (%)"].idxmax(), "Cancer Type"]
    
    st.write(f"**Highest PPV:** {best_ppv}")
    st.write(f"**Highest NPV:** {best_npv}")

# Detailed results table
st.subheader("Detailed Results")
st.dataframe(df.round(2), use_container_width=True)

# Educational content
with st.expander("📚 Understanding PPV and NPV"):
    st.markdown("""
    **Positive Predictive Value (PPV):** The probability that a person with a positive test result actually has the disease.
    - Higher PPV = fewer false positives
    - PPV increases with higher disease prevalence
    
    **Negative Predictive Value (NPV):** The probability that a person with a negative test result truly does not have the disease.
    - Higher NPV = fewer false negatives
    - NPV decreases with higher disease prevalence
    
    **Sensitivity:** The ability of a test to correctly identify those with the disease (true positive rate).
    
    **Specificity:** The ability of a test to correctly identify those without the disease (true negative rate).
    
    **Note:** This tool uses estimated values for educational purposes. Actual test performance may vary based on specific protocols, patient populations, and other factors.
    """)

# Interactive prevalence analysis
st.subheader("📊 Interactive Prevalence Analysis")

selected_cancer = st.selectbox("Select cancer type for prevalence analysis:", 
                              [c.title() for c in cancer_types if not (c == "prostate" and sex == "female") and not (c == "ovarian" and sex == "male")])

if selected_cancer:
    cancer_key = selected_cancer.lower()
    
    # Create prevalence range analysis
    prevalence_range = np.logspace(-4, -1, 50)  # 0.01% to 10%
    ppv_range = []
    npv_range = []
    
    sensitivity = TEST_PERFORMANCE[test_type][cancer_key]["sensitivity"]
    specificity = TEST_PERFORMANCE[test_type][cancer_key]["specificity"]
    
    for prev in prevalence_range:
        ppv, npv = calculate_ppv_npv(sensitivity, specificity, prev)
        ppv_range.append(ppv * 100)
        npv_range.append(npv * 100)
    
    # Create the plot
    fig_prev = go.Figure()
    
    fig_prev.add_trace(go.Scatter(
        x=prevalence_range * 100,
        y=ppv_range,
        mode='lines',
        name='PPV',
        line=dict(color='red', width=3)
    ))
    
    fig_prev.add_trace(go.Scatter(
        x=prevalence_range * 100,
        y=npv_range,
        mode='lines',
        name='NPV',
        line=dict(color='blue', width=3)
    ))
    
    # Add current prevalence marker
    current_prev = df[df["Cancer Type"] == selected_cancer]["Prevalence (%)"].iloc[0]
    current_ppv = df[df["Cancer Type"] == selected_cancer]["PPV (%)"].iloc[0]
    current_npv = df[df["Cancer Type"] == selected_cancer]["NPV (%)"].iloc[0]
    
    fig_prev.add_trace(go.Scatter(
        x=[current_prev, current_prev],
        y=[current_ppv, current_npv],
        mode='markers',
        name='Current Values',
        marker=dict(size=12, color='green', symbol='diamond')
    ))
    
    fig_prev.update_layout(
        title=f'PPV and NPV vs Prevalence - {selected_cancer} ({test_type})',
        xaxis_title='Prevalence (%)',
        yaxis_title='Predictive Value (%)',
        xaxis_type='log',
        height=500
    )
    
    st.plotly_chart(fig_prev, use_container_width=True)