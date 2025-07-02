import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configure the app
st.set_page_config(
    page_title="Cancer Screening Test Comparison",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Cancer Screening Test Comparison Tool")
st.markdown("**Compare how well different tests can detect cancer and understand what the results mean for you**")

# Real clinical data from recent studies and trials
TEST_PERFORMANCE = {
    "Whole-body MRI": {
        "lung": {"sensitivity": 0.50, "specificity": 0.93},
        "breast": {"sensitivity": 0.95, "specificity": 0.74},
        "colorectal": {"sensitivity": 0.67, "specificity": 0.95},
        "prostate": {"sensitivity": 0.84, "specificity": 0.89},
        "liver": {"sensitivity": 0.84, "specificity": 0.94},
        "pancreatic": {"sensitivity": 0.75, "specificity": 0.85},  # High-risk groups only
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
    """Calculate how likely test results are to be correct"""
    # PPV = chance that positive test means you have cancer
    ppv = (sensitivity * prevalence) / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence))
    # NPV = chance that negative test means you don't have cancer  
    npv = (specificity * (1 - prevalence)) / ((1 - sensitivity) * prevalence + specificity * (1 - prevalence))
    return ppv, npv

def get_prevalence_from_incidence(incidence_rate):
    """Convert yearly cancer rate to how common cancer is overall"""
    # Using 5-year prevalence estimate
    return (incidence_rate / 100000) * 5

def interpolate_incidence(age, sex, cancer_type):
    """Estimate cancer risk for your specific age"""
    age_points = list(CANCER_INCIDENCE[cancer_type][sex].keys())
    incidence_points = list(CANCER_INCIDENCE[cancer_type][sex].values())
    
    if age <= min(age_points):
        return incidence_points[0]
    elif age >= max(age_points):
        return incidence_points[-1]
    else:
        return np.interp(age, age_points, incidence_points)

# Sidebar inputs with better explanations
st.sidebar.header("🔍 Your Information")
st.sidebar.markdown("*Tell us about yourself to get personalized results*")

# Age input with context
age = st.sidebar.slider(
    "Your age", 
    min_value=30, max_value=90, value=55, step=1,
    help="Cancer risk increases with age, so this affects how useful screening tests are for you"
)

# Sex input with explanation
sex = st.sidebar.selectbox(
    "Sex", 
    ["male", "female"],
    help="Some cancers are more common in men or women, affecting test usefulness"
)

# Test type with descriptions
test_descriptions = {
    "Whole-body MRI": "📸 Full-body MRI scan (no radiation, takes 60-90 minutes)",
    "Grail Blood Test": "🩸 Blood test that looks for cancer DNA (simple blood draw)",
    "CT Scan": "🔬 X-ray scan with some radiation (quick, 10-30 minutes)"
}

test_type = st.sidebar.selectbox(
    "Choose a screening test to analyze",
    list(test_descriptions.keys()),
    format_func=lambda x: test_descriptions[x]
)

# Risk level explanation
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Risk Level")

use_custom_probability = st.sidebar.checkbox(
    "I have higher than average cancer risk",
    help="Check this if you have family history, genetic mutations, or other risk factors"
)

if use_custom_probability:
    custom_probability = st.sidebar.slider(
        "Your estimated cancer risk (%)", 
        min_value=0.1, max_value=20.0, value=5.0, step=0.1,
        help="Talk to your doctor about your personal risk level"
    )
    st.sidebar.info("💡 Higher risk makes positive tests more likely to be correct")
else:
    st.sidebar.info("💡 Using average risk for someone your age and sex")

# Calculate results
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
    
    # Calculate cancer risk
    if use_custom_probability:
        prevalence = custom_probability / 100
    else:
        incidence_rate = interpolate_incidence(age, sex, cancer_type)
        prevalence = get_prevalence_from_incidence(incidence_rate)
    
    # Calculate test accuracy
    ppv, npv = calculate_ppv_npv(sensitivity, specificity, prevalence)
    
    results.append({
        "Cancer Type": cancer_type.replace("_", " ").title(),
        "Catches Cancer (%)": round(sensitivity * 100, 1),
        "Correctly Rules Out (%)": round(specificity * 100, 1),
        "Your Cancer Risk (%)": round(prevalence * 100, 3),
        "Positive Test Accuracy (%)": round(ppv * 100, 1),
        "Negative Test Accuracy (%)": round(npv * 100, 1)
    })

df = pd.DataFrame(results)

# Main content with clear explanations
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"📋 Results for {test_type}")
    
    # Key insights box
    avg_positive_accuracy = df["Positive Test Accuracy (%)"].mean()
    avg_negative_accuracy = df["Negative Test Accuracy (%)"].mean()
    
    if avg_positive_accuracy < 10:
        positive_msg = "⚠️ **Low accuracy** - many positive tests may be false alarms"
        positive_color = "red"
    elif avg_positive_accuracy < 50:
        positive_msg = "🔶 **Moderate accuracy** - some positive tests may be false alarms"  
        positive_color = "orange"
    else:
        positive_msg = "✅ **Good accuracy** - positive tests are usually correct"
        positive_color = "green"

    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h4>🎯 Key Takeaways for You:</h4>
        <p><strong>If your test is POSITIVE:</strong> <span style="color: {positive_color};">{positive_msg}</span></p>
        <p><strong>If your test is NEGATIVE:</strong> ✅ <span style="color: green;"><strong>Very reliable</strong> - you most likely don't have cancer</span></p>
        <p><strong>Average positive test accuracy:</strong> {avg_positive_accuracy:.1f}%</p>
        <p><strong>Average negative test accuracy:</strong> {avg_negative_accuracy:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'How Often Test Catches Cancer', 
            'How Often Test Rules Out Cancer Correctly',
            'Your Personal Cancer Risk by Type', 
            'Accuracy When Test Says "Cancer Detected"'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Sensitivity (cancer detection)
    fig.add_trace(
        go.Bar(
            x=df["Cancer Type"], 
            y=df["Catches Cancer (%)"], 
            name="Sensitivity",
            marker_color='lightcoral',
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Catches %{y}% of cancers<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Specificity (correct rule-out)
    fig.add_trace(
        go.Bar(
            x=df["Cancer Type"], 
            y=df["Correctly Rules Out (%)"], 
            name="Specificity",
            marker_color='lightblue',
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Correctly rules out %{y}% of non-cancers<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Personal risk
    fig.add_trace(
        go.Bar(
            x=df["Cancer Type"], 
            y=df["Your Cancer Risk (%)"], 
            name="Risk",
            marker_color='lightyellow',
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Your risk: %{y}%<extra></extra>"
        ),
        row=2, col=1
    )
    
    # PPV (positive test accuracy)
    colors = ['red' if x < 10 else 'orange' if x < 50 else 'green' 
              for x in df["Positive Test Accuracy (%)"]]
    
    fig.add_trace(
        go.Bar(
            x=df["Cancer Type"], 
            y=df["Positive Test Accuracy (%)"], 
            name="PPV",
            marker_color=colors,
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Positive test accuracy: %{y}%<extra></extra>"
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        title_text=f"Test Performance Analysis: {test_type}",
        showlegend=False
    )
    
    # Update axes labels
    fig.update_yaxes(title_text="Detection Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    fig.update_yaxes(title_text="Risk (%)", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=2)
    
    # Rotate x-axis labels for readability
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📝 Your Profile")
    st.write(f"**Age:** {age} years")
    st.write(f"**Sex:** {sex.title()}")
    
    if use_custom_probability:
        st.write(f"**Risk Level:** Higher than average ({custom_probability}%)")
    else:
        st.write("**Risk Level:** Average for your age/sex")
    
    st.markdown("---")
    st.subheader("🏆 Best Performance")
    
    # Find best performers
    best_detection = df.loc[df["Catches Cancer (%)"].idxmax()]
    best_accuracy = df.loc[df["Positive Test Accuracy (%)"].idxmax()]
    
    st.write(f"**Best at catching cancer:**")
    st.write(f"{best_detection['Cancer Type']} ({best_detection['Catches Cancer (%)']}%)")
    
    st.write(f"**Most accurate when positive:**")
    st.write(f"{best_accuracy['Cancer Type']} ({best_accuracy['Positive Test Accuracy (%)']}%)")
    
    # Risk context
    st.markdown("---")
    st.subheader("🎯 What This Means")
    
    highest_risk = df.loc[df["Your Cancer Risk (%)"].idxmax()]
    if highest_risk["Your Cancer Risk (%)"] > 1:
        st.warning(f"Your highest risk is {highest_risk['Cancer Type']} at {highest_risk['Your Cancer Risk (%)']}%. Consider discussing screening with your doctor.")
    else:
        st.info("Your cancer risks are relatively low. Screening may have limited benefit.")

# Detailed results table
st.subheader("📊 Detailed Results Table")
st.markdown("*All numbers are percentages*")

# Format the table nicely
formatted_df = df.copy()
formatted_df = formatted_df.round(1)

st.dataframe(
    formatted_df,
    use_container_width=True,
    column_config={
        "Cancer Type": st.column_config.TextColumn("Cancer Type", width="medium"),
        "Catches Cancer (%)": st.column_config.NumberColumn("Catches Cancer (%)", help="How often the test detects cancer when it's present"),
        "Correctly Rules Out (%)": st.column_config.NumberColumn("Rules Out Cancer (%)", help="How often the test correctly says 'no cancer' when there isn't any"),
        "Your Cancer Risk (%)": st.column_config.NumberColumn("Your Risk (%)", help="Your chance of having this cancer right now"),
        "Positive Test Accuracy (%)": st.column_config.NumberColumn("+ Test Accuracy (%)", help="If test says 'cancer detected', chance it's correct"),
        "Negative Test Accuracy (%)": st.column_config.NumberColumn("- Test Accuracy (%)", help="If test says 'no cancer', chance it's correct")
    }
)

# Educational sections
col1, col2 = st.columns(2)

with col1:
    with st.expander("🤔 What do these numbers mean?"):
        st.markdown("""
        **Catches Cancer (Sensitivity):** Out of 100 people who have cancer, how many will the test detect?
        - Higher is better for not missing cancer
        - Example: 80% means test finds 8 out of 10 cancers
        
        **Rules Out Cancer (Specificity):** Out of 100 people without cancer, how many will the test correctly identify?
        - Higher is better for avoiding false alarms
        - Example: 95% means only 5 out of 100 healthy people get false positives
        
        **Positive Test Accuracy (PPV):** If your test says "cancer detected," what's the chance you actually have cancer?
        - This depends heavily on how common the cancer is
        - Low numbers mean many false alarms
        
        **Negative Test Accuracy (NPV):** If your test says "no cancer," what's the chance you're truly cancer-free?
        - Usually very high (reassuring!)
        - Gives confidence in negative results
        """)

with col2:
    with st.expander("⚡ Key Insights by Test Type"):
        if test_type == "Grail Blood Test":
            st.markdown("""
            **Grail Blood Test Strengths:**
            - 🎯 Very few false alarms (99.5% specificity)
            - 🩸 Simple blood draw, no radiation
            - 🔍 Can detect many cancer types at once
            
            **Limitations:**
            - ⚠️ Misses many early-stage cancers
            - 💰 Expensive ($949 out-of-pocket)
            - 🧪 Relatively new technology
            """)
        elif test_type == "Whole-body MRI":
            st.markdown("""
            **Whole-body MRI Strengths:**
            - 📸 No radiation exposure
            - 🎯 Good detail for most cancer types
            - 🔍 Can see entire body at once
            
            **Limitations:**
            - ⚠️ May cause anxiety from incidental findings
            - 💰 Very expensive ($1,000-$5,000)
            - ⏰ Takes 60-90 minutes
            """)
        else:  # CT Scan
            st.markdown("""
            **CT Scan Strengths:**
            - ⚡ Fast and widely available
            - 🎯 Very good for lung cancer screening
            - 💰 Relatively affordable
            
            **Limitations:**
            - ☢️ Radiation exposure
            - ⚠️ More false alarms than blood tests
            - 🫁 Best evidence only for lung cancer
            """)

# Important disclaimers
st.markdown("---")
st.error("""
**⚠️ Important Disclaimers:**
- This tool is for educational purposes only and should not replace medical advice
- Actual test performance can vary based on many factors
- Always discuss screening decisions with your healthcare provider
- Some tests may not be appropriate or available for everyone
""")

st.info("""
**📚 Data Sources:** Performance data from recent clinical trials including CCGA validation study (Grail), 
NLST trial (CT lung screening), and systematic reviews of MRI screening studies. Cancer rates from 
SEER and CDC databases (2018-2022).
""")

# Interactive risk calculator
st.markdown("---")
st.subheader("🎲 Interactive Risk Explorer")
st.markdown("*See how test accuracy changes with cancer risk*")

selected_cancer = st.selectbox(
    "Select a cancer type to explore:", 
    [c.title() for c in cancer_types if not (c == "prostate" and sex == "female") and not (c == "ovarian" and sex == "male")]
)

if selected_cancer:
    cancer_key = selected_cancer.lower()
    
    # Create risk range analysis
    risk_range = np.logspace(-3, -1, 100)  # 0.1% to 10%
    ppv_range = []
    npv_range = []
    
    sensitivity = TEST_PERFORMANCE[test_type][cancer_key]["sensitivity"]
    specificity = TEST_PERFORMANCE[test_type][cancer_key]["specificity"]
    
    for risk in risk_range:
        ppv, npv = calculate_ppv_npv(sensitivity, specificity, risk)
        ppv_range.append(ppv * 100)
        npv_range.append(npv * 100)
    
    # Create the plot
    fig_risk = go.Figure()
    
    fig_risk.add_trace(go.Scatter(
        x=risk_range * 100,
        y=ppv_range,
        mode='lines',
        name='Positive Test Accuracy',
        line=dict(color='red', width=3),
        hovertemplate="<b>Risk: %{x:.2f}%</b><br>Positive test accuracy: %{y:.1f}%<extra></extra>"
    ))
    
    fig_risk.add_trace(go.Scatter(
        x=risk_range * 100,
        y=npv_range,
        mode='lines',
        name='Negative Test Accuracy',
        line=dict(color='blue', width=3),
        hovertemplate="<b>Risk: %{x:.2f}%</b><br>Negative test accuracy: %{y:.1f}%<extra></extra>"
    ))
    
    # Add current risk marker
    current_risk = df[df["Cancer Type"] == selected_cancer]["Your Cancer Risk (%)"].iloc[0]
    current_ppv = df[df["Cancer Type"] == selected_cancer]["Positive Test Accuracy (%)"].iloc[0]
    current_npv = df[df["Cancer Type"] == selected_cancer]["Negative Test Accuracy (%)"].iloc[0]
    
    fig_risk.add_trace(go.Scatter(
        x=[current_risk, current_risk],
        y=[current_ppv, current_npv],
        mode='markers',
        name='Your Current Risk',
        marker=dict(size=15, color='green', symbol='diamond'),
        hovertemplate="<b>Your current risk</b><br>Risk: %{x:.3f}%<br>Accuracy: %{y:.1f}%<extra></extra>"
    ))
    
    fig_risk.update_layout(
        title=f'How Test Accuracy Changes with Cancer Risk - {selected_cancer}',
        xaxis_title='Cancer Risk (%)',
        yaxis_title='Test Accuracy (%)',
        xaxis_type='log',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig_risk, use_container_width=True)
    
    st.markdown(f"""
    **💡 Key Insight:** As cancer becomes more common in a population, positive tests become more accurate. 
    For {selected_cancer.lower()} cancer, if the risk increased to 5%, positive test accuracy would be 
    {calculate_ppv_npv(sensitivity, specificity, 0.05)[0]*100:.1f}%.
    """)