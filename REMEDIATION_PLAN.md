# Code Remediation Plan
## Cancer Test Performance Visualizer

**Date:** 2025-10-27
**Status:** Post-Critical Fixes
**Completed:** 3/24 issues (Critical tier complete)
**Remaining:** 21 issues (5 High, 9 Medium, 7 Low)

---

## Executive Summary

This document provides a detailed remediation plan for all remaining code quality, security, and architectural issues identified during the comprehensive code review. The critical issues have been resolved. This plan addresses the remaining 21 issues across three priority tiers.

**Estimated Total Effort:** 2-3 weeks (1 developer)
**Recommended Approach:** Incremental implementation in priority order
**Risk Level After Fixes:** Low (currently Medium)

---

## Issue Tracking Summary

| Priority | Count | Status | Estimated Effort |
|----------|-------|--------|------------------|
| Critical | 3 | ✅ **COMPLETED** | 2 hours |
| High | 5 | 🔴 Pending | 1 week |
| Medium | 9 | 🟡 Pending | 1 week |
| Low | 7 | 🟢 Pending | 3-5 days |

---

# HIGH PRIORITY ISSUES (5)

## H1: Unsafe HTML Rendering (3 instances)

**Location:** Lines 54, 56, 61
**Risk:** Medium Security Risk
**Effort:** 2-3 hours

### Current Code:
```python
st.markdown("<h1 class='main-header'>Cancer Screening...</h1>", unsafe_allow_html=True)
st.markdown("""<div class='warning-box'>...""", unsafe_allow_html=True)
```

### Problem:
Using `unsafe_allow_html=True` without sanitization creates XSS vulnerability potential. While currently safe (no user input in these strings), it's a dangerous pattern.

### Remediation Steps:

**Option 1: Use Native Streamlit Components (Recommended)**
```python
# Replace line 56
st.markdown("# Cancer Screening Outcomes Visualizer")

# Replace line 61 with native component
st.warning("**Medical Disclaimer:** This tool is for educational purposes only...")
```

**Option 2: Add HTML Sanitization**
```python
import html

# If HTML is absolutely necessary
def safe_html(content):
    """Sanitize HTML content"""
    return html.escape(content)
```

**Option 3: Move CSS to .streamlit/config.toml**
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f2937"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f9fafb"
textColor = "#1f2937"
```

### Implementation Priority: **Week 1, Day 1-2**

### Testing:
- Verify all styling works without unsafe_allow_html
- Check warning box appearance
- Test theme enforcement

---

## H2: Hardcoded Medical Risk Multipliers

**Location:** Lines 216-325 (get_risk_multiplier_fixed function)
**Risk:** High Maintainability Risk
**Effort:** 1 day

### Problem:
All medical risk data is hardcoded in the function, making updates difficult and error-prone. Medical data should be easily updatable by domain experts without code changes.

### Remediation Steps:

**Step 1: Create Configuration File**

Create `config/risk_factors.yaml`:
```yaml
# Risk multiplier configuration
# Data source: [Add medical literature references]
# Last updated: 2025-10-27

smoking_risk:
  current_smoker:
    lung:
      - pack_years: [0, 20]
        multiplier: 10.0
      - pack_years: [20, 40]
        multiplier: 15.0
      - pack_years: [40, 999]
        multiplier: 20.0
    bladder:
      multiplier: 2.0
    kidney:
      multiplier: 2.0
    pancreatic:
      multiplier: 2.0
    cervical:
      multiplier: 2.0
    esophageal:
      multiplier: 2.0
    gastric:
      multiplier: 2.0
    head_neck:
      multiplier: 2.0
    colorectal:
      multiplier: 1.6
    liver:
      multiplier: 1.6

  former_smoker:
    lung:
      - pack_years: [0, 20]
        multiplier: 5.0
      - pack_years: [20, 40]
        multiplier: 8.0
      - pack_years: [40, 999]
        multiplier: 12.0
    bladder:
      multiplier: 1.5
    kidney:
      multiplier: 1.5
    # ... continue for all cancer types

family_history_risk:
  colorectal:
    base_multiplier_early: 2.5  # if diagnosed < 60
    base_multiplier_late: 1.8   # if diagnosed >= 60
    age_threshold: 60
  breast:
    base_multiplier_early: 2.2  # if diagnosed < 50
    base_multiplier_late: 1.8   # if diagnosed >= 50
    age_threshold: 50
  prostate:
    base_multiplier: 2.0
  ovarian:
    base_multiplier: 2.5
  lung:
    base_multiplier: 1.5
  pancreatic:
    base_multiplier: 1.5

genetic_mutations:
  BRCA1:
    breast: 8.0
    ovarian: 6.0
  BRCA2:
    breast: 5.0
    ovarian: 3.0
    prostate: 3.0
  "Lynch syndrome":
    colorectal: 6.0
    ovarian: 3.0
    endometrial: 5.0
  "TP53 (Li-Fraumeni)":
    breast: 4.0
    lung: 4.0
    colorectal: 4.0
    liver: 4.0
    brain: 4.0
    sarcoma: 4.0

personal_history:
  multiplier: 2.0

# Safety limits
max_individual_risk: 0.60  # 60% cap
max_overall_risk: 0.85     # 85% cap
max_risk_multiplier: 25.0  # 25x cap
```

**Step 2: Add YAML Dependency**

Update `requirements.txt`:
```
pyyaml>=6.0.0,<7.0.0
```

**Step 3: Create Configuration Loader**

Create `config/config_loader.py`:
```python
import yaml
from pathlib import Path
from typing import Dict, Any

class RiskConfig:
    """Load and manage risk factor configuration"""

    def __init__(self, config_path: str = "config/risk_factors.yaml"):
        self.config_path = Path(config_path)
        self._config = None

    @property
    def config(self) -> Dict[str, Any]:
        """Lazy load configuration"""
        if self._config is None:
            self._load_config()
        return self._config

    def _load_config(self):
        """Load YAML configuration file"""
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get_smoking_risk(self, smoking_status: str, cancer_type: str,
                         pack_years: int = 0) -> float:
        """Get smoking risk multiplier"""
        status_key = smoking_status.lower().replace(" ", "_")

        if status_key not in self._config['smoking_risk']:
            return 1.0

        cancer_data = self._config['smoking_risk'][status_key].get(cancer_type)

        if not cancer_data:
            return 1.0

        # Handle pack-year ranges
        if isinstance(cancer_data, list):
            for range_data in cancer_data:
                min_py, max_py = range_data['pack_years']
                if min_py <= pack_years < max_py:
                    return range_data['multiplier']
            return 1.0
        else:
            return cancer_data.get('multiplier', 1.0)

    def get_family_history_risk(self, cancer_type: str,
                                min_age: int, count: int) -> float:
        """Get family history risk multiplier"""
        fh_data = self._config['family_history_risk'].get(cancer_type)

        if not fh_data:
            return 1.3  # default

        # Check for age-based multipliers
        if 'age_threshold' in fh_data:
            threshold = fh_data['age_threshold']
            if min_age < threshold:
                base = fh_data['base_multiplier_early']
            else:
                base = fh_data['base_multiplier_late']
        else:
            base = fh_data['base_multiplier']

        # Diminishing returns for multiple relatives
        family_factor = 1.0 + 0.2 * min(count - 1, 2)

        return base * family_factor

    def get_genetic_mutation_risk(self, mutation: str,
                                   cancer_type: str) -> float:
        """Get genetic mutation risk multiplier"""
        mutations = self._config['genetic_mutations']
        return mutations.get(mutation, {}).get(cancer_type, 1.0)

    def get_personal_history_risk(self) -> float:
        """Get personal history risk multiplier"""
        return self._config['personal_history']['multiplier']

    def get_safety_limits(self) -> Dict[str, float]:
        """Get safety limit values"""
        return {
            'max_individual_risk': self._config['max_individual_risk'],
            'max_overall_risk': self._config['max_overall_risk'],
            'max_risk_multiplier': self._config['max_risk_multiplier']
        }
```

**Step 4: Refactor get_risk_multiplier_fixed()**

Update `streamlit_app.py`:
```python
from config.config_loader import RiskConfig

# At module level
risk_config = RiskConfig()

def get_risk_multiplier_fixed(cancer_type, smoking_status, pack_years,
                              family_history, family_ages, genetic_mutations,
                              personal_history):
    """Calculate risk multiplier using configuration-driven approach"""
    risk_factors = []  # Log-scale risk factors

    # Smoking risk
    smoking_mult = risk_config.get_smoking_risk(
        smoking_status, cancer_type, pack_years
    )
    if smoking_mult > 1.0:
        risk_factors.append(math.log(smoking_mult))

    # Family history
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
        if family_ages_list:
            min_age = min(family_ages_list)
            family_count = len(family_ages_list)

            family_mult = risk_config.get_family_history_risk(
                cancer_type, min_age, family_count
            )
            risk_factors.append(math.log(family_mult))

    # Genetic mutations
    for mutation in genetic_mutations:
        genetic_mult = risk_config.get_genetic_mutation_risk(
            mutation, cancer_type
        )
        if genetic_mult > 1.0:
            risk_factors.append(math.log(genetic_mult))

    # Personal history
    if personal_history:
        personal_mult = risk_config.get_personal_history_risk()
        risk_factors.append(math.log(personal_mult))

    # Combine and apply safety limits
    if not risk_factors:
        return 1.0

    total_log_risk = sum(risk_factors)
    final_multiplier = math.exp(total_log_risk)

    # Get safety limit from config
    max_mult = risk_config.get_safety_limits()['max_risk_multiplier']
    return min(final_multiplier, max_mult)
```

### Implementation Priority: **Week 1, Day 3-5**

### Testing:
- Verify all risk calculations match previous values
- Test edge cases (multiple risk factors)
- Validate YAML parsing
- Test with missing config values (should use defaults)

### Benefits:
- Medical experts can update risks without code changes
- Clear documentation of data sources
- Version control of medical data
- Easier to add new risk factors
- Testable configuration

---

## H3: Hardcoded Risk Thresholds

**Location:** Lines 325, 348, 359, 376, 430, 687-695
**Risk:** Medium Maintainability Risk
**Effort:** 2 hours

### Problem:
Critical thresholds are scattered throughout the code:
- `0.05` - Low risk threshold
- `0.15` - Moderate risk threshold
- `0.60` - Individual cancer risk cap
- `0.85` - Overall risk cap
- `25.0` - Max risk multiplier

### Remediation Steps:

**Add to `config/risk_factors.yaml`:**
```yaml
# Risk classification thresholds
risk_thresholds:
  low: 0.05      # < 5% = low risk
  moderate: 0.15 # < 15% = moderate risk
  high: 0.15     # >= 15% = high risk

# Safety caps (clinical reasonableness limits)
safety_limits:
  max_individual_cancer_risk: 0.60  # Cap individual cancer at 60%
  max_overall_cancer_risk: 0.85     # Cap overall cancer at 85%
  max_risk_multiplier: 25.0         # Cap combined multipliers at 25x

# Display thresholds
display_thresholds:
  minimum_risk_to_show: 0.001  # 0.1% - hide risks below this
  high_risk_recommendation: 0.10  # 10% - strongly recommend screening
  moderate_risk_recommendation: 0.05  # 5% - recommend screening
  low_risk_recommendation: 0.02  # 2% - consider screening
```

**Create constants file `config/constants.py`:**
```python
"""Application constants loaded from configuration"""
from config.config_loader import RiskConfig

config = RiskConfig()

# Risk classification
RISK_LOW_THRESHOLD = config.config['risk_thresholds']['low']
RISK_MODERATE_THRESHOLD = config.config['risk_thresholds']['moderate']

# Safety limits
MAX_INDIVIDUAL_RISK = config.config['safety_limits']['max_individual_cancer_risk']
MAX_OVERALL_RISK = config.config['safety_limits']['max_overall_cancer_risk']
MAX_RISK_MULTIPLIER = config.config['safety_limits']['max_risk_multiplier']

# Display thresholds
MIN_RISK_TO_SHOW = config.config['display_thresholds']['minimum_risk_to_show']
HIGH_RISK_THRESHOLD = config.config['display_thresholds']['high_risk_recommendation']
MODERATE_RISK_THRESHOLD = config.config['display_thresholds']['moderate_risk_recommendation']
LOW_RISK_THRESHOLD = config.config['display_thresholds']['low_risk_recommendation']

# Risk level colors
RISK_COLORS = {
    'low': '#22c55e',      # green
    'moderate': '#f59e0b', # amber
    'high': '#ef4444'      # red
}
```

**Update code to use constants:**
```python
from config.constants import (
    RISK_LOW_THRESHOLD, RISK_MODERATE_THRESHOLD,
    MAX_INDIVIDUAL_RISK, MAX_OVERALL_RISK,
    MIN_RISK_TO_SHOW, RISK_COLORS
)

# Replace line 687-695
if overall_prevalence < RISK_LOW_THRESHOLD:
    risk_level = "Low"
    risk_color = RISK_COLORS['low']
elif overall_prevalence < RISK_MODERATE_THRESHOLD:
    risk_level = "Moderate"
    risk_color = RISK_COLORS['moderate']
else:
    risk_level = "High"
    risk_color = RISK_COLORS['high']
```

### Implementation Priority: **Week 1, Day 5**

---

## H4: Incomplete Test Coverage

**Location:** TEST_PERFORMANCE dictionary (lines 93-178)
**Risk:** Medium Accuracy Risk
**Effort:** 3-4 hours

### Problem:
Most screening tests only cover 1-8 cancer types out of 24:
- Colonoscopy: 1 cancer type
- PSA Test: 1 cancer type
- Mammography: 1 cancer type
- Upper Endoscopy: 2 cancer types
- Low-dose CT: 8 cancer types
- Whole-body MRI: 24 cancer types ✓
- Galleri: 24 cancer types ✓

### Remediation Steps:

**Option 1: Add Missing Data (Requires Medical Research)**

Research and add missing test performance data. Example:

```python
"Mammography": {
    "breast": {"sensitivity": 0.85, "specificity": 0.90},
    # Research and add:
    # - Can mammography detect lung nodules? (historical data exists)
    # - What about incidental findings?
},
```

**Option 2: Document Limitations (Short-term)**

Add metadata to clarify test scope:

```python
TEST_PERFORMANCE = {
    "Colonoscopy": {
        "colorectal": {"sensitivity": 0.95, "specificity": 0.90},
        "_metadata": {
            "description": "Colonoscopy is specific to colorectal cancer screening",
            "note": "Cannot detect other cancer types",
            "data_source": "USPSTF 2021",
            "last_updated": "2025-01-15"
        }
    },
}
```

**Option 3: UI Warning (Immediate Fix)**

Add warning when test has limited coverage:

```python
# In screening test section
if tests:
    for test in tests:
        coverage = len(TEST_PERFORMANCE.get(test, {}))
        total_cancers = len([ct for ct in CANCER_INCIDENCE.keys()
                           if not ((ct in ["prostate", "testicular"] and sex == "female") or
                                 (ct in ["ovarian", "cervical", "endometrial", "uterine"] and sex == "male"))])

        coverage_pct = (coverage / total_cancers * 100) if total_cancers > 0 else 0

        if coverage_pct < 20:
            st.warning(f"⚠️ {test} is highly specific and only screens for {coverage} cancer type(s) ({coverage_pct:.0f}% coverage)")
```

### Implementation Priority: **Week 2, Day 1-2**

### Recommended Approach:
1. Implement Option 3 (UI warning) immediately
2. Add metadata (Option 2) for all tests
3. Research and add missing data (Option 1) over time

---

## H5: Light Theme Hardcoding (Accessibility Issue)

**Location:** Lines 17-54 (CSS enforcement)
**Risk:** Low WCAG Compliance Risk
**Effort:** 2-3 hours

### Problem:
Forces light theme, ignoring user's system preferences and accessibility needs (WCAG 2.1 SC 1.4.11).

### Remediation Steps:

**Option 1: Respect System Preferences (Recommended)**

Remove forced theme and use `.streamlit/config.toml`:

```toml
[theme]
# Light theme (default)
primaryColor = "#1f2937"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f9fafb"
textColor = "#1f2937"
font = "sans serif"

# Note: Users can override with dark mode in browser
```

Remove CSS override from `streamlit_app.py`.

**Option 2: Add Theme Toggle**

```python
# Add to sidebar
theme_preference = st.sidebar.radio(
    "Theme Preference",
    ["Light (Default)", "Dark", "System"],
    index=0
)

if theme_preference == "Dark":
    st.markdown("""
    <style>
        .stApp {
            background-color: #1f2937 !important;
            color: #f9fafb !important;
        }
    </style>
    """, unsafe_allow_html=True)
elif theme_preference == "System":
    # Use CSS prefers-color-scheme
    st.markdown("""
    <style>
        @media (prefers-color-scheme: dark) {
            .stApp {
                background-color: #1f2937 !important;
                color: #f9fafb !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)
```

**Option 3: Document Rationale**

If light theme is medically important (e.g., color accuracy for medical data), document this:

```python
# Medical rationale for light theme
# Light theme is enforced to ensure:
# 1. Consistent color representation for risk levels (red/amber/green)
# 2. Clinical data readability
# 3. Sankey diagram color accuracy
# Reference: ISO 13485:2016 Medical Device Color Standards
```

### Implementation Priority: **Week 2, Day 3**

### Recommendation:
Use Option 1 (respect preferences) unless there's a medical rationale, then use Option 3.

---

# MEDIUM PRIORITY ISSUES (9)

## M1: Missing Type Hints

**Location:** 7 functions without type hints
**Effort:** 2-3 hours

### Functions Needing Type Hints:
1. `get_risk_multiplier_fixed()` (line 216)
2. `calculate_overall_prevalence_fixed()` (line 327)
3. `calculate_per_cancer_prevalence_fixed()` (line 361)
4. `combine_tests()` (line 387)
5. `validate_inputs()` (line 449)
6. `create_sankey_diagram()` (line 471)
7. `interpolate_incidence()` - already has @st.cache_data

### Remediation:

Add type hints using Python 3.9+ syntax:

```python
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go

def get_risk_multiplier_fixed(
    cancer_type: str,
    smoking_status: str,
    pack_years: int,
    family_history: List[str],
    family_ages: Dict[str, List[int]],
    genetic_mutations: List[str],
    personal_history: bool
) -> float:
    """Calculate risk multiplier using logarithmic combination"""
    # ... implementation

def calculate_overall_prevalence_fixed(
    age: int,
    sex: str,
    risk_multipliers: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float]]:
    """Calculate overall cancer prevalence with proper probability combination"""
    # ... implementation

def calculate_per_cancer_prevalence_fixed(
    age: int,
    sex: str,
    risk_multipliers: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """Calculate personalized risk for each cancer type"""
    # ... implementation

def combine_tests(
    tests: List[str],
    mode: str,
    age: int,
    sex: str,
    risk_multipliers: Dict[str, float]
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    """Combine multiple tests with proper probability handling"""
    # ... implementation

def validate_inputs(
    age: int,
    smoking_status: str,
    pack_years: int,
    family_ages: Dict[str, List[int]]
) -> Tuple[List[str], List[str]]:
    """Validate user inputs and return error messages"""
    # ... implementation

def create_sankey_diagram(
    population: int,
    overall_prevalence: float,
    sens: float,
    spec: float,
    adjusted_biopsy_rate: float,
    comp_rate: float
) -> go.Figure:
    """Create the Sankey diagram with fixed positioning"""
    # ... implementation
```

### Implementation Priority: **Week 2, Day 4**

### Benefits:
- Better IDE autocomplete
- Early error detection
- Self-documenting code
- Easier refactoring

---

## M2: Missing Input Validation

**Location:** Various user input sections
**Effort:** 3 hours

### Current Gaps:
- Age slider (validated ✓)
- Pack-years slider (validated ✓)
- Family ages (validated ✓)
- **Missing:** Test selection validation
- **Missing:** Risk multiplier sanity checks
- **Missing:** Prevalence calculation validation

### Remediation:

Add comprehensive validation:

```python
def validate_test_selection(tests: List[str], sex: str) -> Tuple[List[str], List[str]]:
    """Validate test selections are appropriate for user"""
    errors = []
    warnings = []

    # Check sex-specific tests
    if "PSA Test" in tests and sex == "female":
        errors.append("PSA Test is only applicable for males (prostate screening)")

    if "Mammography" in tests and sex == "male":
        warnings.append("Mammography for males is uncommon (only for gynecomastia/breast cancer risk)")

    if "HPV Test" in tests and sex == "male":
        errors.append("HPV Test (cervical screening) is only applicable for females")

    # Check for redundant tests
    if "Whole-body MRI" in tests and len(tests) > 1:
        warnings.append("Whole-body MRI covers most cancers - other tests may be redundant")

    if "Galleri Blood Test" in tests and len(tests) > 1:
        warnings.append("Galleri multi-cancer test - combining with specific tests may not add value")

    return errors, warnings

def validate_risk_calculations(
    prevalence: float,
    risk_multiplier: float,
    cancer_type: str
) -> None:
    """Validate risk calculation results are reasonable"""
    if prevalence < 0 or prevalence > 1:
        raise ValueError(f"Invalid prevalence for {cancer_type}: {prevalence}")

    if risk_multiplier < 0.1 or risk_multiplier > 100:
        raise ValueError(f"Extreme risk multiplier for {cancer_type}: {risk_multiplier}")

    if prevalence > 0.9:
        import warnings
        warnings.warn(f"Extremely high prevalence for {cancer_type}: {prevalence*100:.1f}%")
```

Use in code:

```python
# After test selection
if tests:
    test_errors, test_warnings = validate_test_selection(tests, sex)
    if test_errors:
        for error in test_errors:
            st.error(error)
    if test_warnings:
        for warning in test_warnings:
            st.warning(warning)
```

### Implementation Priority: **Week 2, Day 5**

---

## M3: Hardcoded Population Size

**Location:** Lines 634-635
**Effort:** 1 hour

### Current Code:
```python
per_thousand = st.checkbox("Show per 1000 people", value=False)
population = 1000 if per_thousand else 100
```

### Problem:
Only supports 100 or 1000. Users may want other values for better understanding.

### Remediation:

```python
# Replace checkbox with options
population_display = st.radio(
    "Natural Frequency Display",
    ["Per 100 people", "Per 1,000 people", "Per 10,000 people", "Custom"],
    index=0,
    help="Choose how to display frequencies"
)

if population_display == "Custom":
    population = st.number_input(
        "Custom population size",
        min_value=10,
        max_value=1000000,
        value=100,
        step=10,
        help="Enter any population size for frequency display"
    )
else:
    population_map = {
        "Per 100 people": 100,
        "Per 1,000 people": 1000,
        "Per 10,000 people": 10000
    }
    population = population_map[population_display]
```

### Implementation Priority: **Week 3, Day 1**

---

## M4: Missing Logging

**Location:** Throughout application
**Effort:** 2-3 hours

### Current State:
No logging for debugging or error tracking.

### Remediation:

Create `utils/logging_config.py`:

```python
import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = "app.log"):
    """Configure application logging"""

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger

# Usage in streamlit_app.py
import logging
from utils.logging_config import setup_logging

logger = setup_logging()

# Add logging throughout
logger.info(f"User started session: age={age}, sex={sex}")
logger.info(f"Selected tests: {tests}")
logger.debug(f"Risk multipliers calculated: {risk_multipliers}")

try:
    overall_prevalence, individual_risks = calculate_overall_prevalence_fixed(
        age, sex, risk_multipliers
    )
    logger.info(f"Overall prevalence calculated: {overall_prevalence:.4f}")
except Exception as e:
    logger.error(f"Error calculating prevalence: {e}", exc_info=True)
    st.error("An error occurred during risk calculation. Please try again.")
```

### Implementation Priority: **Week 3, Day 1**

---

## M5: Hardcoded Test Costs

**Location:** Lines 961-965
**Effort:** 1 hour

### Current Code:
```python
cost_per_test = {"Mammography": 150, "Colonoscopy": 800, ...}
```

### Remediation:

Move to `config/test_parameters.yaml`:

```yaml
test_costs:
  # All costs in USD
  # Source: Medicare 2025 reimbursement rates
  # Last updated: 2025-01-15
  "Mammography": 150
  "Colonoscopy": 800
  "Low-dose CT Scan": 300
  "PSA Test": 50
  "Whole-body MRI": 2000
  "Galleri Blood Test": 1000
  "Upper Endoscopy": 750
  "Dermoscopy": 200
  "HPV Test": 75
  "Skin Exam": 100

# Average costs for procedures
procedure_costs:
  biopsy_average: 1500
  follow_up_imaging: 500
  specialist_consultation: 250
```

Update config loader and use constants.

### Implementation Priority: **Week 3, Day 2**

---

## M6: Data Timestamp Issue

**Location:** Line 63
**Effort:** 30 minutes

### Current Code:
```python
st.markdown("Enter your details... Data as of July 2025.")
```

### Problem:
This is a future date (written in 2024), creates confusion.

### Remediation:

**Option 1: Use Actual Data Date**
```python
DATA_VERSION = "2024-12"  # Move to config
st.markdown(f"Enter your details... Data as of {DATA_VERSION}.")
```

**Option 2: Add Version Info to Config**

Create `config/metadata.yaml`:
```yaml
data_version:
  cancer_incidence_data: "2024-12"
  test_performance_data: "2024-12"
  risk_multiplier_data: "2024-12"

data_sources:
  cancer_incidence: "SEER 2024 Database"
  test_performance: "Medical literature meta-analysis 2024"
  risk_factors: "USPSTF 2024, various studies"

last_updated: "2024-12-15"
version: "1.0.0"
```

```python
from config.metadata import get_data_version

st.markdown(f"Enter your details... Data as of {get_data_version()}.")
```

### Implementation Priority: **Week 3, Day 2**

---

## M7: Missing Risk Factors

**Location:** Risk calculation logic
**Effort:** 1-2 days (requires research)

### Current Risk Factors:
- Age ✓
- Sex ✓
- Smoking ✓
- Family history ✓
- Genetic mutations ✓
- Personal history ✓

### Missing Important Factors:
- Alcohol consumption (liver, esophageal, breast cancer)
- BMI/obesity (endometrial, kidney, liver, breast cancer)
- Physical activity (colorectal, breast cancer)
- Diet (colorectal, gastric cancer)
- Environmental exposures (lung, bladder cancer)
- Diabetes (pancreatic, liver cancer)
- HPV status (cervical, head/neck cancer)

### Remediation (Long-term):

Add to sidebar:
```python
# Additional risk factors
st.subheader("Additional Risk Factors (Optional)")

# Alcohol
alcohol = st.selectbox(
    "Alcohol Consumption",
    ["None", "Light (1-7 drinks/week)", "Moderate (8-14/week)",
     "Heavy (15+/week)"],
    help="Average drinks per week"
)

# BMI
bmi_category = st.selectbox(
    "BMI Category",
    ["Underweight (<18.5)", "Normal (18.5-24.9)",
     "Overweight (25-29.9)", "Obese (30+)"],
    index=1
)

# Physical activity
activity = st.selectbox(
    "Physical Activity Level",
    ["Sedentary", "Light", "Moderate", "Very Active"],
    help="Average weekly exercise level"
)
```

Add research-backed multipliers to config.

### Implementation Priority: **Week 3, Day 3-5** (Phase 2)

---

## M8: Performance - Limited Caching

**Location:** Only 1 function cached
**Effort:** 1-2 hours

### Current State:
Only `interpolate_incidence()` uses `@st.cache_data`

### Remediation:

Add caching to expensive calculations:

```python
@st.cache_data
def calculate_overall_prevalence_fixed(age, sex, risk_multipliers=None):
    """Calculate overall cancer prevalence"""
    # ... existing implementation

@st.cache_data
def calculate_per_cancer_prevalence_fixed(age, sex, risk_multipliers=None):
    """Calculate per-cancer prevalence"""
    # ... existing implementation

@st.cache_data(hash_funcs={dict: lambda x: str(sorted(x.items()))})
def combine_tests(tests_tuple, mode, age, sex, risk_multipliers_tuple):
    """Combine tests - convert lists/dicts to tuples for hashing"""
    tests = list(tests_tuple)
    risk_multipliers = dict(risk_multipliers_tuple) if risk_multipliers_tuple else None
    # ... existing implementation

# When calling combine_tests:
sens, spec, cancer_outcomes = combine_tests(
    tuple(tests),  # Convert list to tuple
    mode_short,
    age,
    sex,
    tuple(risk_multipliers.items()) if risk_multipliers else None
)
```

### Implementation Priority: **Week 3, Day 2**

### Testing:
- Verify cache hits with repeated inputs
- Test cache invalidation when inputs change
- Measure performance improvement

---

## M9: Dictionary Operations in Loops

**Location:** Lines 387-447 (combine_tests function)
**Effort:** 2 hours

### Current Code:
```python
for cancer_type in applicable_cancers:  # O(n)
    for test in applicable_tests:  # O(m)
        if test in TEST_PERFORMANCE and cancer_type in TEST_PERFORMANCE[test]:  # Dict lookup
```

### Problem:
O(n*m) complexity with repeated dictionary lookups.

### Remediation:

Optimize with set operations and pre-computed lookups:

```python
@st.cache_data
def build_test_coverage_index():
    """Pre-compute test coverage for fast lookups"""
    coverage = {}
    for test_name, test_data in TEST_PERFORMANCE.items():
        coverage[test_name] = set(test_data.keys()) - {'_metadata'}
    return coverage

# Module level
TEST_COVERAGE_INDEX = build_test_coverage_index()

def combine_tests(tests, mode, age, sex, risk_multipliers):
    """Optimized version with pre-computed lookups"""
    cancer_outcomes = {}

    # Get applicable cancers (optimize with set)
    all_cancers = set(CANCER_INCIDENCE.keys())
    sex_inappropriate = set()
    if sex == "female":
        sex_inappropriate = {"prostate", "testicular"}
    else:
        sex_inappropriate = {"ovarian", "cervical", "endometrial", "uterine"}

    applicable_cancers = all_cancers - sex_inappropriate

    # Pre-compute test coverage union
    tests_set = set(tests)
    covered_cancers = set()
    for test in tests_set:
        covered_cancers |= TEST_COVERAGE_INDEX.get(test, set())

    # Only process cancers that can be detected
    cancers_to_process = applicable_cancers & covered_cancers

    for cancer_type in cancers_to_process:
        sens_combined = 0 if mode == "Parallel" else 1
        spec_combined = 1 if mode == "Parallel" else 0

        # Find applicable tests for this cancer
        for test in tests_set:
            if cancer_type in TEST_COVERAGE_INDEX.get(test, set()):
                perf = TEST_PERFORMANCE[test][cancer_type]
                sens = perf["sensitivity"]
                spec = perf["specificity"]

                if mode == "Parallel":
                    sens_combined = 1 - (1 - sens_combined) * (1 - sens)
                    spec_combined *= spec
                else:
                    sens_combined *= sens
                    spec_combined = 1 - (1 - spec_combined) * (1 - spec)

        # Calculate prevalence
        incidence_rate = interpolate_incidence(age, sex, cancer_type)
        prevalence = (incidence_rate / 100000) * 10

        if risk_multipliers and cancer_type in risk_multipliers:
            prevalence *= risk_multipliers[cancer_type]

        prevalence = min(prevalence, 0.60)

        cancer_outcomes[cancer_type] = {
            "sensitivity": sens_combined,
            "specificity": spec_combined,
            "prevalence": prevalence
        }

    # Calculate aggregates
    if cancer_outcomes:
        overall_sens = sum(co["sensitivity"] for co in cancer_outcomes.values()) / len(cancer_outcomes)
        overall_spec = sum(co["specificity"] for co in cancer_outcomes.values()) / len(cancer_outcomes)
    else:
        overall_sens = 0
        overall_spec = 1

    return overall_sens, overall_spec, cancer_outcomes
```

### Implementation Priority: **Week 3, Day 3**

### Benefits:
- Reduced time complexity
- Faster execution for multiple tests
- Better scalability

---

# LOW PRIORITY ISSUES (7)

## L1: Monolithic File Structure

**Effort:** 1-2 days
**Priority:** Code Organization

### Current State:
Single 1,072-line file

### Recommended Structure:
```
cancer_test_performance_viz/
├── streamlit_app.py           # Main app (200 lines)
├── config/
│   ├── __init__.py
│   ├── config_loader.py       # Configuration management
│   ├── constants.py           # Application constants
│   ├── risk_factors.yaml      # Risk multiplier data
│   ├── test_parameters.yaml   # Test performance & costs
│   └── metadata.yaml          # Data versioning
├── data/
│   ├── __init__.py
│   ├── cancer_incidence.py    # CANCER_INCIDENCE data
│   ├── test_performance.py    # TEST_PERFORMANCE data
│   └── downstream_risks.py    # DOWNSTREAM_RISKS data
├── models/
│   ├── __init__.py
│   ├── risk_calculator.py     # Risk calculation logic
│   └── test_combiner.py       # Test combination logic
├── ui/
│   ├── __init__.py
│   ├── sidebar.py             # Sidebar components
│   ├── risk_display.py        # Risk visualization
│   └── sankey.py              # Sankey diagram
├── utils/
│   ├── __init__.py
│   ├── validation.py          # Input validation
│   └── logging_config.py      # Logging setup
├── tests/
│   ├── test_risk_calculator.py
│   ├── test_validation.py
│   └── test_config_loader.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── README.md
```

### Implementation Priority: **Week 4** (after functional improvements)

---

## L2: Missing Docstrings

**Effort:** 2-3 hours
**Priority:** Documentation

Add comprehensive docstrings:

```python
def get_risk_multiplier_fixed(
    cancer_type: str,
    smoking_status: str,
    pack_years: int,
    family_history: List[str],
    family_ages: Dict[str, List[int]],
    genetic_mutations: List[str],
    personal_history: bool
) -> float:
    """
    Calculate personalized risk multiplier for a specific cancer type.

    Uses logarithmic combination of risk factors to prevent extreme values.
    All multipliers are combined on log scale and capped at 25x maximum.

    Args:
        cancer_type: Cancer type key (e.g., 'lung', 'breast', 'colorectal')
        smoking_status: One of 'Never smoked', 'Former smoker', 'Current smoker'
        pack_years: Cumulative smoking exposure (packs per day × years)
        family_history: List of family cancer histories (e.g., ['Breast cancer'])
        family_ages: Dict mapping cancer types to list of diagnosis ages
        genetic_mutations: List of known mutations (e.g., ['BRCA1'])
        personal_history: Whether patient has prior cancer diagnosis

    Returns:
        Risk multiplier (1.0 = average risk, >1.0 = increased risk)

    Examples:
        >>> get_risk_multiplier_fixed('lung', 'Current smoker', 30, [], {}, [], False)
        15.0  # 30 pack-years = 15x risk for lung cancer

        >>> get_risk_multiplier_fixed('breast', 'Never smoked', 0,
        ...                           ['Breast cancer'], {'Breast cancer': [45]},
        ...                           ['BRCA1'], False)
        17.6  # Combined family history + BRCA1

    Notes:
        - Risk factors are combined on log scale to prevent multiplication overflow
        - Individual factors capped at reasonable medical values
        - Final multiplier capped at 25.0 for clinical reasonableness
        - Uses diminishing returns for multiple family members (max 1.4x bonus)

    References:
        - Smoking risks: Surgeon General Report 2014
        - BRCA risks: Kuchenbaecker et al. JAMA 2017
        - Lynch syndrome: Win et al. JAMA Oncol 2017
    """
    # Implementation...
```

### Implementation Priority: **Week 4, Day 1-2**

---

## L3-L7: Additional Low Priority Items

**L3: No Unit Tests** - Add pytest suite (3-4 days)
**L4: No Configuration File** - Already covered in H2 (bundled)
**L5: Missing Accessibility (ARIA)** - Add ARIA labels (2-3 hours)
**L6: Repeated Constants** - Consolidate with H3 (bundled)
**L7: Inconsistent Naming** - Refactor with L1 (bundled)

---

# IMPLEMENTATION ROADMAP

## Week 1: High Priority Security & Configuration (5 days)

**Day 1-2:** H1 - Remove unsafe HTML, migrate to native components
**Day 3-4:** H2 - Extract risk multipliers to YAML configuration
**Day 5:** H3 - Extract thresholds to configuration

**Deliverables:**
- No `unsafe_allow_html` in codebase
- `config/risk_factors.yaml` complete
- `config/config_loader.py` working
- All tests passing

---

## Week 2: High Priority Quality & Medium Priority Fixes (5 days)

**Day 1-2:** H4 - Add test coverage warnings, metadata
**Day 3:** H5 - Theme accessibility improvements
**Day 4:** M1 - Add type hints to all functions
**Day 5:** M2 - Add comprehensive input validation

**Deliverables:**
- Test coverage warnings visible to users
- Type hints on all functions
- Enhanced input validation
- Theme respects user preferences

---

## Week 3: Medium Priority Enhancements (5 days)

**Day 1:** M3 - Flexible population display + M4 - Logging
**Day 2:** M5 - Extract test costs + M6 - Fix timestamps + M8 - Performance
**Day 3:** M9 - Optimize dictionary operations
**Day 4-5:** M7 - Research and add additional risk factors (Phase 1)

**Deliverables:**
- Logging infrastructure in place
- All hardcoded values in configuration
- Performance improvements measured
- 2-3 additional risk factors added

---

## Week 4: Low Priority & Code Quality (5 days)

**Day 1-2:** L2 - Add comprehensive docstrings
**Day 3-5:** L1 - Refactor to modular structure
**Day 5:** L3 - Basic unit test suite

**Deliverables:**
- Well-documented codebase
- Modular file structure
- Basic test coverage (>50%)
- Final code review

---

# TESTING STRATEGY

## Unit Tests (pytest)

```python
# tests/test_risk_calculator.py
def test_get_risk_multiplier_smoking():
    """Test smoking risk calculations"""
    # Never smoker
    assert get_risk_multiplier_fixed('lung', 'Never smoked', 0, [], {}, [], False) == 1.0

    # Current smoker, high pack-years
    mult = get_risk_multiplier_fixed('lung', 'Current smoker', 50, [], {}, [], False)
    assert 15 <= mult <= 25  # Should be high but capped

def test_risk_cap():
    """Test that extreme risk factors are capped at 25x"""
    # Pile on every risk factor
    mult = get_risk_multiplier_fixed(
        'breast',
        'Current smoker',
        80,  # High smoking
        ['Breast cancer'],
        {'Breast cancer': [35]},  # Young diagnosis
        ['BRCA1', 'BRCA2', 'TP53 (Li-Fraumeni)'],  # Multiple mutations
        True  # Personal history
    )
    assert mult <= 25.0  # Should hit cap

def test_prevalence_bounds():
    """Test prevalence stays between 0 and 1"""
    prev, _ = calculate_overall_prevalence_fixed(70, 'male', None)
    assert 0 <= prev <= 1
```

## Integration Tests

```python
# tests/test_integration.py
def test_full_workflow():
    """Test complete risk calculation workflow"""
    # High-risk profile
    risk_mults = {ct: get_risk_multiplier_fixed(
        ct, 'Current smoker', 40,
        ['Lung cancer'], {'Lung cancer': [55]},
        [], False
    ) for ct in CANCER_INCIDENCE.keys()}

    prev, risks = calculate_overall_prevalence_fixed(65, 'male', risk_mults)

    assert prev > 0.15  # Should be high risk
    assert risks['lung'] > risks['brain']  # Lung should be highest
```

## Configuration Tests

```python
# tests/test_config.py
def test_config_loads():
    """Test configuration loads without errors"""
    config = RiskConfig()
    assert config.config is not None
    assert 'smoking_risk' in config.config

def test_smoking_risk_lookup():
    """Test configuration lookups work"""
    config = RiskConfig()
    mult = config.get_smoking_risk('Current smoker', 'lung', 30)
    assert mult > 1.0
```

## Manual Testing Checklist

- [ ] Enter various age/sex combinations
- [ ] Test all screening test selections
- [ ] Verify Sankey diagram renders correctly
- [ ] Test with extreme risk factor combinations
- [ ] Verify error messages display correctly
- [ ] Test on mobile devices
- [ ] Test with screen readers (accessibility)
- [ ] Verify theme works in light/dark mode
- [ ] Load test with multiple concurrent users
- [ ] Test with slow network (check performance)

---

# SUCCESS METRICS

## Code Quality Metrics

- [ ] Zero uses of `unsafe_allow_html` (or justified and documented)
- [ ] 100% of functions have type hints
- [ ] 100% of functions have docstrings
- [ ] 80%+ test coverage
- [ ] Zero hardcoded magic numbers
- [ ] Zero bare except clauses
- [ ] Logging in all critical paths

## Performance Metrics

- [ ] Page load time < 3 seconds
- [ ] Risk calculation < 500ms
- [ ] Sankey diagram render < 1 second
- [ ] Cache hit rate > 80% for repeated inputs

## User Experience Metrics

- [ ] Zero validation errors slip through
- [ ] All medical data sourced and cited
- [ ] Accessibility score (WAVE) > 95%
- [ ] Mobile responsive (all screen sizes)
- [ ] Clear error messages for all input errors

## Maintainability Metrics

- [ ] Medical data updateable without code changes
- [ ] Configuration changes require only YAML edits
- [ ] New risk factors addable in < 1 hour
- [ ] New screening tests addable in < 2 hours
- [ ] Onboarding time for new developer < 4 hours

---

# RISK MITIGATION

## Risks During Implementation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Breaking existing functionality | High | Medium | Comprehensive testing before each merge |
| Configuration errors | Medium | Medium | Schema validation for YAML files |
| Performance regression | Low | Low | Benchmark before/after changes |
| Data accuracy issues | High | Low | Medical review of all configuration changes |
| User confusion from UI changes | Medium | Medium | A/B testing, gradual rollout |

## Rollback Plan

1. All changes in feature branches
2. Git tag before each major change
3. Keep old code commented (temporarily)
4. Ability to toggle new features via config

---

# MAINTENANCE PLAN

## Regular Updates Needed

**Monthly:**
- Review medical literature for updated risk factors
- Check for new screening tests
- Update test performance data if new studies available

**Quarterly:**
- Update cancer incidence data (SEER releases)
- Review and update test costs
- Dependency security updates

**Annually:**
- Major medical data refresh
- Comprehensive code audit
- User feedback review and prioritization

## Documentation to Maintain

- `README.md` - User guide
- `CONTRIBUTING.md` - Developer guide
- `CHANGELOG.md` - Version history
- `config/README.md` - Configuration guide
- Inline code comments
- Medical data sources and citations

---

# APPENDIX A: Configuration Schema

## risk_factors.yaml Schema

```yaml
smoking_risk:
  <smoking_status>:
    <cancer_type>:
      - pack_years: [min, max]
        multiplier: float
      # OR
      multiplier: float

family_history_risk:
  <cancer_type>:
    base_multiplier: float
    # OR
    base_multiplier_early: float
    base_multiplier_late: float
    age_threshold: int

genetic_mutations:
  <mutation_name>:
    <cancer_type>: float

personal_history:
  multiplier: float

risk_thresholds:
  low: float
  moderate: float
  high: float

safety_limits:
  max_individual_cancer_risk: float
  max_overall_cancer_risk: float
  max_risk_multiplier: float
```

---

# APPENDIX B: Testing Infrastructure Setup

## Install Testing Dependencies

```bash
pip install pytest pytest-cov pytest-mock hypothesis
```

Update `requirements.txt`:
```
# Testing (dev only)
pytest>=7.4.0,<8.0.0
pytest-cov>=4.1.0,<5.0.0
pytest-mock>=3.11.0,<4.0.0
hypothesis>=6.92.0,<7.0.0
```

## Run Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test file
pytest tests/test_risk_calculator.py

# Verbose output
pytest -v
```

---

# APPENDIX C: Pre-Commit Hooks (Recommended)

Install pre-commit:
```bash
pip install pre-commit
```

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/PyCQA/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100', '--ignore=E203,W503']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML]
```

Install hooks:
```bash
pre-commit install
```

---

# QUESTIONS FOR STAKEHOLDER

Before implementation, please clarify:

1. **Medical Data Sources**: Do you have specific medical literature we should cite for risk multipliers?

2. **Theme Preference**: Is there a clinical reason for forcing light theme, or can we make it user-configurable?

3. **Test Coverage**: Should we add missing test performance data through research, or document limitations?

4. **Additional Risk Factors**: Which additional risk factors are most important to add (alcohol, BMI, etc.)?

5. **Timeline**: Is 3-week implementation timeline acceptable, or do you need accelerated delivery?

6. **Testing Support**: Do you have medical domain experts available to validate configuration changes?

7. **Deployment**: What's the deployment environment (local, cloud, both)?

8. **User Base**: Who are the primary users (patients, clinicians, educators)?

---

**End of Remediation Plan**

This plan should be treated as a living document and updated as implementation progresses.
