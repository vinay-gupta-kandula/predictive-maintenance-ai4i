"""
AI-Driven Predictive Maintenance Dashboard
===========================================
A Streamlit application for monitoring machine health and predicting failures
using sensor and operational data with explainable AI.

Author: Vinay Gupta Kandula
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
import time
import os
import json

warnings.filterwarnings('ignore')

# ========================================
# CONFIGURATION
# ========================================

class Config:
    """Centralized configuration for dashboard."""
    
    # Paths - Use pathlib for cross-platform compatibility
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    MODEL_DIR = PROJECT_ROOT / "models"
    
    # Data files
    ENGINEERED_DATA = PROCESSED_DIR / "ai4i2020_features.csv"
    RAW_DATA = RAW_DIR / "ai4i2020.csv"
    
    # Model files
    MODEL_PATH = MODEL_DIR / "best_model.joblib"
    IMPUTER_PATH = MODEL_DIR / "imputer.joblib"
    FEATURE_LIST_PATH = MODEL_DIR / "feature_list.json"
    THRESHOLD_PATH = MODEL_DIR / "threshold.txt"
    
    # Business parameters
    DEFAULT_COST_FP = 500      # Cost of preventive maintenance
    DEFAULT_COST_FN = 50000    # Cost of unplanned breakdown
    
    # Model parameters
    TARGET_COL = "label"
    
    # Type encoding
    TYPE_ENCODING = {"L": 0, "M": 1, "H": 2}
    
    # Risk thresholds
    RISK_LOW = 0.3
    RISK_HIGH = 0.7


# ========================================
# HELPER FUNCTIONS
# ========================================

@st.cache_data
def load_data():
    """Load and preprocess the predictive maintenance dataset."""
    eng_path = Config.ENGINEERED_DATA
    raw_path = Config.RAW_DATA
    
    # Try engineered dataset first
    if eng_path.exists():
        df = pd.read_csv(eng_path)
        st.sidebar.success("📁 Using engineered dataset")
    elif raw_path.exists():
        df = pd.read_csv(raw_path)
        st.sidebar.warning("⚠️ Using raw dataset - predictions may degrade")
        st.sidebar.info("💡 Run feature engineering notebook for better results")
    else:
        st.error(f"❌ No dataset found in {Config.DATA_DIR}")
        st.info(f"Expected: {eng_path} or {raw_path}")
        st.stop()
    
    # Clean column names (matching notebook preprocessing)
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(' ', '_', regex=False)
        .str.replace('[', '', regex=False)
        .str.replace(']', '', regex=False)
        .str.replace('(', '', regex=False)
        .str.replace(')', '', regex=False)
    )
    
    # Encode Type column if it exists and is not already encoded
    if 'Type' in df.columns and df['Type'].dtype == 'object':
        df['Type'] = df['Type'].map(Config.TYPE_ENCODING)
        st.sidebar.info("✓ Type column encoded (L=0, M=1, H=2)")
    
    # Ensure UDI exists
    if 'UDI' not in df.columns:
        df.insert(0, 'UDI', range(1, len(df) + 1))
        st.sidebar.info(f"✓ Created UDI column (1 to {len(df)})")
    
    return df


@st.cache_resource
def load_model_and_artifacts():
    """Load the trained model, imputer, features, and threshold."""
    model_path = Config.MODEL_PATH
    imputer_path = Config.IMPUTER_PATH
    feature_path = Config.FEATURE_LIST_PATH
    threshold_path = Config.THRESHOLD_PATH
    
    if not model_path.exists():
        st.error(f"❌ Model not found at: {model_path}")
        st.info("💡 Train the model first using: notebooks/03_Model_Training.ipynb")
        st.stop()
    
    try:
        model = joblib.load(model_path)
        
        # Load imputer if exists
        imputer = None
        if imputer_path.exists():
            imputer = joblib.load(imputer_path)
        
        # Load feature list
        feature_names = None
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                feature_names = json.load(f)
        
        # Load optimal threshold
        threshold = 0.5  # Default
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                threshold = float(f.read().strip())
        
        return model, imputer, feature_names, threshold
        
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {e}")
        st.stop()


def prepare_features_for_model(df, feature_names):
    """
    Prepare feature matrix matching the model's expected features.
    
    Returns X with features in the correct order expected by the model.
    """
    if feature_names is None:
        # Fallback: use all numeric columns except identifiers and target
        drop_cols = ['UDI', 'Product_ID', 'label']
        cols_to_drop = [col for col in drop_cols if col in df.columns]
        X = df.drop(columns=cols_to_drop, errors='ignore')
        return X
    
    # Check for missing features
    missing = set(feature_names) - set(df.columns)
    if missing:
        st.error(f"❌ Missing {len(missing)} required features:")
        st.code("\n".join(sorted(missing)))
        st.info("💡 Run feature engineering: notebooks/02_Feature_Engineering.ipynb")
        st.stop()
    
    # Return features in model's expected order
    X = df[feature_names].copy()
    return X


@st.cache_resource
def get_shap_explainer(_model, X_sample):
    """Create and cache SHAP explainer for model interpretability."""
    # Use subset for faster computation
    background = shap.sample(X_sample, min(100, len(X_sample)))
    
    try:
        # Try TreeExplainer first (faster for tree models)
        explainer = shap.TreeExplainer(_model, background)
        return explainer
    except:
        # Fallback to general explainer
        def model_predict(X):
            return _model.predict_proba(X)[:, 1]
        
        explainer = shap.Explainer(model_predict, background)
        return explainer


def calculate_risk_metrics(pred_proba, cost_fp, cost_fn, threshold=0.5):
    """Calculate business metrics based on prediction probability."""
    pred_label = 1 if pred_proba >= threshold else 0
    
    if pred_label == 1:  # Predicted failure
        expected_cost = (1 - pred_proba) * cost_fp
        recommendation = "🔧 **SCHEDULE MAINTENANCE**"
        reason = f"High failure risk ({pred_proba:.1%}) detected"
    else:  # Predicted healthy
        expected_cost = pred_proba * cost_fn
        recommendation = "✅ **CONTINUE OPERATION**"
        reason = f"Machine is healthy (risk: {pred_proba:.1%})"
    
    # Determine risk level
    if pred_proba < Config.RISK_LOW:
        risk_level = "🟢 LOW"
    elif pred_proba < Config.RISK_HIGH:
        risk_level = "🟡 MEDIUM"
    else:
        risk_level = "🔴 HIGH"
    
    return {
        'expected_cost': expected_cost,
        'recommendation': recommendation,
        'reason': reason,
        'risk_level': risk_level,
        'pred_label': pred_label
    }


def perform_statistical_analysis(df, features, target):
    """Perform t-tests between failure and healthy states."""
    if target not in df.columns:
        st.warning(f"Target column '{target}' not found")
        return pd.DataFrame()
    
    failure_data = df[df[target] == 1]
    healthy_data = df[df[target] == 0]
    
    if len(failure_data) == 0 or len(healthy_data) == 0:
        st.warning("Insufficient data for statistical analysis")
        return pd.DataFrame()
    
    results = []
    
    for col in features:
        if col not in df.columns:
            continue
            
        # Only test numeric columns
        if df[col].dtype not in ['float64', 'int64']:
            continue
        
        fail_vals = failure_data[col].dropna()
        healthy_vals = healthy_data[col].dropna()
        
        if len(fail_vals) < 2 or len(healthy_vals) < 2:
            continue
        
        try:
            t_stat, p_val = stats.ttest_ind(fail_vals, healthy_vals, equal_var=False)
            
            # Calculate Cohen's d
            pooled_std = np.sqrt(
                ((len(fail_vals) - 1) * fail_vals.std()**2 + 
                 (len(healthy_vals) - 1) * healthy_vals.std()**2) / 
                (len(fail_vals) + len(healthy_vals) - 2)
            )
            cohens_d = (fail_vals.mean() - healthy_vals.mean()) / pooled_std if pooled_std > 0 else 0
            
            results.append({
                'Feature': col,
                'Failure Mean': fail_vals.mean(),
                'Healthy Mean': healthy_vals.mean(),
                'Difference': fail_vals.mean() - healthy_vals.mean(),
                't-statistic': t_stat,
                'p-value': p_val,
                "Cohen's d": cohens_d,
                'Significant': '✅ Yes' if p_val < 0.05 else '❌ No'
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(results).sort_values('p-value') if results else pd.DataFrame()


def color_risk_row(row):
    """Apply row-wise background color based on failure_risk value."""
    try:
        risk_str = row['failure_risk']
        if isinstance(risk_str, str) and '%' in risk_str:
            risk = float(risk_str.strip('%')) / 100
        else:
            risk = float(risk_str)
        
        if risk >= Config.RISK_HIGH:
            bg_color = '#ff4444'
            text_color = 'white'
        elif risk >= Config.RISK_LOW:
            bg_color = '#ffcc00'
            text_color = 'black'
        else:
            bg_color = '#28a745'
            text_color = 'white'
        
        return [f'background-color: {bg_color}; color: {text_color}; font-weight: bold' for _ in row]
    except:
        return ['' for _ in row]


# ========================================
# PAGE SETUP
# ========================================

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔧 AI-Driven Predictive Maintenance Dashboard")
st.markdown("""
Monitor machine health and predict failures using sensor & operational data.  
*Powered by Explainable AI (SHAP) for transparent decision-making.*
""")


# ========================================
# LOAD DATA AND MODEL
# ========================================

with st.spinner("Loading data and model..."):
    df = load_data()
    model, imputer, feature_names, optimal_threshold = load_model_and_artifacts()


# ========================================
# PREPARE FEATURES
# ========================================

# Create feature matrix for model
X = prepare_features_for_model(df, feature_names)

# Apply imputer if available
if imputer is not None:
    X_processed = pd.DataFrame(
        imputer.transform(X),
        columns=X.columns,
        index=X.index
    )
else:
    X_processed = X.copy()

st.sidebar.success(f"✅ Loaded {len(df)} records with {len(X.columns)} features")


# ========================================
# SIDEBAR - MODEL INFO
# ========================================

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Information")

model_name = type(model).__name__
st.sidebar.write(f"**Model Type:** {model_name}")

if Config.MODEL_PATH.exists():
    mtime = os.path.getmtime(Config.MODEL_PATH)
    model_date = pd.to_datetime(mtime, unit='s')
    st.sidebar.caption(f"📅 Model trained: {model_date.strftime('%Y-%m-%d %H:%M')}")

st.sidebar.caption(f"🎯 Optimal Threshold: {optimal_threshold:.3f}")

if st.sidebar.button("⬅️ Reset All Selections"):
    st.session_state.clear()
    st.rerun()


# ========================================
# SIDEBAR - VIEW MODE & SETTINGS
# ========================================

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Dashboard Settings")

st.sidebar.subheader("🔄 Live Monitoring")
auto_refresh = st.sidebar.checkbox(
    "Enable Auto-Refresh",
    value=False,
    help="Automatically refresh predictions every 30 seconds"
)

if auto_refresh:
    refresh_interval = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=10,
        max_value=300,
        value=30,
        step=10
    )
    st.sidebar.info(f"🔄 Auto-refreshing every {refresh_interval}s")
    time.sleep(refresh_interval)
    st.rerun()

st.sidebar.markdown("---")

view_mode = st.sidebar.radio(
    "View Mode:",
    ["🔍 Single Machine Analysis", "🏭 Fleet Overview"],
    help="Choose between detailed single machine view or fleet-wide overview"
)

st.sidebar.markdown("---")


# ========================================
# FLEET OVERVIEW MODE
# ========================================

if view_mode == "🏭 Fleet Overview":
    st.subheader("🏭 Fleet Health Dashboard")
    
    with st.spinner("Analyzing fleet health..."):
        predictions = model.predict_proba(X_processed)[:, 1]
        
        # Apply probability smoothing to avoid extreme values
        epsilon = 0.001
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        df_display = df.copy()
        df_display['failure_risk'] = predictions
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Machines", len(df))
    with col2:
        high_risk = (predictions >= Config.RISK_HIGH).sum()
        st.metric("High Risk (≥70%)", high_risk, 
                 delta=f"{high_risk/len(df)*100:.1f}%",
                 delta_color="inverse")
    with col3:
        medium_risk = ((predictions >= Config.RISK_LOW) & (predictions < Config.RISK_HIGH)).sum()
        st.metric("Medium Risk (30-70%)", medium_risk)
    with col4:
        low_risk = (predictions < Config.RISK_LOW).sum()
        st.metric("Low Risk (<30%)", low_risk,
                 delta=f"{low_risk/len(df)*100:.1f}%")
    
    st.markdown("---")
    
    # Top at-risk machines
    st.subheader("⚠️ Top 10 At-Risk Machines")
    
    display_cols = ['UDI', 'failure_risk']
    
    # Add available sensor columns
    sensor_cols = [col for col in ['Air_temperature_K', 'Process_temperature_K', 
                                   'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
                  if col in df_display.columns]
    display_cols.extend(sensor_cols[:3])
    
    at_risk = df_display.nlargest(10, 'failure_risk')[display_cols].copy()
    at_risk['failure_risk'] = at_risk['failure_risk'].apply(lambda x: f"{x:.1%}")
    
    styled_df = at_risk.style.apply(color_risk_row, axis=1)
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    st.caption("🔴 **Red** = High Risk (≥70%) | 🟡 **Yellow** = Medium Risk (30-70%) | 🟢 **Green** = Low Risk (<30%)")
    
    # Download button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        download_cols = ['UDI', 'failure_risk'] + sensor_cols
        csv_data = df_display[download_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download Fleet Predictions (CSV)",
            data=csv_data,
            file_name=f"fleet_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Risk distribution
    st.subheader("📊 Risk Distribution Across Fleet")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(predictions, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(Config.RISK_LOW, color='green', linestyle='--', label='Low Risk Threshold', linewidth=2)
    ax.axvline(Config.RISK_HIGH, color='red', linestyle='--', label='High Risk Threshold', linewidth=2)
    ax.set_xlabel('Failure Risk Probability')
    ax.set_ylabel('Number of Machines')
    ax.set_title('Distribution of Failure Risk Across Fleet')
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    st.info("💡 Click 'Single Machine Analysis' in the sidebar to investigate specific machines")
    st.stop()


# ========================================
# SINGLE MACHINE ANALYSIS MODE
# ========================================

st.sidebar.subheader("🔍 Machine Selection")

machine_ids = sorted(df["UDI"].unique())

# Quick action buttons
st.sidebar.markdown("**Quick Find:**")
col_a, col_b = st.sidebar.columns(2)

with col_a:
    if st.button("🔴 High Risk", help="Jump to highest risk machine", use_container_width=True):
        all_predictions = model.predict_proba(X_processed)[:, 1]
        highest_risk_idx = np.argmax(all_predictions)
        highest_risk_udi = df.iloc[highest_risk_idx]['UDI']
        st.session_state['selected_udi'] = highest_risk_udi
        st.rerun()

with col_b:
    if st.button("🟢 Low Risk", help="Jump to lowest risk machine", use_container_width=True):
        all_predictions = model.predict_proba(X_processed)[:, 1]
        lowest_risk_idx = np.argmin(all_predictions)
        lowest_risk_udi = df.iloc[lowest_risk_idx]['UDI']
        st.session_state['selected_udi'] = lowest_risk_udi
        st.rerun()

# Machine selection
default_udi = st.session_state.get('selected_udi', machine_ids[0])
if default_udi not in machine_ids:
    default_udi = machine_ids[0]

selected_id = st.sidebar.selectbox(
    "Select Machine UDI:",
    machine_ids,
    index=machine_ids.index(default_udi),
    help="Choose a machine to analyze"
)

st.session_state['selected_udi'] = selected_id

# Get machine data
machine_data = df[df["UDI"] == selected_id].copy()

if machine_data.empty:
    st.error(f"No data found for Machine UDI {selected_id}")
    st.stop()

machine_data = machine_data.tail(1)
X_machine = X_processed.loc[machine_data.index]

st.markdown("---")


# ========================================
# MACHINE STATUS
# ========================================

st.subheader(f"📍 Machine Status: UDI {selected_id}")

with st.expander("📋 View Raw Sensor Data", expanded=False):
    st.dataframe(machine_data, use_container_width=True)


# ========================================
# PREDICTION
# ========================================

st.markdown("---")
st.subheader("🧠 Failure Prediction")

st.info("📊 Probabilities are calibrated to prevent overconfident predictions (smoothed between 0.1% and 99.9%)")

try:
    pred_proba = model.predict_proba(X_machine)[:, 1][0]
    
    # Apply probability smoothing to avoid extreme values (0 or 1)
    # This prevents overconfident predictions and ensures realistic expected costs
    epsilon = 0.001  # Small value to prevent exact 0 or 1
    pred_proba = np.clip(pred_proba, epsilon, 1 - epsilon)
    
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    st.stop()

# Business parameters
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Business Parameters")

cost_fp = st.sidebar.number_input(
    "Cost of False Positive ($)",
    min_value=0,
    value=Config.DEFAULT_COST_FP,
    step=100,
    help="Cost of unnecessary maintenance"
)

cost_fn = st.sidebar.number_input(
    "Cost of False Negative ($)",
    min_value=0,
    value=Config.DEFAULT_COST_FN,
    step=1000,
    help="Cost of unplanned downtime"
)

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(optimal_threshold),
    step=0.01,
    help="Probability threshold for failure classification"
)

metrics = calculate_risk_metrics(pred_proba, cost_fp, cost_fn, threshold)

# Display results
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Failure Probability", f"{pred_proba:.1%}")
    
with col2:
    status = "⚠️ FAILURE" if metrics['pred_label'] == 1 else "✅ HEALTHY"
    st.metric("Status", status)
    
with col3:
    st.metric("Risk Level", metrics['risk_level'])

# Business recommendation
st.markdown("---")
st.subheader("💼 Business Recommendation")

col1, col2 = st.columns([2, 1])

with col1:
    st.success(metrics['recommendation'])
    st.caption(metrics['reason'])

with col2:
    # Display expected cost with appropriate precision
    if metrics['expected_cost'] < 10:
        st.metric("Expected Cost", f"${metrics['expected_cost']:.2f}")
    else:
        st.metric("Expected Cost", f"${metrics['expected_cost']:,.0f}")

with st.expander("ℹ️ How is Expected Cost calculated?"):
    if metrics['pred_label'] == 1:
        st.write(f"""
        **Scheduled Maintenance Scenario:**
        - Predicted: Failure (schedule maintenance)
        - If prediction is wrong (machine was healthy): Cost = ${cost_fp:,}
        - Probability of being wrong: {(1-pred_proba):.3%}
        - **Expected Cost = {(1-pred_proba):.3%} × ${cost_fp:,} = ${metrics['expected_cost']:.2f}**
        
        *Note: Very high confidence predictions result in very low expected costs.*
        """)
    else:
        st.write(f"""
        **Continue Operation Scenario:**
        - Predicted: Healthy (continue operation)
        - If prediction is wrong (machine fails): Cost = ${cost_fn:,}
        - Probability of being wrong: {pred_proba:.3%}
        - **Expected Cost = {pred_proba:.3%} × ${cost_fn:,} = ${metrics['expected_cost']:.2f}**
        
        *Note: Very low failure risk results in very low expected costs.*
        """)


# ========================================
# SHAP EXPLANATIONS
# ========================================

st.markdown("---")
st.subheader("🔍 Model Explanation: Why this prediction?")

st.info("Using SHAP (SHapley Additive exPlanations) to show which features drove this prediction")

with st.spinner("Generating SHAP explanations..."):
    try:
        explainer = get_shap_explainer(model, X_processed)
        shap_values = explainer(X_machine)
        
        st.markdown("#### Feature Contributions to Prediction")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Handle binary classification - extract failure class (index 1)
        if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
            # Shape is (n_samples, n_features, n_classes) - extract failure class
            shap_values_failure = shap.Explanation(
                values=shap_values.values[0, :, 1],
                base_values=shap_values.base_values[0, 1] if hasattr(shap_values.base_values[0], '__len__') else shap_values.base_values[0],
                data=shap_values.data[0],
                feature_names=shap_values.feature_names
            )
            shap.plots.waterfall(shap_values_failure, show=False)
        else:
            # For PermutationExplainer or other explainers
            shap.plots.waterfall(shap_values[0], show=False)
        
        st.pyplot(fig)
        plt.close(fig)
        
        st.caption("""
        🔴 Red bars push prediction toward FAILURE  
        🔵 Blue bars push prediction toward HEALTHY  
        The waterfall shows cumulative impact of each feature on the prediction.
        """)
        
    except Exception as e:
        st.error(f"❌ Could not generate SHAP explanation: {e}")
        st.info("💡 This might happen with certain model types or data formats")


# ========================================
# GLOBAL FEATURE IMPORTANCE
# ========================================

st.markdown("---")
st.subheader("🌎 Global Feature Importance")

st.info("Features that generally drive failure predictions across all machines")

try:
    with st.spinner("Calculating global importances..."):
        sample_size = min(500, len(X_processed))
        X_sample = X_processed.sample(n=sample_size, random_state=42)
        
        explainer_global = get_shap_explainer(model, X_sample)
        shap_values_global = explainer_global(X_sample)
        
        # Handle SHAP output format
        if hasattr(shap_values_global, 'values'):
            shap_array = shap_values_global.values
        else:
            shap_array = shap_values_global
        
        # Handle multi-class output (samples, features, classes)
        if len(shap_array.shape) == 3:
            shap_array = shap_array[:, :, 1]  # Use class 1 (failure)
        
        importances = np.abs(shap_array).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            top_features = importance_df.head(10)
            ax.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'])
            ax.invert_yaxis()
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('Top 10 Most Important Features')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.dataframe(
                importance_df.head(10).reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )
            
except Exception as e:
    st.error(f"❌ Could not calculate global importance: {e}")


# ========================================
# STATISTICAL ANALYSIS
# ========================================

if st.sidebar.checkbox("📊 Show Statistical Analysis", value=False):
    st.markdown("---")
    st.subheader("📊 Statistical Sensor Analysis")
    
    st.info("""
    Comparing sensor readings between machines that failed vs. healthy machines  
    Uses independent t-tests to identify statistically significant differences
    """)
    
    with st.spinner("Performing statistical tests..."):
        stat_results = perform_statistical_analysis(df, X.columns, Config.TARGET_COL)
        
        if not stat_results.empty:
            show_all = st.checkbox("Show all features (not just significant)", value=False)
            
            if not show_all:
                display_results = stat_results[stat_results['p-value'] < 0.05]
            else:
                display_results = stat_results
            
            if not display_results.empty:
                st.dataframe(
                    display_results.style.format({
                        'Failure Mean': '{:.2f}',
                        'Healthy Mean': '{:.2f}',
                        'Difference': '{:.2f}',
                        't-statistic': '{:.2f}',
                        'p-value': '{:.4f}',
                        "Cohen's d": '{:.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.caption("""
                **Interpretation:**  
                - p-value < 0.05 = statistically significant difference  
                - Cohen's d > 0.5 = medium effect size, > 0.8 = large effect  
                - Positive difference = sensor value higher during failures
                """)
            else:
                st.info("No statistically significant differences found (p < 0.05)")
        else:
            st.warning("No statistical tests could be performed on the available features")


# ========================================
# FOOTER
# ========================================

st.markdown("---")
st.caption("""
**Predictive Maintenance Dashboard** | Built with Streamlit &SHAP  
Developed by **Vinay Gupta Kandula** | For questions or issues, contact your data science team
""")