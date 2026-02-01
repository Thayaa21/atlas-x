import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from src.utils.feature_map import FEATURE_LABELS

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = OpenAI(api_key=api_key)

@st.cache_resource
def load_assets():
    model = joblib.load("src/models/atlass_x_xgb_v2.pkl")
    data = pd.read_parquet("data/processed/train_clustered.parquet")
    return model, data

def get_llm_explanation(prob, features, shap_df):
    """Refined AI Auditor that explains direction without hallucinating percentages."""
    if not api_key:
        return "⚠️ OpenAI API Key not found in .env file."

    # Map features and describe their 'push' direction
    impact_descriptions = []
    for _, row in shap_df.iterrows():
        label = FEATURE_LABELS.get(row['feature'], row['feature'])
        direction = "INCREASED" if row['impact'] > 0 else "DECREASED"
        impact_descriptions.append(f"- {label}: This factor {direction} the fraud risk score.")
    
    impact_summary = "\n".join(impact_descriptions)

    prompt = f"""
    You are a Senior Financial Crimes Investigator at a major bank. 
    A transaction was flagged with a {prob:.2%} probability of fraud.
    
    Key Findings from the Model:
    {impact_summary}
    
    Task: Write a precise 3-sentence summary for a human agent.
    1. Identify the top contributing risk factor (the one that 'increased' the risk most).
    2. Explain the behavioral context (e.g., unusual email domain or high purchase velocity).
    3. Do NOT mention SHAP values or specific percentages for individual features.
    4. Provide a final 'Approve' or 'Hold' recommendation.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a concise financial fraud expert."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}"
def main():
    st.title("🛡️ ATLAS-X: Financial Decision Intelligence")
    st.markdown("---")

    model, data = load_assets()

    # 1. Initialize Session State for the ID List
    # This prevents the dropdown options from changing every time you click a button
    if 'id_list' not in st.session_state:
        st.session_state.id_list = data['TransactionID'].sample(100).values

    # Sidebar: User Input
    st.sidebar.header("Investigation Console")
    
    # Use the 'locked' list from session state
    tx_id = st.sidebar.selectbox(
        "Select Transaction ID to Audit", 
        st.session_state.id_list,
        key="tx_selector"
    )
    
    # 2. Extract specific record
    record = data[data['TransactionID'] == tx_id]
    features = record.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1)

    # 3. Cache the Model Prediction & SHAP
    # We do this so the heavy math doesn't repeat unnecessarily
    prob = model.predict_proba(features)[0][1]
    threshold = 0.10 
    is_fraud = prob >= threshold

    # --- UI Layout ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(label="Fraud Probability", value=f"{prob:.2%}")
        if is_fraud:
            st.error("🚨 ACTION REQUIRED: HIGH RISK")
        else:
            st.success("✅ TRANSACTION CLEARED")

        st.write("**Transaction Details:**")
        st.dataframe(record[['TransactionAmt', 'card1', 'ProductCD', 'Transaction_Hour']].T)

    with col2:
        st.subheader("Decision Explainer (Waterfall)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(features) 

        # Map SHAP names for the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        
        # Axis label mapping logic
        ax = plt.gca()
        ylabels = [label.get_text() for label in ax.get_yticklabels()]
        new_labels = []
        for label in ylabels:
            clean_name = label.split('=')[0].strip()
            new_labels.append(label.replace(clean_name, FEATURE_LABELS.get(clean_name, clean_name)))
        ax.set_yticklabels(new_labels)

        st.pyplot(plt.gcf())
        plt.clf()

    # --- AI Auditor Section ---
    st.markdown("---")
    st.subheader("🤖 AI Auditor Analysis")
    
    # Now, clicking this button will re-run the script, but because we saved
    # the 'id_list' in session_state, the ID and data will stay the same!
    if st.button("Generate Human-Readable Report"):
        with st.spinner("AI is auditing the transaction DNA..."):
            shap_df = pd.DataFrame({
                'feature': features.columns,
                'impact': shap_values.values[0]
            }).sort_values(by='impact', ascending=False).head(5)
            
            explanation = get_llm_explanation(prob, features, shap_df)
            st.info(explanation)

    # --- Contextual Intelligence ---
    st.subheader("Contextual Intelligence")
    m1, m2, m3 = st.columns(3)
    m1.metric("Identity Cluster Risk", f"{record['cluster_fraud_rate'].values[0]:.1%}")
    m2.metric("Transaction Amount", f"${record['TransactionAmt'].values[0]:,.2f}")
    m3.metric("Cost-Benefit Threshold", f"{threshold}")

if __name__ == "__main__":
    main()