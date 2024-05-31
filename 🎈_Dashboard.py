import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Biomedical Translation Task - WMT 24", layout="centered")
st.sidebar.header("Navigation")
st.sidebar.info("Use the sidebar to navigate between pages.")
st.markdown("# Welcome to the Biomedical Shared Task of WMT 24")
st.markdown("Use the navigation on the left to explore the different sections of this app.")



def display_summary():
    summary = """
    ## Biomedical Shared Task of WMT 24

    This interface presents the performance metrics of different models developed for the **Biomedical Shared Task of WMT 24**.
    The metrics include BLEU, ChrF, and COMET scores.

    ### Performance Summary:
    - **Model A** (Base Model) shows minimal performance across all metrics.
    - **Model B** (Monolingual finetuned on WMT24) has slightly lower performance compared to others.
    - **Model C** (Monolingual finetuned on WMT24 and WMT21) excels in all metrics, making it the top-performing model.

    **Model E** is finetuned on three language pairs: en-it, en-de, and en-es, achieving lower performance.

    Explore the tabs to view detailed statistics and visualizations. Below is a quick visualization of the model performance:
    """
    return summary

st.markdown(display_summary())

def plot_statistics(df):
    df = pd.DataFrame(df)
    df = df.reset_index()
    df['Model_Submodel'] = df.apply(lambda row: f"{row['Model']}\n{row['Submodel']}" if row['Submodel'] else row['Model'], axis=1)
    df_melted = df.melt(id_vars=["Model", "Submodel", "Model_Submodel"], value_vars=["BLEU", "ChrF", "COMET"],
                        var_name="Metric", value_name="Score")

    fig = px.bar(df_melted, x="Model_Submodel", y="Score", color="Metric", barmode="group",
                 labels={"Model_Submodel": "Model"},
                 title="Model Performance Metrics")
    fig.update_layout(xaxis_tickangle=-45, width=800, height=400, margin=dict(l=20, r=20, t=30, b=20))
    return fig

# Sample data for demonstration
model_data = {
    "Model": ["Model A", "Model B", "Model C"],
    "Submodel": ["", "", ""],
    "BLEU": [0.92, 0.89, 0.94],
    "ChrF": [0.91, 0.88, 0.93],
    "COMET": [0.90, 0.87, 0.92]
}

st.subheader("Key Findings")
st.write("""
1. Monolingual finetuning might achieve higher performance increase, compared to multilingual finetuning.
2. Performance continues to increase above 12k parallel sentences.
3. Lower-quality in-domain data might not always improve performance.
""")

st.plotly_chart(plot_statistics(model_data), use_container_width=False)
