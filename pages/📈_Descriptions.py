import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Biomedical Translation Task - WMT 24", layout="wide")

# Improved Model Descriptions
model_descriptions = """
### Model Descriptions

**EN-DE Translation Models:**
- **Mistral Base Model**: The base instruction-tuned Mistral-7B model.
- **Mistral Model B**: A monolingual fine-tuned Mistral-7B  on a smaller biomedical corpus from WMT-24 dataset (8000 sentences).
- **Mistral Model C**: Mistral Model B, with additional data (4000+ sentences), augmented using terminology-based filtering, with terminology extracted using SciBERT.

**Multilingual Translation Models:**
- **Mistral Multilingual Model E**: A fine-tuned version of the Mistral Base Model, trained on three language pairs: English-Italian (en-it), English-German (en-de), and English-Spanish (en-es), using WMT 2024 dataset (8000 sentences).
"""

st.markdown("# Biomedical Translation Task - WMT 24")
st.markdown(model_descriptions)

# Sample data for demonstration
mistral_base = {
    "Language Pair": ["en-it", "en-de", "en-fr"],
    "BLEU": [16, 11, 27],
    "ChrF": [48.37, 48.62, 56.80],
    "COMET": [79, 81, 81]
}

mistral_model_b = {
    "Language Pair": ["en-de"],
    "BLEU": [22.0],
    "ChrF": [53.65],
    "COMET": [84.6]
}

mistral_model_c = {
    "Language Pair": ["en-de"],
    "BLEU": [25.0],
    "ChrF": [57.47],
    "COMET": [84.6]
}

mistral_multilingual_e = {
    "Language Pair": ["en-it", "en-de", "en-fr"],
    "BLEU": [0.0, 0, 30],
    "ChrF": [0.0, 0.0, 59],
    "COMET": [0, 0, 84]
}

# Create DataFrames
df_mistral_base = pd.DataFrame(mistral_base)
df_mistral_b = pd.DataFrame(mistral_model_b)
df_mistral_c = pd.DataFrame(mistral_model_c)
df_mistral_e = pd.DataFrame(mistral_multilingual_e)

# Add model names
df_mistral_base["Model"] = "Mistral Base"
df_mistral_b["Model"] = "Mistral Model B"
df_mistral_c["Model"] = "Mistral Model C"
df_mistral_e["Model"] = "Mistral Multilingual Model E"

# Filter Mistral Base model to include only en-de for monolingual comparison
df_mistral_base_en_de = df_mistral_base[df_mistral_base["Language Pair"] == "en-de"]

# Separate DataFrames for plotting
df_monolingual = pd.concat([df_mistral_base_en_de, df_mistral_b, df_mistral_c])
df_multilingual = pd.concat([df_mistral_base, df_mistral_e])

def plot_statistics(df, title):
    df = df.reset_index(drop=True)
    df['Model_Language Pair'] = df.apply(lambda row: f"{row['Model']} ({row['Language Pair']})" if row['Language Pair'] else row['Model'], axis=1)
    df_melted = df.melt(id_vars=["Model", "Language Pair", "Model_Language Pair"], value_vars=["BLEU", "ChrF", "COMET"],
                        var_name="Metric", value_name="Score")

    fig = px.bar(df_melted, x="Model_Language Pair", y="Score", color="Metric", barmode="group",
                 labels={"Model_Language Pair": "Model (Language Pair)"},
                 title=title)
    fig.update_layout(xaxis_tickangle=-45, width=1000, height=600, margin=dict(l=20, r=20, t=40, b=20))
    return fig

st.markdown("## Model Performance Summary")

tab1, tab2 = st.tabs(["Monolingual Models", "Multilingual Models"])

with tab1:
    st.markdown("### Monolingual Models")
    st.dataframe(df_monolingual, width=1000)
    st.plotly_chart(plot_statistics(df_monolingual, "Monolingual Model Performance Metrics"), use_container_width=False)

with tab2:
    st.markdown("### Multilingual Models")
    "The multilingual model finetuned has currently been evaluated only on the en-fr language pair."
    "We compare it to the base model's performance on the same language pair."
    st.dataframe(df_multilingual, width=1000)
    st.plotly_chart(plot_statistics(df_multilingual, "Multilingual Model Performance Metrics"), use_container_width=False)
