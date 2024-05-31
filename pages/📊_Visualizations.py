import pandas as pd
import plotly.express as px
import streamlit as st

# Define the data for each model
data = {
    "Model": ["Mistral Base", "Mistral Base", "Mistral Base", "Mistral B", "Mistral C", "Mistral Multilingual E", "Mistral Multilingual E", "Mistral Multilingual E"],
    "Language Pair": ["en-it", "en-de", "en-fr", "en-de", "en-de", "en-it", "en-de", "en-fr"],
    "BLEU": [16, 11, 27, 22.0, 25.0, 0.0, 0, 30],
    "ChrF": [48.37, 48.62, 56.80, 53.65, 57.47, 0.0, 0.0, 59],
    "COMET": [79, 81, 81, 84.6, 84.6, 0, 0, 84]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Streamlit app for visualization
def main():
    st.title("Data Visualizations")

    # Dynamic filters and box plots
    st.header("Models Compared by Metric")
    selected_metric = st.selectbox("Select a Metric for Box Plot", ["BLEU", "ChrF", "COMET"])
    filtered_models = st.multiselect("Select Models to Display", df['Model'].unique(), default=df['Model'].unique())
    filtered_df = df[df['Model'].isin(filtered_models)]
    box_fig = px.box(filtered_df, x='Model', y=selected_metric, color='Language Pair',
                     title=f"Box Plot of {selected_metric} by Model and Language Pair")
    st.plotly_chart(box_fig)

    # Create interactive plots for each metric
    st.header("Metric Comparisons")

    metrics = df.columns[2:]  # This dynamically pulls metric names from DataFrame
    df_first = df[df['Model'] != 'Mistral Base']
    for metric in metrics:
        fig = px.bar(df_first, x='Model', y=metric, color='Language Pair', title=f"{metric} Scores by Model and Language Pair")
        st.plotly_chart(fig)

    st.header("Detailed Metric Analysis")
    st.subheader("BLEU vs ChrF Scores")
    fig = px.scatter(df, x="BLEU", y="ChrF", color="Model", size="COMET", hover_data=["Language Pair"], title="BLEU vs ChrF Scores Colored by Model")
    st.plotly_chart(fig)

    # Optionally, add a heatmap or correlation matrix
    st.header("Correlation Heatmap of Metrics")
    corr_matrix = df[["BLEU", "ChrF", "COMET"]].corr()
    fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix of Performance Metrics")
    st.plotly_chart(fig)




if __name__ == "__main__":
    main()
