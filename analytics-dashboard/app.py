import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt

from sklearn import datasets
iris_dataset = datasets.load_iris()

# Title
st.title("Analytics Dashboard")

# Layout Customization

with st.expander("Iris Dataset"):
    st.write("Iris Dataset")
    st.write("Titanic Dataset")




col1, col2 = st.columns(2)



def load_scatter_plot():
    categories = iris_dataset.feature_names
    data = iris_dataset.data
    df = pd.DataFrame(data, columns=categories)

    sepal_scatter_chart = alt.Chart(df).mark_circle().encode(
        x=categories[0],
        y=categories[1],
    )

    petal_scatter_chart = alt.Chart(df).mark_circle().encode(
        x=categories[2],
        y=categories[3],
    )


    return [sepal_scatter_chart, petal_scatter_chart]



# Scatter Plot
sepal, petal = load_scatter_plot()
with col1:
    st.title("Scatter Plot")
    st.altair_chart(sepal, use_container_width=True)


with col2:
    st.title(".")
    st.altair_chart(petal, use_container_width=True)


# DataFrame
st.title("Dataframe")
categories = iris_dataset.feature_names
data = iris_dataset.data
df = pd.DataFrame(data, columns=categories)
st.dataframe(df, use_container_width=True)

# Summary
st.title("Summary Statistics")
st.dataframe(df.describe(), use_container_width=True)




