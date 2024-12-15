#!/usr/bin/env python
# coding: utf-8

# In[10]:


from transformers import pipeline
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

# Load a pre-trained model for text classification
classifier = pipeline('text-classification', model='unitary/toxic-bert')



# In[18]:


st.title("Threat/Behaviour Detection")
st.write("")
st.write("""This exercise uses a combination of supervised and unsupervised approaches to analyze the intensity of threat.
Supervised learning is applied first to classify the toxicity of the text using a pre-trained BERT model (unitary/toxic-bert), then
unsupervised learning is applied to identify the outliers of the toxicity score using Isolation Forest algorithm.""")

# load data
data = pd.read_excel("data.xlsx")
#data = pd.read_excel("/mount/src/threat_analysis/data.xlsx")

#new_text = {"text": "EDP called and told police that he is going to get a gun and shoot people. Police search the area and apprehended EDP to hospital."}

new_text = st.text_area(label="Enter your text", value="It is a nice day and everything is beautiful. This text will be added to the dataset for analysis. It will not be unusual as there is no risks.")

if st.button("Analyze"):
    new_text = {"text": new_text}
    df_new_text = pd.DataFrame([new_text])
    data = pd.concat([data, df_new_text], ignore_index=True)
    array_pred = []

    for row_index, row in data.iterrows():
        result = classifier(row["text"])
        array_pred.append(result[0]["score"])
    data["score"] = pd.DataFrame(array_pred)


    # Initialize the Isolation Forest model
    iso_forest = IsolationForest(contamination="auto", random_state=42)
    
    # Fit the model on the scores
    iso_forest.fit(data[['score']])
    
    # Predict outliers: -1 indicates an outlier, 1 indicates normal
    data['analysis'] = iso_forest.predict(data[['score']])
    data["analysis"] = data["analysis"].map({-1:"Unusual", 1:""})
    data = data.drop(columns=["score"])
    st.table(data)



# In[ ]:




