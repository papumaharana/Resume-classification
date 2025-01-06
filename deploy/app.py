#!/usr/bin/env python
# coding: utf-8

# In[99]:


import PyPDF2
from docx import Document
import os
import re
import subprocess

import pickle
import nltk
import spacy
import string
import numpy as np
import pandas as pd
import streamlit as st


# ## Import models:

# In[102]:


model = pickle.load(open('deploy/model_Gb.pkl', 'rb'))


# ## Creating some functions to extract data from uploaded resume file:

# In[120]:


## Creating function for extract data with cleaned text:
nlp = spacy.load('en_core_web_sm')

## function for data cleaning:
def clean_data(text):
    text1 = ' '.join(re.findall('\w+', text))
    doc = nlp(text1)
    clean_text = [word.lemma_ for word in doc if not word.is_stop and not word.is_punct and not word.is_bracket and not word.is_currency and not word.is_space]
    clean_text = ' '.join(clean_text)
    return clean_text

## Reading pdf file:
def read_pdf(file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    ##Clean data:
    text = clean_data(text)
    return text

## Reading docx file:
def read_docx(file):
    """Extract text from a DOCX file."""
    doc = Document(file)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    ##Clean data:
    text = clean_data(text)
    return text

## Converting .doc file to .docx file:
def convert_doc_to_docx(doc_file):
    """Convert a .doc file to .docx using LibreOffice."""
    temp_docx_file = doc_file.replace(".doc", ".docx")
    subprocess.run(["soffice", "--headless", "--convert-to", "docx", doc_file], check=True)
    return temp_docx_file

## Reading doc file:
def read_doc(file):
    # Save the uploaded file
    temp_file = f"temp_{file.name}"
    with open(temp_file, "wb") as f:
        f.write(file.getbuffer())
    # Convert .doc to .docx
    converted_file = convert_doc_to_docx(temp_file)
    # Read the converted .docx file
    text = read_docx(converted_file)
    # Clean up temporary files
    os.remove(temp_file)
    os.remove(converted_file)
    ##Clean data:
    text = clean_data(text)
    return text 


# # Streamlit App main codes:

# In[124]:


df = pd.DataFrame([], columns=['Uploaded_files', 'Predicted_profile'])
file_name = []
pred_prof = []

st.title("Resume Classification App")

uploaded_file = st.file_uploader("Upload a DOCX, DOC, or PDF file", type=["docx", "doc", "pdf"], accept_multiple_files=True)
for file in uploaded_file:
    if file is not None:
        file_extension = os.path.splitext(file.name)[-1].lower()
        
        if file_extension == ".pdf":
            try:
                text = read_pdf(file)
                ##Prediction:
                pred = model.predict([text])
                file_name.append(file.name)
                if pred==['react_dev']:
                    pred_prof.append('React developer')
                elif pred==['sql_dev']:
                    pred_prof.append('Sql developer')
                elif pred==['workday']:
                    pred_prof.append('Workday')
                else:
                    pred_prof.append('Peoplesoft')
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
        
        elif file_extension == ".docx":
            try:
                text = read_docx(file)
                ##Prediction:
                pred = model.predict([text])
                file_name.append(file.name)
                if pred==['react_dev']:
                    pred_prof.append('React developer')
                elif pred==['sql_dev']:
                    pred_prof.append('Sql developer')
                elif pred==['workday']:
                    pred_prof.append('Workday')
                else:
                    pred_prof.append('Peoplesoft')
            except Exception as e:
                st.error(f"Error reading DOCX: {e}")
        
        elif file_extension == ".doc":
            try:
                text = read_doc(file)
                ##Prediction:
                pred = model.predict([text])
                file_name.append(file.name)
                if pred==['react_dev']:
                    pred_prof.append('React developer')
                elif pred==['sql_dev']:
                    pred_prof.append('Sql developer')
                elif pred==['workday']:
                    pred_prof.append('Workday')
                else:
                    pred_prof.append('Peoplesoft')
            except Exception as e:
                st.error(f"Error reading DOC: {e}")
                
        else:
            st.error("Unsupported file type.")


if len(pred_prof) > 0:
    df['Uploaded_files'] = file_name
    df['Predicted_profile'] = pred_prof
    st.subheader('Predictions:')
    st.table(df.style.format())


# In[ ]:




