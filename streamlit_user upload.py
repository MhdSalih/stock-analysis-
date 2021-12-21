#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from apyori import apriori
from datetime import datetime





st.title('MARKET BASKET ANALYSIS')
st.write('')

upload_file=st.sidebar.file_uploader(label="upload your csv or excel file",type=["csv","xlsx"])

global df
if upload_file is not None:
  print(upload_file)
  print('hello')
  try:
    df = pd.read_csv(upload_file)
  except Exception as e:
    print(e)
    df = pd.read_excel(uploade_file)
    
    
    

df=df[df["salesanalysis1"]==1]
df.Design= df.Design.str.lower()
df["Design"]=df["Design"].astype('category')
df["QUANTITY"]=1
df2=df[["VOCNO","Design","VOCDATE",'QUANTITY']]





basket=df2.groupby(["VOCNO","Design"])["QUANTITY"].sum().unstack().reset_index().fillna(0).set_index("VOCNO")
basket=pd.DataFrame(basket)





def encode_unit(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    
basket_set = basket.applymap(encode_unit)




frequent_itemsets = apriori(basket_set, min_support=0.08, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
st.write(rules)

