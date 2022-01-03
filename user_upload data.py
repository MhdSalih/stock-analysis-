import streamlit as st
import pandas as pd
import numpy as np
from apyori import apriori
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
from PIL import Image

# In[15]:

image = Image.open('WhatsApp-Image-2021-12-23-at-3.05.01-PM-_1_.jpg')
image1=Image.open('WhatsApp-Image-2021-12-23-at-3.05.01-PM-_2_.jpg')
first,center,last=st.beta_columns(3)
first.image(image)
center.write("")
last.image(image1)
with open('analysis data.csv', 'rb') as my_file:
    st.download_button(label = 'Download', data = my_file, file_name = 'filename.xlsx', mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')  

st.title('Market Basket Analysis')
st.write("Market basket analysis is a data mining technique used by retailers to increase sales by better understanding customer purchasing patterns. It involves analyzing large data sets, such as purchase history, to reveal product groupings, as well as products that are likely to be purchased together.")
new={'PN':"diamond pendant",
'RN':"diamond ring",
'BT':"diamond bracelet",
'ER':"diamond earring",
'BN':"diamond bangle",
'NK':"diamond necklace",
'BR':"gold bracelet",
'GB':"gold bracelet",
'GN':"diamond necklae",
'REP':"jewellery repairing",
'CUF':"diamond cufflink",
'GBT':"gold bracelet with color stone",
'DIA':"gold chain",
'AN':"diamond anklet", 
'GE':"gold earring",
'SUT':"diamond suiti",
'GPN':"gold pendant with colour stone",
'GER':"gold earring",
'RP':"platinum ring",
'GNK':"gold necklace",
'NP':"nose pin", 
'GBNC':'gold bangle with colour stone',
'GHN':"gold hand chain",
'BRCH':"gold brooch",
'GP':"gold pendant",
'JEW':"gold chain",
'GRN':"gold ring with color stone",
'CRN':"diamond crown",
'HC':"hand chain",
'DJEW':"cufflink", 
'BB':"diamond belly button"}

upload_file=st.sidebar.file_uploader(label="Upload your csv or excel file",type=["csv","xlsx"])

global df
if upload_file is not None:
    print(upload_file)
    print('hello')
    try:
        df = pd.read_csv(upload_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(upload_file)
    

    df["Design"]= df.Design.str.lower()
    df["Design"]=df["Design"].astype('category')
    df=df.replace({'Category_Code':new})
    df['Design'] = np.where(df['Design']== "diamond",df["Category_Code"] , df['Design'])
    df["QUANTITY"]=1
    df2=df[["VOCNO","Design",'QUANTITY']]

    basket=df2.groupby(["VOCNO","Design"])["QUANTITY"].sum().unstack().reset_index().fillna(0).set_index("VOCNO")
    basket=pd.DataFrame(basket)

    def encode_unit(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1
    
    basket_set = basket.applymap(encode_unit)

    frequent_itemsets = apriori(basket_set, min_support=0.08, use_colnames=True)
#print (frequent_itemsets)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    rules.columns = map(str.upper, rules.columns)
    st.markdown(
    """<style>
        .dataframe {text-align: left !important}
    </style>
    """, unsafe_allow_html=True) 
    
#frequent_itemsets

# In[16]:


    st.write(rules.head(40))
