import streamlit as st
import pandas as pd
import numpy as np
from apyori import apriori
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
from PIL import Image
from io import BytesIO
import xlsxwriter
import base64
import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:

image = Image.open('WhatsApp-Image-2021-12-23-at-3.05.01-PM-_1_.jpg')
image1=Image.open('WhatsApp-Image-2021-12-23-at-3.05.01-PM-_2_.jpg')
first,center,last=st.columns(3)
first.image(image)
center.write("")
last.image(image1)
data=pd.read_csv("analysis data.csv")
def st_csv_download_button(df):
    csv = df.to_csv(index=False) #if no filename is given, a string is returned
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download Sample Template</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)  

st_csv_download_button(data)    
#check1 = st.sidebar.button("sample template")


st.title('Market Basket Analysis')
st.write("Market basket analysis is a data mining technique used by retailers to increase sales by better understanding customer purchasing patterns. It involves analyzing large data sets, such as purchase history, to reveal product groupings, as well as products that are likely to be purchased together.")
st.write("----------------------------------------------------------------------------------------")
#st.write("An association rule has two parts: an **antecedent** (if) and a **consequent** (then). An antecedent is an item found within the data. A consequent is an item found in combination with the antecedent. ... Association rules are calculated from itemsets, which are made up of two or more items.")
#st.write("**consequent**  : item found in combination with the antecedent")
#st.write("**support**     : Support is an indication of how frequently the items appear in the data. It refers to how often a given rule appears in the database being mined.")
#st.write("**confidence**  : Confidence indicates the number of times the if-then statements are found true.Confidence refers to the amount of times a given rule turns out to be true in practice. A rule may show a strong correlation in a data set because it appears very often but may occur far less when applied. This would be a case of high support, but low confidence.Conversely, a rule might not particularly stand out in a data set, but continued analysis shows that it occurs very frequently. This would be a case of high confidence and low support. Using these measures helps analysts separate causation from correlation, and allows them to properly value a given rule. ")
#st.write("**lift**        : lift can be used to compare confidence with expected confidence, or how many times an if-then statement is expected to be found true. It is the ratio of confidence to support. If the lift value is a negative value, then there is a negative correlation between datapoints. If the value is positive, there is a positive correlation, and if the ratio equals 1, then there is no correlation.")
#if check1:
 #       st.write(data.head(10))
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
    
    def rules_to_coordinates(rules):
        rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent)[0])
        rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent)[0])
        rules['rule'] = rules.index
        return rules[['antecedent','consequent','rule']]

#print (frequent_itemsets)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    rules=rules.head(40)
    rules= rules.sort_values(by = 'confidence', ascending = False)
    rules.columns = map(str.upper, rules.columns)
    
    st.markdown(
    """<style>
        .dataframe {text-align: left !important}
    </style>
    """, unsafe_allow_html=True) 
    
#frequent_itemsets

# In[16]:


    st.write(rules)
    st.write("----------------------------------------------------------------------------------------")
    from pandas.plotting import parallel_coordinates

    # Compute the frequent itemsets
    #frequent_itemsets = apriori(onehot, min_support = 0.15, 
    #                           use_colnames = True, max_len = 2)

    # Compute rules from the frequent itemsets
    rules = association_rules(frequent_itemsets, metric = 'confidence', 
                              min_threshold = 0.55)

    # Convert rules into coordinates suitable for use in a parallel coordinates plot
    coords = rules_to_coordinates(rules.head(40))

    # Generate parallel coordinates plot
    
    fig=plt.figure(figsize=(3,5))
    parallel_coordinates(coords, 'rule')
    #plt.legend([])
    plt.grid(True)
    plt.title('graph', fontsize=25)
    st.pyplot(fig)
    st.write(" ")
    
    fig2=plt.figure(figsize=(12,7))
    sns.scatterplot(x = "support", y = "confidence", 
                   size = "lift", data = rules).set(title="Optimality of the support-confidence border")

    plt.margins(0.01,0.01)
    #st.write("**Optimality of the support-confidence border**")
    st.pyplot(fig2)
    
    st.write("An association rule has two parts: an **antecedent** (if) and a **consequent** (then). An antecedent is an item found within the data. A consequent is an item found in combination with the antecedent. ... Association rules are calculated from itemsets, which are made up of two or more items.")
    #st.write("**consequent**  : item found in combination with the antecedent")
    st.write("**support**     : Support is an indication of how frequently the items appear in the data. It refers to how often a given rule appears in the database being mined.")
    st.write("**confidence**  : Confidence indicates the number of times the if-then statements are found true.Confidence refers to the amount of times a given rule turns out to be true in practice. A rule may show a strong correlation in a data set because it appears very often but may occur far less when applied. This would be a case of high support, but low confidence.Conversely, a rule might not particularly stand out in a data set, but continued analysis shows that it occurs very frequently. This would be a case of high confidence and low support. Using these measures helps analysts separate causation from correlation, and allows them to properly value a given rule. ")
    st.write("**lift**        : lift can be used to compare confidence with expected confidence, or how many times an if-then statement is expected to be found true. It is the ratio of confidence to support. If the lift value is a negative value, then there is a negative correlation between datapoints. If the value is positive, there is a positive correlation, and if the ratio equals 1, then there is no correlation.")

