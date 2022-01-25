import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import altair as alt
# from matplotlib import pyplot as plt
import math

import pandas as pd
from datetime import datetime, timedelta

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from urllib.error import URLError

date_data = pd.read_csv("Date_data.csv")

# main csv file
main_data_file = st.sidebar.file_uploader("Upload Main CSV", type=["csv"])

# holiday csv file
data_file = st.sidebar.file_uploader("Upload Holiday CSV", type=["csv"])

butt = st.sidebar.button('SUBMIT')

def button_False():
    butt = False

if not (main_data_file is not None and data_file is not None):
    st.error("Please upload both the files.")

else:
    main_file_details = {"filename":main_data_file.name, "filetype":main_data_file.type,
                            "filesize":main_data_file.size}
    # st.write(main_file_details)
    all_data = pd.read_csv(main_data_file)
    # st.dataframe(all_data)

    file_details = {"filename":data_file.name, "filetype":data_file.type,
                            "filesize":data_file.size}
    # st.write(file_details)
    holiday_data = pd.read_csv(data_file)
    # st.dataframe(holiday_data)
    
    filter_list = [ i for i in all_data.columns if "CODE" in i]

    try:
        filt = st.sidebar.selectbox("Select a Filter", filter_list, on_change=button_False)
        filter_options = all_data[filt].unique()
        filt_opt = st.sidebar.selectbox("Select a Filter", filter_options, on_change=button_False)
        sales_type = st.sidebar.radio(
            "Select Sales", ('Daily', 'Monthly'), on_change=button_False
            )
        
        sel_type = st.sidebar.radio(
        "Select between SARIMA or ARIMA",
        ('ARIMA', 'SARIMA'), on_change=button_False
        )
        
        rima_type = st.sidebar.radio("Select " + str(sel_type), ('Monthly', 'Daily'), on_change=button_False
            )
        if butt:
            all_data_2=all_data.copy()
            all_data_int=all_data_2[all_data_2[filt] == filt_opt]
            all_data_2=all_data_int
        #     print(len(all_data_2))
            # st.write(filter,"FILTER DONE") 
            ##################################################
            # st.dataframe(all_data_2)

            ## Daily Sales

            all_data_3=all_data_2[["Date","Sales_quantity"]].groupby("Date").sum().sort_values("Sales_quantity").reset_index()
            all_data_4 = date_data.merge(all_data_3,how="left",on="Date")
            all_data_4=all_data_4[["Date","Sales_quantity"]]
            all_data_4=all_data_4.fillna(0)
            # all_data_4
            all_data_5=all_data_4.merge(holiday_data,how="left",on="Date")
            all_data_5=all_data_5[["Date","Sales_quantity","StartDate","EndDate"]]
            all_data_5=all_data_5.ffill()
            # all_data_5

            all_data_5=all_data_5.fillna("01-01-2014") 
            all_data_5["Date"]=pd.to_datetime(all_data_5['Date'], format='%d-%m-%Y')
            all_data_5["StartDate"]=pd.to_datetime(all_data_5['StartDate'], format='%d-%m-%Y')
            all_data_5["EndDate"]=pd.to_datetime(all_data_5['EndDate'], format='%d-%m-%Y')

            # Holiday
            all_data_5["Date_Flag"] = np.where((all_data_5["Date"]>=all_data_5["StartDate"]) & (all_data_5["Date"]<=all_data_5["EndDate"]),1,0)
            all_data_5['Date_Flag'].value_counts()
            all_data_5.columns=['Date', 'Sales', 'StartDate', 'EndDate', 'Holiday_Flag']
            
            # st.dataframe(all_data_5)

            ###### Daily Data

            Daily_Data = all_data_5[['Date','Sales']]
            Daily_Data=Daily_Data.set_index('Date')  

            ###### Monthly Data
            Monthly_Data=pd.DataFrame(Daily_Data.groupby(by=[Daily_Data.index.year,Daily_Data.index.month])["Sales"].sum())
            Monthly_Data.reset_index(level=1,inplace=True)

            Monthly_Data.columns=["Month","Sales"]
            Monthly_Data.reset_index(inplace=True)
            Monthly_Data.columns=["Year","Month","Sales"]
            Monthly_Data['Date']="01-"+Monthly_Data['Month'].astype(str)+"-"+Monthly_Data['Year'].astype(str)
            Monthly_Data["Date"]=pd.to_datetime(Monthly_Data['Date'], format='%d-%m-%Y')
            Monthly_Data=Monthly_Data[["Date","Sales"]]
            Monthly_Data=Monthly_Data.set_index('Date')
            
            if sales_type == 'Daily' and butt:
                st.write("# Sales Data")
                fig, ax = plt.subplots()
                ax.plot(Daily_Data["Sales"])
                ax.set_xlabel('Date')
                ax.set_ylabel('Sales')
                ax.set_title('Daily Time Series Plot')
                st.pyplot(fig)

            if sales_type == 'Monthly' and butt:
                st.write("# Sales Data")
                fig, ax = plt.subplots()
                ax.plot(Monthly_Data["Sales"])
                ax.set_xlabel('Date')
                ax.set_ylabel('Sales')
                ax.set_title('Monthly Time Series Plot')
                st.pyplot(fig)  

            if sel_type == 'ARIMA' and butt:
                if rima_type == 'Daily' and butt:
                    daily_train_df=Daily_Data.head(2000)
                    daily_test_df=Daily_Data.tail(189)

                    daily_train = daily_train_df["Sales"]
                    daily_test = daily_test_df["Sales"]
                    # fitting a stepwise model:
                    rs_fit = pm.auto_arima(daily_train, start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                                           start_P=0, seasonal=False, d=1, D=1, trace=True,
                                           n_jobs=-1,  # We can run this in parallel by controlling this option
                                           error_action='ignore',  # don't want to know if an order does not work
                                           suppress_warnings=True,  # don't want convergence warnings
                                           stepwise=False, random=True, random_state=42,  # we can fit a random search (not exhaustive)
                                           n_fits=10)


                    daily_predictions = rs_fit.predict(n_periods=189)
                    daily_predictions_1=[math.ceil(i) if i>0 else 0 for i in daily_predictions]
                    daily_test_df["Predictions"]=daily_predictions_1

                    fig, ax = plt.subplots()
                    ax.plot(daily_train, label='training')
                    ax.plot(daily_test, label='actual')
                    ax.plot(daily_test_df['Predictions'], label='forecast')
                    # plt.fill_between(lower_series.index, lower_series, upper_series, 
                    #                  color='k', alpha=.15)
                    ax.set_title('Forecast vs Actuals')
                    legend = ax.legend(loc='upper left', fontsize=8)


                    # plt.show()
                    st.pyplot(fig)

                    final_df=pd.DataFrame(daily_test_df.groupby(by=[daily_test_df.index.year,daily_test_df.index.month])[["Sales","Predictions"]].sum())
                    final_df.reset_index(level=1,inplace=True)

                    final_df.columns=["Month","Sales","Predictions"]
                    final_df["Lower"]=0.95*final_df["Predictions"]
                    final_df["Upper"]=1.05*final_df["Predictions"]
                    final_df["Monthly_Accuracy"]=100-np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    final_df.reset_index(inplace=True)
                    st.dataframe(final_df, 900, 900)

                if rima_type == 'Monthly' and butt:
                    monthly_train_df=Monthly_Data.head(60)
                    monthly_test_df=Monthly_Data.tail(12)

                    monthly_train = monthly_train_df["Sales"]
                    monthly_test = monthly_test_df["Sales"]
                    # fitting a stepwise model:
                    rs_fit = pm.auto_arima(monthly_train, start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                                           start_P=0, seasonal=False, d=1, D=1, trace=True,
                                           n_jobs=-1,  # We can run this in parallel by controlling this option
                                           error_action='ignore',  # don't want to know if an order does not work
                                           suppress_warnings=True,  # don't want convergence warnings
                                           stepwise=False, random=True, random_state=42,  # we can fit a random search (not exhaustive)
                                           n_fits=10)


                    monthly_predictions = rs_fit.predict(n_periods=12)
                    
                    monthly_predictions_1=[math.ceil(i) if i>0 else 0 for i in monthly_predictions]
                    monthly_test_df["Predictions"]=monthly_predictions_1

                    fig, ax = plt.subplots()
                    ax.plot(monthly_train, label='training')
                    ax.plot(monthly_test, label='actual')
                    ax.plot(monthly_test_df['Predictions'], label='forecast')
                    # plt.fill_between(lower_series.index, lower_series, upper_series, 
                    #                  color='k', alpha=.15)
                    ax.set_title('Forecast vs Actuals')
                    legend = ax.legend(loc='upper left', fontsize=8)

                    # plt.show()
                    st.pyplot(fig)

                    final_df=monthly_test_df
                    # final_df.columns=["Year","Month","Sales","Predictions"]
                    final_df["Lower"]=0.95*final_df["Predictions"]
                    final_df["Upper"]=1.05*final_df["Predictions"]
                    final_df["Monthly_Accuracy"]=100-np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    st.dataframe(final_df, 900, 900)




            if sel_type == 'SARIMA' and butt:
                if rima_type == 'Daily' and butt:
                    daily_train_df=Daily_Data.head(2000)
                    daily_test_df=Daily_Data.tail(189)

                    daily_train = daily_train_df["Sales"]
                    daily_test = daily_test_df["Sales"]
                    # fitting a stepwise model:
                    rs_fit = pm.auto_arima(daily_train, start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                                           start_P=0, seasonal=True, d=1, D=1, trace=True,
                                           n_jobs=-1,  # We can run this in parallel by controlling this option
                                           error_action='ignore',  # don't want to know if an order does not work
                                           suppress_warnings=True,  # don't want convergence warnings
                                           stepwise=False, random=True, random_state=42,  # we can fit a random search (not exhaustive)
                                           n_fits=10)


                    daily_predictions = rs_fit.predict(n_periods=189)

                    daily_predictions_1=[math.ceil(i) if i>0 else 0 for i in daily_predictions]
                    daily_test_df["Predictions"]=daily_predictions_1

                    fig, ax = plt.subplots()
                    ax.plot(daily_train, label='training')
                    ax.plot(daily_test, label='actual')
                    ax.plot(daily_test_df['Predictions'], label='forecast')
                    # plt.fill_between(lower_series.index, lower_series, upper_series, 
                    #                  color='k', alpha=.15)
                    ax.set_title('Forecast vs Actuals')
                    legend = ax.legend(loc='upper left', fontsize=8)

                    final_df=pd.DataFrame(daily_test_df.groupby(by=[daily_test_df.index.year,daily_test_df.index.month])[["Sales","Predictions"]].sum())
                    final_df.reset_index(level=1,inplace=True)

                    final_df.columns=["Month","Sales","Predictions"]
                    final_df["Lower"]=0.95*final_df["Predictions"]
                    final_df["Upper"]=1.05*final_df["Predictions"]
                    final_df["Monthly_Accuracy"]=100-np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    final_df.reset_index(inplace=True)
                    st.dataframe(final_df, 900, 900)

                if rima_type == 'Monthly' and butt:
                    monthly_train_df=Monthly_Data.head(60)
                    monthly_test_df=Monthly_Data.tail(12)

                    monthly_train = monthly_train_df["Sales"]
                    monthly_test = monthly_test_df["Sales"]
                    # fitting a stepwise model:
                    rs_fit = pm.auto_arima(monthly_train, start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                                           start_P=0, seasonal=True, d=1, D=1, trace=True,
                                           n_jobs=-1,  # We can run this in parallel by controlling this option
                                           error_action='ignore',  # don't want to know if an order does not work
                                           suppress_warnings=True,  # don't want convergence warnings
                                           stepwise=False, random=True, random_state=42,  # we can fit a random search (not exhaustive)
                                           n_fits=10)


                    monthly_predictions = rs_fit.predict(n_periods=12)

                    monthly_predictions_1=[math.ceil(i) if i>0 else 0 for i in monthly_predictions]
                    monthly_test_df["Predictions"]=monthly_predictions_1

                    fig, ax = plt.subplots()
                    ax.plot(monthly_train, label='training')
                    ax.plot(monthly_test, label='actual')
                    ax.plot(monthly_test_df['Predictions'], label='forecast')
                    # plt.fill_between(lower_series.index, lower_series, upper_series, 
                    #                  color='k', alpha=.15)
                    ax.set_title('Forecast vs Actuals')
                    legend = ax.legend(loc='upper left', fontsize=8)

                    # plt.show()
                    st.pyplot(fig)

                    final_df=monthly_test_df
                    # final_df.columns=["Year","Month","Sales","Predictions"]
                    final_df["Lower"]=0.95*final_df["Predictions"]
                    final_df["Upper"]=1.05*final_df["Predictions"]
                    final_df["Monthly_Accuracy"]=100-np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    st.dataframe(final_df, 900, 900)

    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )


