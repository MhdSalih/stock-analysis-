import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import altair as alt
import math
import pandas as pd
from datetime import datetime, timedelta
import pmdarima as pm
from urllib.error import URLError
from fbprophet import Prophet

date_data = pd.read_csv("Date_data.csv")

file1, file2 = st.sidebar.columns(2) # to align the input box 

# main csv file
with file1:
    main_data_file = st.file_uploader("Upload Main CSV", type=["csv"])
# holiday csv file
with file2:
    data_file = st.file_uploader("Upload Holiday CSV", type=["csv"])

def button_False():
    butt = False

if not (main_data_file is not None and data_file is not None):
    st.error("Please upload both the files.")

else:
    main_file_details = {"filename":main_data_file.name, "filetype":main_data_file.type, "filesize":main_data_file.size}
    # st.write(main_file_details)
    all_data = pd.read_csv(main_data_file)
    # st.dataframe(all_data)

    file_details = {"filename":data_file.name, "filetype":data_file.type, "filesize":data_file.size}
    # st.write(file_details)
    holiday_data = pd.read_csv(data_file)
    # st.dataframe(holiday_data)
    
    filter_list = [ i for i in all_data.columns if "CODE" in i]
    holiday_list = holiday_data["Name"].unique()


    try:
        filt = st.sidebar.selectbox("Select a filter", filter_list, on_change=button_False)
        holt = st.sidebar.multiselect("Select a holiday", holiday_list, on_change=button_False)

        filter_options = all_data[filt].unique()
        
        filcol1, filcol2 = st.sidebar.columns(2) # for alignment
        
        with filcol1:
            filt_opt = st.selectbox("Select an Option", filter_options, on_change=button_False)

        with filcol2:
            sales_type = st.radio(
            "Select level of Sales", ('Daily', 'Monthly'), on_change=button_False
            )
        
        col1, col2 = st.sidebar.columns(2) # for alignment

        with col1:
            sel_type = st.radio("Select a model",('ARIMA', 'SARIMA', 'FB PROPHET'), on_change=button_False)
        
        with col2:
            rima_type = st.radio("Accuracy Level", ('Daily', 'Monthly'), on_change=button_False)

        butt = st.sidebar.button('SUBMIT')  

        if butt:
            holiday_data=holiday_data[holiday_data["Name"].isin(holt)]
            all_data_2=all_data.copy()
            all_data_int=all_data_2[all_data_2[filt] == filt_opt]
            all_data_2=all_data_int
            # st.dataframe(all_data_2)

            ## Daily Sales Formation

            all_data_3=all_data_2[["Date","Sales_quantity"]].groupby("Date").sum().sort_values("Sales_quantity").reset_index()
            all_data_4 = date_data.merge(all_data_3,how="left",on="Date")
            all_data_4=all_data_4[["Date","Sales_quantity"]]
            all_data_4=all_data_4.fillna(0)
            # all_data_4
            all_data_4_1=all_data_4.merge(holiday_data,how="left",on="Date")

            all_data_5=all_data_4_1[["Date","Sales_quantity","StartDate","EndDate"]]
            # all_data_5
            all_data_5=all_data_5.ffill()
            # all_data_5

            all_data_5=all_data_5.fillna("01-01-2014") 
            all_data_5["Date"]=pd.to_datetime(all_data_5['Date'], format='%d-%m-%Y')
            all_data_5["StartDate"]=pd.to_datetime(all_data_5['StartDate'], format='%d-%m-%Y')
            all_data_5["EndDate"]=pd.to_datetime(all_data_5['EndDate'], format='%d-%m-%Y')
            # all_data_5.head(10)
            all_data_5.columns=['Date', 'Sales', 'StartDate', 'EndDate']
            # all_data_5

            # # Holiday
            all_data_5["Date_Flag"] = np.where((all_data_5["Date"]>=all_data_5["StartDate"]) & (all_data_5["Date"]<=all_data_5["EndDate"]),1,0)
            all_data_5['Date_Flag'].value_counts()
            all_data_5.columns=['Date', 'Sales', 'StartDate', 'EndDate', 'Holiday_Flag']
            # st.dataframe(all_data_5)

            ###### Daily Data

            Daily_Data = all_data_5[['Date','Sales',"Holiday_Flag"]]
            Daily_Data=Daily_Data.set_index('Date')
            # Daily_Data

            ###### Monthly Data
            Monthly_Data=pd.DataFrame(Daily_Data.groupby(by=[Daily_Data.index.year,Daily_Data.index.month])[["Sales","Holiday_Flag"]].sum())
            Monthly_Data.reset_index(level=1,inplace=True)
            Monthly_Data.columns=["Month","Sales","Holiday_Flag"]
            Monthly_Data.reset_index(inplace=True)
            Monthly_Data.columns=["Year","Month","Sales","Holiday_Flag"]
            Monthly_Data['Date']="01-"+Monthly_Data['Month'].astype(str)+"-"+Monthly_Data['Year'].astype(str)
            Monthly_Data["Date"]=pd.to_datetime(Monthly_Data['Date'], format='%d-%m-%Y')
            Monthly_Data=Monthly_Data[["Date","Sales","Holiday_Flag"]]
            Monthly_Data=Monthly_Data.set_index('Date')

            sales_holiday_Data=all_data_4_1[["Date","Sales_quantity","Name","StartDate","EndDate"]]

            sales_holiday_Data["Date"]=pd.to_datetime(sales_holiday_Data['Date'], format='%d-%m-%Y')
            sales_holiday_Data["StartDate"]=pd.to_datetime(sales_holiday_Data['StartDate'], format='%d-%m-%Y')
            sales_holiday_Data["EndDate"]=pd.to_datetime(sales_holiday_Data['EndDate'], format='%d-%m-%Y')
            sales_holiday_Data=sales_holiday_Data.dropna()[['Date',"Sales_quantity","Name"]]
            # sales_holiday_Data

            sales_holiday_Data=sales_holiday_Data.reset_index(drop=True)
            Holiday_Filter=sales_holiday_Data.Name.unique()
            # print(Holiday_Filter)

            holiday_selection = holt
            sales_holiday_Data=sales_holiday_Data[sales_holiday_Data["Name"].isin(holiday_selection)]
            sales_holiday_Data=sales_holiday_Data.reset_index(drop=True)

            sales_holiday_Data['Flag']=[2*(i%3) for i in sales_holiday_Data.index ]
            # sales_holiday_Data.reset_index()["Date"][i]
            sales_holiday_Data["Month"]=[i.month for i in sales_holiday_Data["Date"]]
            # sales_holiday_Data
            holiday_colors=["red","blue","green","orange","yellow","white","magenta","purple","pink","brown"]
            holiday_color_df=pd.DataFrame(columns=["Holiday","Color"])


            holiday_color_df['Holiday']=holiday_data["Name"].unique()
            holiday_color_df['Color']=holiday_colors[0:len(holt)]
            holiday_color_df=holiday_color_df.set_index('Holiday')

            if sales_type == 'Daily' and butt:
                Daily_Data["Holiday_Flag"].replace(0, np.nan, inplace=True)
                st.write("# Sales Data")
                fig, ax = plt.subplots()
                ax.plot(Daily_Data["Sales"], color="orange")
                # ax.plot(Daily_Data["Holiday_Flag"]*10)
                ax.set_xlabel('Date')
                ax.set_ylabel('Sales')
                ax.set_title('Daily Time Series Plot')
                for i in range(0,len(sales_holiday_Data)):
                #     print(i)
                    holiday_name=sales_holiday_Data.reset_index()["Name"][i]
                    ax.text(sales_holiday_Data.reset_index()["Date"][i],4+sales_holiday_Data.reset_index()['Flag'][i]+1.5*sales_holiday_Data.reset_index()['Date'][i].month,sales_holiday_Data.reset_index()["Name"][i], color =holiday_color_df.loc[holiday_name][0])
                    ax.vlines(sales_holiday_Data.reset_index()["Date"][i],0,3.5+sales_holiday_Data.reset_index()['Flag'][i]+1.5*sales_holiday_Data.reset_index()['Date'][i].month, linestyle='dotted', colors =holiday_color_df.loc[holiday_name])
                st.pyplot(fig)

            if sales_type == 'Monthly' and butt:
                st.write("# Sales Data")
                fig, ax = plt.subplots()
                ax.plot(Monthly_Data["Sales"], color="orange")
                # ax.plot(Monthly_Data["Holiday_Flag"])
                ax.set_xlabel('Date')
                ax.set_ylabel('Sales')
                ax.set_title('Monthly Time Series Plot')
                for i in range(0,len(sales_holiday_Data)):
                #     print(i)
                    holiday_name=sales_holiday_Data.reset_index()["Name"][i]
                    ax.text(sales_holiday_Data.reset_index()["Date"][i],(50+sales_holiday_Data.reset_index()['Flag'][i]+1.5*sales_holiday_Data.reset_index()['Date'][i].month),sales_holiday_Data.reset_index()["Name"][i], color =holiday_color_df.loc[holiday_name][0])
                    ax.vlines(sales_holiday_Data.reset_index()["Date"][i],0,48+sales_holiday_Data.reset_index()['Flag'][i]+1.5*sales_holiday_Data.reset_index()['Date'][i].month, linestyle='dotted', colors =holiday_color_df.loc[holiday_name])
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
                                           n_fits=25)


                    daily_predictions = rs_fit.predict(n_periods=189)
                    daily_predictions_1=[math.ceil(i) if i>0 else 0 for i in daily_predictions]
                    daily_test_df["Predictions"]=daily_predictions_1

                    fig, ax = plt.subplots()
                    ax.plot(daily_train, label='training')
                    ax.plot(daily_test, label='actual')
                    ax.plot(daily_test_df['Predictions'], label='forecast')
                    ax.set_title('Forecast vs Actuals')
                    legend = ax.legend(loc='upper left', fontsize=8)
                    st.pyplot(fig)

                    final_df=pd.DataFrame(daily_test_df.tail(60))[["Sales","Predictions"]]
                    final_df.reset_index(inplace=True)
                    final_df.columns=["Date","Sales","Predictions"]
                    final_df["Lower"]=0.95*final_df["Predictions"]
                    final_df["Upper"]=1.05*final_df["Predictions"]
                    final_df["Daily_Accuracy"]=100-np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    final_df["Daily_MAPE"]=np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    final_df["Daily_MAE"]=np.abs((final_df["Predictions"]-final_df["Sales"]))


                    st.write("Overall Error Percentage -",math.floor(100*np.abs(final_df["Predictions"].sum()-final_df["Sales"].sum())/final_df["Sales"].sum()),"%")
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
                                           n_fits=25)


                    monthly_predictions = rs_fit.predict(n_periods=12)                    
                    monthly_predictions_1=[math.ceil(i) if i>0 else 0 for i in monthly_predictions]
                    monthly_test_df["Predictions"]=monthly_predictions_1

                    fig, ax = plt.subplots()
                    ax.plot(monthly_train, label='training')
                    ax.plot(monthly_test, label='actual')
                    ax.plot(monthly_test_df['Predictions'], label='forecast')
                    ax.set_title('Forecast vs Actuals')
                    legend = ax.legend(loc='upper left', fontsize=8)
                    st.pyplot(fig)

                    final_df=monthly_test_df
                    final_df.columns=["Month","Sales","Predictions"]
                    final_df["Lower"]=0.95*final_df["Predictions"]
                    final_df["Upper"]=1.05*final_df["Predictions"]
                    final_df["Monthly_Accuracy"]=100-np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    final_df["Monthly_MAPE"]=np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    final_df["Monthly_MAE"]=np.abs((final_df["Predictions"]-final_df["Sales"]))

                    st.write("Overall Error Percentage -",math.floor(100*np.abs(final_df["Predictions"].sum()-final_df["Sales"].sum())/final_df["Sales"].sum()),"%")
                    st.dataframe(final_df.tail(7), 900, 900)


            if sel_type == 'SARIMA' and butt:
                if rima_type == 'Daily' and butt:
                    daily_train_df=Daily_Data.head(2000)
                    daily_test_df=Daily_Data.tail(189)
                    daily_train = daily_train_df["Sales"]
                    daily_test = daily_test_df["Sales"]
                    # fitting a stepwise model:
                    rs_fit = pm.auto_arima(daily_train, start_p=1, start_q=1, max_p=2, max_q=2, m=12,
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
                    ax.set_title('Forecast vs Actuals')
                    legend = ax.legend(loc='upper left', fontsize=8)
                    st.pyplot(fig)

                    final_df=pd.DataFrame(daily_test_df.tail(60))[["Sales","Predictions"]]
                    final_df.reset_index(inplace=True)
                    final_df.columns=["Date","Sales","Predictions"]
                    final_df["Lower"]=0.95*final_df["Predictions"]
                    final_df["Upper"]=1.05*final_df["Predictions"]
                    final_df["Daily_Accuracy"]=100-np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    final_df["Daily_MAPE"]=np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    final_df["Daily_MAE"]=np.abs((final_df["Predictions"]-final_df["Sales"]))

                    st.write("Overall Error Percentage -",math.floor(100*np.abs(final_df["Predictions"].sum()-final_df["Sales"].sum())/final_df["Sales"].sum()),"%")
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
                                           n_fits=25)


                    monthly_predictions = rs_fit.predict(n_periods=12)
                    monthly_predictions_1=[math.ceil(i) if i>0 else 0 for i in monthly_predictions]
                    monthly_test_df["Predictions"]=monthly_predictions_1

                    fig, ax = plt.subplots()
                    ax.plot(monthly_train, label='training')
                    ax.plot(monthly_test, label='actual')
                    ax.plot(monthly_test_df['Predictions'], label='forecast')
                    ax.set_title('Forecast vs Actuals')
                    legend = ax.legend(loc='upper left', fontsize=8)
                    st.pyplot(fig)

                    final_df=monthly_test_df
                    final_df.columns=["Month","Sales","Predictions"]
                    final_df["Lower"]=0.95*final_df["Predictions"]
                    final_df["Upper"]=1.05*final_df["Predictions"]
                    final_df["Monthly_Accuracy"]=100-np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    final_df["Monthly_MAPE"]=np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    final_df["Monthly_MAE"]=np.abs((final_df["Predictions"]-final_df["Sales"]))

                    st.write("Overall Error Percentage -",math.floor(100*np.abs(final_df["Predictions"].sum()-final_df["Sales"].sum())/final_df["Sales"].sum()),"%")
                    st.dataframe(final_df.tail(7), 900, 900)

            if sel_type == 'FB PROPHET' and butt:
                if rima_type == 'Daily' and butt:
                    daily_train_df=Daily_Data.head(2000)
                    daily_test_df=Daily_Data.tail(189)

                    daily_train = daily_train_df["Sales"]
                    daily_test = daily_test_df["Sales"]

                    daily_train_fb=pd.DataFrame(daily_train).reset_index(drop=False)
                    daily_test_fb=pd.DataFrame(daily_test).reset_index(drop=False)

                    daily_train_fb.columns=['ds','y']
                    daily_test_fb.columns=['ds','y']
                    # set how many days to forecast
                    # forecast_length = 30
                    # instantiate and fit the model
                    m = Prophet(daily_seasonality=True)
                    m.fit(daily_train_fb)
                    # create the prediction dataframe 'forecast_length' days past the fit data
                    # future = m.make_future_dataframe(periods=forecast_length)
                    # make the forecast to the end of the 'future' dataframe
                    daily_predictions_df = m.predict(daily_test_fb)
                    daily_predictions=daily_predictions_df['yhat']

                    # Forecast
                    # fc, se, conf = stepwise_fit.predict(6, alpha=0.05)  # 95% conf

                    # Make as pandas series
                    # fc_series = pd.Series(predictions, index=test.index)
                    # lower_series = pd.Series(conf[:, 0], index=test.index)
                    # upper_series = pd.Series(conf[:, 1], index=test.index)
                    daily_predictions_1=[math.ceil(i) if i>0 else 0 for i in daily_predictions]
                    daily_test_df["Predictions"]=daily_predictions_1


                    fig, ax = plt.subplots()
                    ax.plot(daily_train, label='training')
                    ax.plot(daily_test, label='actual')
                    ax.plot(daily_test_df['Predictions'], label='forecast')
                    ax.set_title('Forecast vs Actuals')
                    legend = ax.legend(loc='upper left', fontsize=8)
                    st.pyplot(fig)

                    final_df=pd.DataFrame(daily_test_df.tail(60))[["Sales","Predictions"]]
                    final_df.reset_index(inplace=True)
                    final_df.columns=["Date","Sales","Predictions"]
                    final_df["Lower"]=0.95*final_df["Predictions"]
                    final_df["Upper"]=1.05*final_df["Predictions"]
                    final_df["Daily_Accuracy"]=100-np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    final_df["Daily_MAPE"]=np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    final_df["Daily_MAE"]=np.abs((final_df["Predictions"]-final_df["Sales"]))

                    st.write("Overall Error Percentage -",math.floor(100*np.abs(final_df["Predictions"].sum()-final_df["Sales"].sum())/final_df["Sales"].sum()),"%")
                    st.dataframe(final_df, 900, 900)

                if rima_type == 'Monthly' and butt:
                    monthly_train_df=Monthly_Data.head(60)
                    monthly_test_df=Monthly_Data.tail(12)

                    monthly_train = monthly_train_df["Sales"]
                    monthly_test = monthly_test_df["Sales"]


                    monthly_train_fb=pd.DataFrame(monthly_train).reset_index(drop=False)
                    monthly_test_fb=pd.DataFrame(monthly_test).reset_index(drop=False)

                    monthly_train_fb.columns=['ds','y']
                    monthly_test_fb.columns=['ds','y']


                    m = Prophet()
                    m.fit(monthly_train_fb)
                    # create the prediction dataframe 'forecast_length' days past the fit data
                    # future = m.make_future_dataframe(periods=forecast_length)
                    # make the forecast to the end of the 'future' dataframe
                    monthly_predictions_df = m.predict(monthly_test_fb)
                    monthly_predictions=monthly_predictions_df['yhat']

                    monthly_predictions_1=[math.ceil(i) if i>0 else 0 for i in monthly_predictions]
                    monthly_test_df["Predictions"]=monthly_predictions_1

                    
                    fig, ax = plt.subplots()
                    ax.plot(monthly_train, label='training')
                    ax.plot(monthly_test, label='actual')
                    ax.plot(monthly_test_df['Predictions'], label='forecast')
                    ax.set_title('Forecast vs Actuals')
                    legend = ax.legend(loc='upper left', fontsize=8)
                    st.pyplot(fig)

                    final_df=monthly_test_df
                    final_df.columns=["Month","Sales","Predictions"]
                    final_df["Lower"]=0.95*final_df["Predictions"]
                    final_df["Upper"]=1.05*final_df["Predictions"]
                    final_df["Monthly_Accuracy"]=100-np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    final_df["Monthly_MAPE"]=np.abs((final_df["Predictions"]-final_df["Sales"])*100/final_df["Sales"])
                    final_df["Monthly_MAE"]=np.abs((final_df["Predictions"]-final_df["Sales"]))

                    st.write("Overall Error Percentage -",math.floor(100*np.abs(final_df["Predictions"].sum()-final_df["Sales"].sum())/final_df["Sales"].sum()),"%")
                    st.dataframe(final_df.tail(7), 900, 900)

    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )


















