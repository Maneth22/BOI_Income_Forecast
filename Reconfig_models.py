import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, \
    mean_squared_log_error
import statsmodels.api as sm
from matplotlib.pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from bs4 import BeautifulSoup
import requests
import datetime
import joblib
import pmdarima as pm


def pick_Best_model(main_df, column_array):
    Lin_models_array = []
    Ols_models_array = []

    Lin_Score_array = []
    Ols_Score_array = []

    for i in range(len(column_array)):
        Xlin = main_df[['Mean Currency rate', 'unemployment']].values
        y = main_df[column_array[i]].values

        X_train, X_test, y_train, y_test = train_test_split(Xlin, y, test_size=0.25, random_state=10)

        LR_ex = LinearRegression()
        LR_Model = LR_ex.fit(X_train, y_train)

        predictions = LR_ex.predict(X_test)
        Lin_rScore = r2_score(y_test, predictions)

        Lin_Score_array.append(Lin_rScore)
        Lin_models_array.append(LR_Model)

        Yols = main_df[column_array[i]]
        Xols = main_df[['Mean Currency rate', 'unemployment']]
        Xols = sm.add_constant(Xols)

        model = sm.OLS(Yols, Xols)
        olsModel = model.fit()
        ols_rScore = olsModel.rsquared

        Ols_Score_array.append(ols_rScore)
        Ols_models_array.append(olsModel)

    LR_scoreMean = np.mean(Lin_Score_array)
    Ols_Score_array = np.mean(Ols_Score_array)
    file_path = 'Models/modeltype.txt'
    for i in range(len(column_array)):
        column_name = column_array[i]
        filename = 'models/' + column_name[:8] + '.joblib'

        if Ols_Score_array > LR_scoreMean:
            model = Ols_models_array[i]

            with open(file_path, 'w') as file:
                file.write("ols")


        else:
            model = Lin_models_array[i]

            with open(file_path, 'w') as file:
                file.write("lin")
            print("lin")
        saveModel(model, filename)

    print("ols :" + str(Ols_Score_array) + "\n lin :" + str(LR_scoreMean))


def sarima(df):
    SARIMA_model = pm.auto_arima(df.iloc[:, 0], start_p=1, start_q=1,
                                 test='adf',
                                 max_p=3, max_q=3,
                                 m=4,  # 12 is the frequncy of the cycle
                                 start_P=0,
                                 seasonal=True,  # set to seasonal
                                 d=None,
                                 D=1,  # order of the seasonal differencing
                                 trace=False,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)

    return SARIMA_model


def forecast(df, ARIMA_model, periods=10):
    # Forecast
    n_periods = periods
    fitted, confint = ARIMA_model.predict(n_periods=n_periods, return_conf_int=True)

    last_date = df.index[-2]
    index_of_fc = pd.date_range(start=last_date, periods=n_periods, freq='Y')

    formatted_index = index_of_fc.strftime('%d-%m-%Y')

    fitted_series = pd.Series(fitted.values, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    lower_series.iloc[0] = df.iloc[:, 0].iloc[-1]
    upper_series.iloc[0] = df.iloc[:, 0].iloc[-1]
    fitted_series.iloc[0] = df.iloc[:, 0].iloc[-1]
    print(index_of_fc)

    # Plot
    plt.figure(figsize=(15, 7))
    plt.plot(df.iloc[:, 0], color='#1f76b4', label='Actual')

    plt.fill_between(lower_series.index, lower_series, upper_series, color='green', alpha=.15,
                     label='Confidence Interval')
    plt.plot(fitted_series, color='darkgreen', label='Forecast')
    plt.title("ARIMA/SARIMA - Forecast with Daily Seasonality")
    plt.xlabel("Date")
    plt.ylabel(df.columns)
    plt.legend()


    return lower_series, fitted_series, upper_series

def dateTimeIndex(df):
    df['date'] = df.index

    df['date'] = pd.to_datetime(df['date'], format='%Y')
    df = df.set_index(df['date'])
    df = df.drop('date', axis=1)
    return df


def get_mean(link):
    r = requests.get(link)
    soup = BeautifulSoup(r.content, 'html.parser')

    s = soup.find('table', class_='history-rates-data')
    rows = s.find_all('tr')

    mean_curr = []

    for row in rows:
        columns = [item for item in row.find_all('td') if 'month-footer' not in item.get('class', [])]
        if columns:
            row_date = columns[0].text.strip()
            mod_date = row_date.split('\n')
            # print(mod_date[0])
            if mod_date[0] >= str(datetime.date):
                row_num = columns[1].text.strip().split('\n')
                row_value = row_num[0].split(' ')
                value = row_value[3]
                value = float(value)
                mean_curr.append(value)

            else:
                continue
    value_mean = np.mean(mean_curr)

    return value_mean



def saveModel(model, filename):
    joblib.dump(model, filename)

def Reconfig():
    df_columns = ["Food, Beverages and Tobacco Products", "Textile, Wearing Apparel and\nLeather Products",
                  "Wood and Wood Products", "Paper, Paper Products, Printing\nand Publishing"
        , "Chemicals, Petroleum, Coal, Rubber\nand Plastic Products", "Non-Metallic Mineral Products",
                  "Fabricated Metal, Machinery and Transport Equipment",
                  "Manufactured Products (Not Elsewhere Specified)",
                  "Services and Infrastructure"]

    df = pd.read_csv('Resources/Dataset/boi_df_sector.csv')
    df_ext = pd.read_csv('Resources/Dataset/external_factors.csv')

    df_ext['Year'] = pd.to_datetime(df_ext['Year'])
    df_ext = df_ext.set_index('Year')
    df_ext.index = df_ext.index.year

    df = df.set_index('Year')

    df['unemployment'] = df_ext['unemployment']
    df['Export/Import_ratio'] = df_ext['Export/Import_ratio']

    today = datetime.date.today()
    month_in_words = today.strftime('%B')

    yesterday = today - datetime.timedelta(days=1)
    day = yesterday.strftime('%d')
    Year = yesterday.strftime('%y')
    date = str(month_in_words) + " " + str(int(day)) + ", 20" + str(Year)

    gm2022 = get_mean('https://www.exchange-rates.org/exchange-rate-history/usd-lkr-2022')
    gm2023 = get_mean('https://www.exchange-rates.org/exchange-rate-history/usd-lkr-2023')
    print(gm2023)
    df_MCrate = pd.DataFrame(df['Mean Currency rate'])
    df_unemployment = pd.DataFrame(df['unemployment'])

    df_MCrate = dateTimeIndex(df_MCrate)
    df_unemployment = dateTimeIndex(df_unemployment)

    sarima_un = sarima(df_unemployment)
    forecast(df_unemployment, sarima_un)

    val = df_MCrate.iloc[:, 0]
    date = df_MCrate.index

    n_val = []

    n_val = np.append(n_val, gm2022)
    n_val = np.append(n_val, gm2023)

    date_new = []

    date_new = np.append(date_new, pd.to_datetime('2022-01-01'))
    date_new = np.append(date_new, pd.to_datetime('2023-01-01'))
    formatted_dates = np.array([d.strftime('%Y-%m-%d') for d in date_new])

    temp_data = {'value': n_val, 'date': formatted_dates}

    n_dataframe = pd.DataFrame(temp_data, index=formatted_dates)

    data_now = {'date': date,
                'value': val}

    df_new_ = pd.DataFrame(data_now)
    df_new_['date'] = np.array([d.strftime('%Y-%m-%d') for d in df_new_.index])

    df_new_ = df_new_._append(n_dataframe, ignore_index=True)
    df_new_ = df_new_.set_index('date')

    sarima_cr = sarima(df_new_)
    sarima_up = sarima(df_unemployment)

    forecast_currency_Lower, forecast_currency_fitted, forecast_currency_Upper = forecast(df_MCrate, sarima_cr)
    forecast_emp_Lower, forecast_emp_fitted, forecast_emp_Upper = forecast(df_unemployment, sarima_up)

    data = {
        'forecast_currency_Upper': forecast_currency_Upper,
        'forecast_currency_fitted': forecast_currency_fitted,
        'forecast_currency_Lower': forecast_currency_Lower
    }
    df_mean = pd.DataFrame(data)
    df_mean = df_mean.dropna()

    df_mean = df_mean.drop(df_mean[df_mean.index < '2022-12-31'].index)
    df_mean = df_mean.drop(df_mean[df_mean.index > '2029-12-31'].index)
    df_mean.to_csv('Resources/Dataset/MeanCurrency.csv')

    data_emp = {
        'forecast_emp_Upper': forecast_emp_Upper,
        'forecast_emp_fitted': forecast_emp_fitted,
        'forecast_emp_Lower': forecast_emp_Lower
    }
    df_emp = pd.DataFrame(data_emp)
    df_emp = df_emp.dropna()
    df_emp = df_emp.drop(df_emp[df_emp.index < '2022-12-31'].index)
    df_emp.to_csv('Resources/Dataset/Unemployment_predictions.csv')
    pick_Best_model(df, df_columns)


Reconfig()



