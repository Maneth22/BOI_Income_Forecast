import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, \
    mean_squared_log_error
import statsmodels.api as sm
from matplotlib.pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from datetime import date, timedelta
import joblib
from Models import *

final_df_array = []
currency_set = pd.DataFrame(columns=['cFitted', 'cLower', 'cHigher'])
alted_array = []
name_array = []


def GetMostProfitable(new_df_array):
    meanValue = 0
    profitable_sector = ""
    for i in range(len(new_df_array)):
        current_df = new_df_array[i]
        if meanValue < current_df['fitted'].mean():
            meanValue = current_df['fitted'].mean()
            profitable_sector = current_df.name
    return profitable_sector


def coefficient_of_variation( new_df_array):
    for i in range(len(new_df_array)):
        current_df = new_df_array[i]
        meanVal = current_df['fitted'].mean()
        sd = current_df['fitted'].std()
        CV = (sd / meanVal) * 100
        CV = np.round(CV, 2)
        print(current_df.name + " " + str(CV) + "%")


def plotData(df_alted, df_var, columnName):
    X = df_alted.index
    y = df_alted[columnName]
    gap_value = []
    gap_index = [df_alted.index]
    gap_value.append(df_alted[columnName].iloc[-1])
    gap_value.append(df_var.iloc[0, 0])
    gap_data = pd.DataFrame({'value': gap_value}, index=[df_alted.index[-1], df_var.index[0]])

    meanVal = y.mean()
    sd = y.std()
    CV = (sd / meanVal) * 100
    CV = np.round(CV, 2)

    plot_data = {
        'Actual_data': {
            'index': X.astype(str).tolist(),
            'values': y.values.tolist()
        },
        'Forecast_data': {
            'index': df_var.index.astype(str).tolist(),
            'values': df_var['fitted'].values.tolist()
        },
        'up_bound': {
            'index': df_var.index.astype(str).tolist(),
            'values': df_var['higher'].values.tolist()
        },
        'low_bound': {
            'index': df_var.index.astype(str).tolist(),
            'values': df_var['lower'].values.tolist()
        },
        'Gap_data': {
            'index': gap_data.index.astype(str).tolist(),
            'values': gap_data['value'].values.tolist()
        }
    }

    """plt.figure(figsize=(15, 7))
    plt.plot(X, y, label='Real Data')
    plt.plot(gap_data.index, gap_data['value'], label='Gap', linestyle='--')
    plt.fill_between(df_var.index, df_var['lower'], df_var['higher'], color='green', alpha=.15,
                     label='Confidence Interval')
    plt.plot(df_var.index, df_var['fitted'], color='darkgreen', label='Forecast')
    plt.title("Income Generation Forecasting ||" + " Coefficient of Varience: " + str(CV) + "%", color='red')
    plt.xlabel("Date")
    plt.ylabel(columnName)
    plt.legend()
    plt.show()"""

    return plot_data


def newData(currency_set, new_df_array, df_dep):
    currency_set['cFitted'] = df_dep['forecast_currency_fitted']
    currency_set['cLower'] = df_dep['forecast_currency_Lower']
    currency_set['cHigher'] = df_dep['forecast_currency_Upper']

    for i in range(len(new_df_array)):
        dataframe = new_df_array[i]
        for j in range(len(dataframe)):
            dataframe['fitted'].iloc[j] = dataframe['fitted'].iloc[j] / currency_set['cFitted'].iloc[j]
            dataframe['lower'].iloc[j] = dataframe['lower'].iloc[j] / currency_set['cLower'].iloc[j]
            dataframe['higher'].iloc[j] = dataframe['higher'].iloc[j] / currency_set['cHigher'].iloc[j]
            alted_array.append(dataframe)
        name_array.append(dataframe.name)


def LoadModel(column_name):
    filename = 'Models/' + column_name[:8] + '.joblib'
    loaded_model = joblib.load(filename)
    return loaded_model


def sendData():
    df = pd.read_csv('Resources/Dataset/boi_df_sector.csv')
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df = df.set_index(df['Year'])
    df = df.drop('Year', axis=1)

    df_Mean_Currency = pd.read_csv('Resources/Dataset/MeanCurrency.csv')
    df_Mean_Currency.iloc[:, 0] = pd.to_datetime(df_Mean_Currency.iloc[:, 0])
    df_Mean_Currency = df_Mean_Currency.set_index(df_Mean_Currency.iloc[:, 0])
    df_Mean_Currency = df_Mean_Currency.drop(df_Mean_Currency.columns[0], axis=1)

    USD_df = pd.read_csv('Resources/Dataset/USD-boi-sector.csv')
    USD_df['Year'] = pd.to_datetime(USD_df['Year'])
    USD_df = USD_df.set_index(USD_df['Year'])
    USD_df = USD_df.drop('Year', axis=1)

    df_Unemployment = pd.read_csv('Resources/Dataset/Unemployment_predictions.csv')
    df_Unemployment.iloc[:, 0] = pd.to_datetime(df_Unemployment.iloc[:, 0])
    df_Unemployment = df_Unemployment.set_index(df_Unemployment.iloc[:, 0])

    Food_loaded_model = LoadModel('Food, Beverages and Tobacco Products')
    Textile_loaded_model = LoadModel('Textile, Wearing Apparel and\nLeather Products')
    Wood_loaded_model = LoadModel('Wood and Wood Products')
    Paper_loaded_model = LoadModel('Paper, Paper Products, Printing\nand Publishing')
    Chemicals_loaded_model = LoadModel('Chemicals, Petroleum, Coal, Rubber\nand Plastic Product')
    Mineral_loaded_model = LoadModel('Non-Metallic Mineral Products')
    Equipment_loaded_model = LoadModel('Fabricated Metal, Machinery and Transport Equipment')
    Manufactured_loaded_model = LoadModel('Manufactured Products (Not Elsewhere Specified)')
    Services_loaded_model = LoadModel('Services and Infrastructure')

    df_dep = pd.DataFrame(
        columns=['forecast_currency_Upper', 'forecast_currency_fitted', 'forecast_currency_Lower', 'forecast_emp_Upper',
                 'forecast_emp_fitted', 'forecast_emp_Lower'])
    df_dep['forecast_currency_Upper'] = df_Mean_Currency['forecast_currency_Upper']
    df_dep['forecast_currency_fitted'] = df_Mean_Currency['forecast_currency_fitted']
    df_dep['forecast_currency_Lower'] = df_Mean_Currency['forecast_currency_Lower']
    df_dep['forecast_emp_Upper'] = df_Unemployment['forecast_emp_Upper']
    df_dep['forecast_emp_fitted'] = df_Unemployment['forecast_emp_fitted']
    df_dep['forecast_emp_Lower'] = df_Unemployment['forecast_emp_Lower']
    print(df_dep.index)
    #df_dep = df_dep.set_index(df_Unemployment.iloc[:, 0])
    model_array = [Food_loaded_model, Textile_loaded_model, Wood_loaded_model, Paper_loaded_model,
                   Chemicals_loaded_model, Mineral_loaded_model, Equipment_loaded_model, Manufactured_loaded_model,
                   Services_loaded_model]

    index_arr = df_dep.index

    fitted_currencyEmp = df_dep[['forecast_currency_fitted', 'forecast_emp_fitted']].values
    higher_currencyEmp = df_dep[['forecast_currency_Upper', 'forecast_emp_Upper']].values
    lower_currencyEmp = df_dep[['forecast_currency_Lower', 'forecast_emp_Lower']].values

    fitted_currencyEmp_ols = df_dep[['forecast_currency_fitted', 'forecast_emp_fitted']]
    higher_currencyEmp_ols = df_dep[['forecast_currency_Upper', 'forecast_emp_Upper']]
    lower_currencyEmp_ols = df_dep[['forecast_currency_Lower', 'forecast_emp_Lower']]

    data_array = [fitted_currencyEmp, higher_currencyEmp, lower_currencyEmp]
    data_array_ols = [fitted_currencyEmp_ols, higher_currencyEmp_ols, lower_currencyEmp_ols]

    with open('Models/modeltype.txt', 'r') as file:
        content = file.read()
    if content=='ols':
        print('s')
        for i in range(len(model_array)):
            model = model_array[i]
            fitted_arr = []
            df_val = pd.DataFrame(columns=['fitted', 'higher', 'lower'])
            for j in range(len(data_array_ols)):
                added_val =sm.add_constant(data_array_ols[j])
                val = model.predict(added_val)
                fitted_arr.append(val)
            df_val['fitted'] = fitted_arr[0]
            df_val['higher'] = fitted_arr[1]
            df_val['lower'] = fitted_arr[2]
            final_df_array.append(df_val)
    else:
        for i in range(len(model_array)):
            model = model_array[i]
            fitted_arr = []
            df_val = pd.DataFrame(columns=['fitted', 'higher', 'lower'])
            for j in range(len(data_array)):
                print(data_array[j])
                val = model.predict(data_array[j])
                fitted_arr.append(val)
            df_val['fitted'] = fitted_arr[0]
            df_val['higher'] = fitted_arr[1]
            df_val['lower'] = fitted_arr[2]
            final_df_array.append(df_val)
    food = final_df_array[0]
    Textile = final_df_array[1]
    wood = final_df_array[2]
    paper = final_df_array[3]
    chemicals = final_df_array[4]
    Mineral = final_df_array[5]
    Equipment = final_df_array[6]
    Manufactured = final_df_array[7]
    Services = final_df_array[8]

    food = food.set_index(index_arr)
    Textile = Textile.set_index(index_arr)
    wood = wood.set_index(index_arr)
    paper = paper.set_index(index_arr)
    chemicals = chemicals.set_index(index_arr)
    Mineral = Mineral.set_index(index_arr)
    Equipment = Equipment.set_index(index_arr)
    Manufactured = Manufactured.set_index(index_arr)
    Services = Services.set_index(index_arr)

    food.name = 'Food, Beverages and Tobacco Products'
    Textile.name = 'Textile, Wearing Apparel and\nLeather Products'
    wood.name = 'Wood and Wood Products'
    paper.name = 'Paper, Paper Products, Printing\nand Publishing'
    chemicals.name = 'Chemicals, Petroleum, Coal, Rubber\nand Plastic Products'
    Mineral.name = 'Non-Metallic Mineral Products'
    Equipment.name = 'Fabricated Metal, Machinery and Transport Equipment'
    Manufactured.name = 'Manufactured Products (Not Elsewhere Specified)'
    Services.name = 'Services and Infrastructure'

    new_df_array = [food,
                    Textile,
                    wood,
                    paper,
                    chemicals,
                    Mineral,
                    Equipment,
                    Manufactured,
                    Services
                    ]

    newData(currency_set, new_df_array, df_dep)

    food_plotting_data = plotData(USD_df, food, name_array[0])
    Textile_plotting_data = plotData(USD_df, Textile, name_array[1])
    wood_plotting_data = plotData(USD_df, wood, name_array[2])
    paper_plotting_data = plotData(USD_df, paper, name_array[3])
    chemicals_plotting_data = plotData(USD_df, chemicals, name_array[4])
    Mineral_plotting_data = plotData(USD_df, Mineral, name_array[5])
    Equipment_plotting_data = plotData(USD_df, Equipment, name_array[6])
    Manufactured_plotting_data = plotData(USD_df, Manufactured, name_array[7])
    Services_plotting_data = plotData(USD_df, Services, name_array[8])
    Currency = currData(df_Mean_Currency)
    val = GetMostProfitable(new_df_array)
    print(val)

    json_array = [food_plotting_data, Textile_plotting_data, wood_plotting_data, paper_plotting_data,
                  chemicals_plotting_data, Mineral_plotting_data, Equipment_plotting_data, Manufactured_plotting_data,
                  Services_plotting_data, Currency]
    return json_array


def currData(df_Mean_Currency):
    plot_data = {
        'Forecast_data': {
            'index': df_Mean_Currency.index.astype(str).tolist(),
            'values': df_Mean_Currency['forecast_currency_fitted'].values.tolist()
        },
        'up_bound': {
            'index': df_Mean_Currency.index.astype(str).tolist(),
            'values': df_Mean_Currency['forecast_currency_Upper'].values.tolist()
        },
        'low_bound': {
            'index': df_Mean_Currency.index.astype(str).tolist(),
            'values': df_Mean_Currency['forecast_currency_Lower'].values.tolist()
        }
    }
    return plot_data


sendData()