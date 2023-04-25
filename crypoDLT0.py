import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import plot, savefig, figure
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Integer, Categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, Conv1D, Flatten
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_percentage_error,  mean_squared_error
import xgboost as xgb
import tensorflow as tf
from TaylorDiagram import TaylorDiagram
import time

#-------------------------=======================--------------------------------------
# Miscellaneous functions
#-------------------------=======================--------------------------------------
def mbv(y_true, y_predicted):#mean bias
    tt1 = sum(y_true - y_predicted)/len(y_true)
    tt2=np.sqrt(sum(np.abs((y_true - y_predicted)-tt1)**2)/(len(y_true)-1))
    return tt1/tt2

def nse(y_true, y_predicted):# compuute Nash-Sutcliffe coefficient of Efficiency (NSE)
    return 1-(sum((y_true - y_predicted)**2)/sum((y_true - np.mean(y_true))**2))

def add_features(data): # This module create and add a few features to the dataframe

    #print(type(data))
    #print(data.dtypes)
    data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
    data['year'] = data['Date'].dt.year
    fdata= data.iloc[:, [1,2,3,4,5,6]] 
    for m in range(2018,2022):
        data_no=fdata[(fdata['year']== m)]
        da_index=data_no.index
        for x in range(len(da_index)):
            fdata.loc[da_index[x],'w_Price']= round((fdata.loc[da_index[x],'year']* fdata.loc[da_index[x],'Price'])/sum(data_no['year']),4)
    sma10 = fdata['Price'].rolling(10).mean()
    fdata['SMA3'] = np.round(fdata['Price'].rolling(3).mean(), decimals=3)
    fdata['SMA10'] = np.round(sma10, decimals=3)
    fdata['SMA20'] = np.round(fdata['Price'].rolling(20).mean(), decimals=3)
    modPrice = fdata['Price'].copy()
    modPrice.iloc[0:10] = sma10[0:10]
    fdata['EMA10'] = np.round(modPrice.ewm(span=10, adjust=False).mean(), decimals=3)
    return fdata.dropna()

def divide_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    temp_data = scaler.fit_transform(data.values)
    #divide data using n_train-length
    train = temp_data[:n_train_length, :]
    test = temp_data[n_train_length:, :]
    s1, t1 = train[:, 1:11], train[:, 0] # s1 (features), t1 (target for training set)
    s2, t2 = test[:, 1:11], test[:, 0] # s2 (features), t2 (target for testing set)     
    return s1,t1,s2,t2

def create_gru_model(units):
    model = Sequential()
    model.add(GRU(units, input_shape=(1, 10),activation='tanh'))
    model.add(Dense(units, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1,activation='relu')) 
    model.compile(loss='mse', optimizer='RMSprop')
    return model

def create_mlp_model(units):
    model = Sequential()
    model.add(Dense(units, input_dim=10, activation='relu'))
    model.add(Dense(units,  activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1,activation='relu'))
    model.compile(loss='mse', optimizer='RMSprop')
    return model

def create_cnn_model(filters, units):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=2, activation='relu', input_shape=(10,1)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(units, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss="mse", optimizer='RMSprop')
    return model

def combine_lists(df1,df2,df3,df4,df5,df6,ls1,ls2,ls3,ls4,ls5,ls6):
    df1 = pd.DataFrame(data=np.transpose(df1), columns=['forcast'])
    df2 = pd.DataFrame(data=np.transpose(df2), columns=['forcast'])
    df3 = pd.DataFrame(data=np.transpose(df3), columns=['forcast'])
    df4 = pd.DataFrame(data=np.transpose(df4), columns=['forcast'])
    df5 = pd.DataFrame(data=np.transpose(df5), columns=['forcast'])
    df6 = pd.DataFrame(data=np.transpose(df6), columns=['forcast'])
    ls1 = pd.DataFrame(data=np.transpose(ls1), columns=['residual'])
    ls2 = pd.DataFrame(data=np.transpose(ls2), columns=['residual'])
    ls3 = pd.DataFrame(data=np.transpose(ls3), columns=['residual'])
    ls4 = pd.DataFrame(data=np.transpose(ls4), columns=['residual'])
    ls5 = pd.DataFrame(data=np.transpose(ls5), columns=['residual'])
    ls6 = pd.DataFrame(data=np.transpose(ls6), columns=['residual'])
    return [df1,  df2, df3, df4, df5, df6],[ls1,  ls2, ls3, ls4, ls5, ls6]

def add_columns_results(df,b,tp):
    if (b==1):
        if (tp==1):
            for index, row in df.iterrows():
                df.at[index,'model']='ADA';df.at[index,'bitcoin']='BTC'
        elif (tp==2):
            for index, row in df.iterrows():
                df.at[index,'model']='GBM';df.at[index,'bitcoin']='BTC'
        elif(tp==3):
            for index, row in df.iterrows():
                df.at[index,'model']='XGB';df.at[index,'bitcoin']='BTC'
        elif(tp==4):
            for index, row in df.iterrows():
                df.at[index,'model']='GRU';df.at[index,'bitcoin']='BTC'
        elif(tp==5):
            for index, row in df.iterrows():
                df.at[index,'model']='MLP';df.at[index,'bitcoin']='BTC'
        else:
            for index, row in df.iterrows():
                df.at[index,'model']='CNN';df.at[index,'bitcoin']='BTC'
    elif(b==2):
        if (tp==1):
            for index, row in df.iterrows():
                df.at[index,'model']='ADA';df.at[index,'bitcoin']='ETH'
        elif (tp==2):
            for index, row in df.iterrows():
                df.at[index,'model']='GBM';df.at[index,'bitcoin']='ETH'
        elif(tp==3):
            for index, row in df.iterrows():
                df.at[index,'model']='XGB';df.at[index,'bitcoin']='ETH'
        elif(tp==4):
            for index, row in df.iterrows():
                df.at[index,'model']='GRU';df.at[index,'bitcoin']='ETH'
        elif(tp==5):
            for index, row in df.iterrows():
                df.at[index,'model']='MLP';df.at[index,'bitcoin']='ETH'
        else:
            for index, row in df.iterrows():
                df.at[index,'model']='CNN';df.at[index,'bitcoin']='ETH'
    elif(b==3):
        if (tp==1):
            for index, row in df.iterrows():
                df.at[index,'model']='ADA';df.at[index,'bitcoin']='BNB'
        elif (tp==2):
            for index, row in df.iterrows():
                df.at[index,'model']='GBM';df.at[index,'bitcoin']='BNB'
        elif(tp==3):
            for index, row in df.iterrows():
                df.at[index,'model']='XGB';df.at[index,'bitcoin']='BNB'
        elif(tp==4):
            for index, row in df.iterrows():
                df.at[index,'model']='GRU';df.at[index,'bitcoin']='BNB'
        elif(tp==5):
            for index, row in df.iterrows():
                df.at[index,'model']='MLP';df.at[index,'bitcoin']='BNB'
        else:
            for index, row in df.iterrows():
                df.at[index,'model']='CNN';df.at[index,'bitcoin']='BNB'
    elif(b==4):
        if (tp==1):
            for index, row in df.iterrows():
                df.at[index,'model']='ADA';df.at[index,'bitcoin']='LTC'
        elif (tp==2):
            for index, row in df.iterrows():
                df.at[index,'model']='GBM';df.at[index,'bitcoin']='LTC'
        elif(tp==3):
            for index, row in df.iterrows():
                df.at[index,'model']='XGB';df.at[index,'bitcoin']='LTC'
        elif(tp==4):
            for index, row in df.iterrows():
                df.at[index,'model']='GRU';df.at[index,'bitcoin']='LTC'
        elif(tp==5):
            for index, row in df.iterrows():
                df.at[index,'model']='MLP';df.at[index,'bitcoin']='LTC'
        else:
            for index, row in df.iterrows():
                df.at[index,'model']='CNN';df.at[index,'bitcoin']='LTC'
    elif(b==5):
        if (tp==1):
            for index, row in df.iterrows():
                df.at[index,'model']='ADA';df.at[index,'bitcoin']='XLM'
        elif (tp==2):
            for index, row in df.iterrows():
                df.at[index,'model']='GBM';df.at[index,'bitcoin']='XLM'
        elif(tp==3):
            for index, row in df.iterrows():
                df.at[index,'model']='XGB';df.at[index,'bitcoin']='XLM'
        elif(tp==4):
            for index, row in df.iterrows():
                df.at[index,'model']='GRU';df.at[index,'bitcoin']='XLM'
        elif(tp==5):
            for index, row in df.iterrows():
                df.at[index,'model']='MLP';df.at[index,'bitcoin']='XLM'
        else:
            for index, row in df.iterrows():
                df.at[index,'model']='CNN';df.at[index,'bitcoin']='XLM'
    elif(b==6):
        if (tp==1):
            for index, row in df.iterrows():
                df.at[index,'model']='ADA';df.at[index,'bitcoin']='DOGE'
        elif (tp==2):
            for index, row in df.iterrows():
                df.at[index,'model']='GBM'; df.at[index,'bitcoin']='DOGE'
        elif(tp==3):
            for index, row in df.iterrows():
                df.at[index,'model']='XGB'; df.at[index,'bitcoin']='DOGE'
        elif(tp==4):
            for index, row in df.iterrows():
                df.at[index,'model']='GRU'; df.at[index,'bitcoin']='DOGE'
        elif(tp==5):
            for index, row in df.iterrows():
                df.at[index,'model']='MLP'; df.at[index,'bitcoin']='DOGE'
        else:
            for index, row in df.iterrows():
                df.at[index,'model']='CNN'; df.at[index,'bitcoin']='DOGE'     
    return df

def insert_list(lst, obj1,obj2,obj3,obj4,obj5,obj6):
    #insert objects into a list
    # lst - list
    # objects - obj1,obj2,obj3,obj4,obj5,and obj6
    lst.append(pd.DataFrame(data=np.transpose(obj1), columns=['forcast']))
    lst.append(pd.DataFrame(data=np.transpose(obj2), columns=['forcast']))
    lst.append(pd.DataFrame(data=np.transpose(obj3), columns=['forcast']))
    lst.append(pd.DataFrame(data=np.transpose(obj4), columns=['forcast']))
    lst.append(pd.DataFrame(data=np.transpose(obj5), columns=['forcast']))
    lst.append(pd.DataFrame(data=np.transpose(obj6), columns=['forcast']))
    return lst   

def concat_dataframes(df1,  df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19,  df20, df21, df22, df23, df24,df25,df26, df27, df28, df29, df30, df31,  df32, df33, df34, df35, df36):
    nan_value = 0
    # combine all dataframes containing results from predictors into one dataframe
    btc_df = pd.concat([df1,  df2, df3, df4, df5, df6], join='outer', axis=0).fillna(nan_value) # predictions from all models for BTC
    eth_df = pd.concat([df13, df14, df15, df16, df17, df18], join='outer', axis=0).fillna(nan_value) # predictions from all models for ETH
    bnb_df = pd.concat([df19,  df20, df21, df22, df23, df24], join='outer', axis=0).fillna(nan_value) # predictions from all models for BNB
    ltc_df = pd.concat([df7, df8, df9, df10, df11, df12], join='outer', axis=0).fillna(nan_value) # predictions from all models for LTC
    xlm_df = pd.concat([df31,  df32, df33, df34, df35, df36], join='outer', axis=0).fillna(nan_value)# predictions from all models for XLM
    doge_df = pd.concat([df25,df26,df27,df28,df29,df30], join='outer', axis=0).fillna(nan_value) # predictions from all models for DOGE
    return  pd.concat([btc_df,eth_df,bnb_df,ltc_df,xlm_df,doge_df], join='outer', axis=0).fillna(nan_value) #concatenate all result' dataframes

def compute_std_corrs(df,btcoin):
    # Compute standard deviations and correlation statistics for observations and predictions for the 6 cryptocurrencies
    if (btcoin=='BTC'):
        tar_=df[(df['model']== 'GBM') & (df['bitcoin']== 'BTC')].target.to_frame().reset_index(drop=True)
        ada_=df[(df['model']== 'ADA') & (df['bitcoin']== 'BTC')].forcast.to_frame().reset_index(drop=True)
        ada_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(ada_).values)
        ada_corr=round(ada_corr[0,1],5)
        gbm_=df[(df['model']== 'GBM') & (df['bitcoin']== 'BTC')].forcast.to_frame().reset_index(drop=True)
        gbm_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(gbm_).values)
        gbm_corr=round(gbm_corr[0,1],5)
        xgb_=df[(df['model']== 'XGB') & (df['bitcoin']== 'BTC')].forcast.to_frame().reset_index(drop=True)
        xgb_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(xgb_).values)
        xgb_corr=round(xgb_corr[0,1],5)
        gru_=df[(df['model']== 'GRU') & (df['bitcoin']== 'BTC')].forcast.to_frame().reset_index(drop=True)
        gru_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(gru_).values)
        gru_corr=round(gru_corr[0,1],5)
        mlp_=df[(df['model']== 'MLP') & (df['bitcoin']== 'BTC')].forcast.to_frame().reset_index(drop=True)
        mlp_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(mlp_).values)
        mlp_corr=round(mlp_corr[0,1],5)
        cnn_=df[(df['model']== 'CNN') & (df['bitcoin']== 'BTC')].forcast.to_frame().reset_index(drop=True)
        cnn_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(cnn_).values)
        cnn_corr=round(cnn_corr[0,1],5)
    elif(btcoin=='LTC'):
        tar_=df[(df['model']== 'GBM') & (df['bitcoin']== 'LTC')].target.to_frame().reset_index(drop=True)
        ada_=df[(df['model']== 'ADA') & (df['bitcoin']== 'LTC')].forcast.to_frame().reset_index(drop=True)
        ada_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(ada_).values)
        ada_corr=round(ada_corr[0,1],5)
        gbm_=df[(df['model']== 'GBM') & (df['bitcoin']== 'LTC')].forcast.to_frame().reset_index(drop=True)
        gbm_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(gbm_).values)
        gbm_corr=round(gbm_corr[0,1],5)
        xgb_=df[(df['model']== 'XGB') & (df['bitcoin']== 'LTC')].forcast.to_frame().reset_index(drop=True)
        xgb_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(xgb_).values)
        xgb_corr=round(xgb_corr[0,1],5)
        gru_=df[(df['model']== 'GRU') & (df['bitcoin']== 'LTC')].forcast.to_frame().reset_index(drop=True)
        gru_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(gru_).values)
        gru_corr=round(gru_corr[0,1],5)
        mlp_=df[(df['model']== 'MLP') & (df['bitcoin']== 'LTC')].forcast.to_frame().reset_index(drop=True)
        mlp_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(mlp_).values)
        mlp_corr=round(mlp_corr[0,1],5)
        cnn_=df[(df['model']== 'CNN') & (df['bitcoin']== 'LTC')].forcast.to_frame().reset_index(drop=True)
        cnn_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(cnn_).values)
        cnn_corr=round(cnn_corr[0,1],5)
    elif(btcoin=='ETH'):
        tar_=df[(df['model']== 'GBM') & (df['bitcoin']== 'ETH')].target.to_frame().reset_index(drop=True)
        ada_=df[(df['model']== 'ADA') & (df['bitcoin']== 'ETH')].forcast.to_frame().reset_index(drop=True)
        ada_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(ada_).values)
        ada_corr=round(ada_corr[0,1],5)
        gbm_=df[(df['model']== 'GBM') & (df['bitcoin']== 'ETH')].forcast.to_frame().reset_index(drop=True)
        gbm_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(gbm_).values)
        gbm_corr=round(gbm_corr[0,1],5)
        xgb_=df[(df['model']== 'XGB') & (df['bitcoin']== 'ETH')].forcast.to_frame().reset_index(drop=True)
        xgb_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(xgb_).values)
        xgb_corr=round(xgb_corr[0,1],5)
        gru_=df[(df['model']== 'GRU') & (df['bitcoin']== 'ETH')].forcast.to_frame().reset_index(drop=True)
        gru_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(gru_).values)
        gru_corr=round(gru_corr[0,1],5)
        mlp_=df[(df['model']== 'MLP') & (df['bitcoin']== 'ETH')].forcast.to_frame().reset_index(drop=True)
        mlp_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(mlp_).values)
        mlp_corr=round(mlp_corr[0,1],5)
        cnn_=df[(df['model']== 'CNN') & (df['bitcoin']== 'ETH')].forcast.to_frame().reset_index(drop=True)
        cnn_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(cnn_).values)
        cnn_corr=round(cnn_corr[0,1],5)
    elif(btcoin=='BNB'):
        tar_=df[(df['model']== 'GBM') & (df['bitcoin']== 'BNB')].target.to_frame().reset_index(drop=True)
        ada_=df[(df['model']== 'ADA') & (df['bitcoin']== 'BNB')].forcast.to_frame().reset_index(drop=True)
        ada_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(ada_).values)
        ada_corr=round(ada_corr[0,1],5)
        gbm_=df[(df['model']== 'GBM') & (df['bitcoin']== 'BNB')].forcast.to_frame().reset_index(drop=True)
        gbm_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(gbm_).values)
        gbm_corr=round(gbm_corr[0,1],5)
        xgb_=df[(df['model']== 'XGB') & (df['bitcoin']== 'BNB')].forcast.to_frame().reset_index(drop=True)
        xgb_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(xgb_).values)
        xgb_corr=round(xgb_corr[0,1],5)
        gru_=df[(df['model']== 'GRU') & (df['bitcoin']== 'BNB')].forcast.to_frame().reset_index(drop=True)
        gru_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(gru_).values)
        gru_corr=round(gru_corr[0,1],5)
        mlp_=df[(df['model']== 'MLP') & (df['bitcoin']== 'BNB')].forcast.to_frame().reset_index(drop=True)
        mlp_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(mlp_).values)
        mlp_corr=round(mlp_corr[0,1],5)
        cnn_=df[(df['model']== 'CNN') & (df['bitcoin']== 'BNB')].forcast.to_frame().reset_index(drop=True)
        cnn_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(cnn_).values)
        cnn_corr=round(cnn_corr[0,1],5) 
    elif(btcoin=='XLM'):
        tar_=df[(df['model']== 'GBM') & (df['bitcoin']== 'XLM')].target.to_frame().reset_index(drop=True)
        ada_=df[(df['model']== 'ADA') & (df['bitcoin']== 'XLM')].forcast.to_frame().reset_index(drop=True)
        ada_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(ada_).values)
        ada_corr=round(ada_corr[0,1],5)
        gbm_=df[(df['model']== 'GBM') & (df['bitcoin']== 'XLM')].forcast.to_frame().reset_index(drop=True)
        gbm_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(gbm_).values)
        gbm_corr=round(gbm_corr[0,1],5)
        xgb_=df[(df['model']== 'XGB') & (df['bitcoin']== 'XLM')].forcast.to_frame().reset_index(drop=True)
        xgb_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(xgb_).values)
        xgb_corr=round(xgb_corr[0,1],5)
        gru_=df[(df['model']== 'GRU') & (df['bitcoin']== 'XLM')].forcast.to_frame().reset_index(drop=True)
        gru_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(gru_).values)
        gru_corr=round(gru_corr[0,1],5)
        mlp_=df[(df['model']== 'MLP') & (df['bitcoin']== 'XLM')].forcast.to_frame().reset_index(drop=True)
        mlp_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(mlp_).values)
        mlp_corr=round(mlp_corr[0,1],5)
        cnn_=df[(df['model']== 'CNN') & (df['bitcoin']== 'XLM')].forcast.to_frame().reset_index(drop=True)
        cnn_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(cnn_).values)
        cnn_corr=round(cnn_corr[0,1],5)
    elif(btcoin=='DOGE'):
        tar_=df[(df['model']== 'GBM') & (df['bitcoin']== 'DOGE')].target.to_frame().reset_index(drop=True)
        ada_=df[(df['model']== 'ADA') & (df['bitcoin']== 'DOGE')].forcast.to_frame().reset_index(drop=True)
        ada_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(ada_).values)
        ada_corr=round(ada_corr[0,1],5)
        gbm_=df[(df['model']== 'GBM') & (df['bitcoin']== 'DOGE')].forcast.to_frame().reset_index(drop=True)
        gbm_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(gbm_).values)
        gbm_corr=round(gbm_corr[0,1],5)
        xgb_=df[(df['model']== 'XGB') & (df['bitcoin']== 'DOGE')].forcast.to_frame().reset_index(drop=True)
        xgb_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(xgb_).values)
        xgb_corr=round(xgb_corr[0,1],5)
        gru_=df[(df['model']== 'GRU') & (df['bitcoin']== 'DOGE')].forcast.to_frame().reset_index(drop=True)
        gru_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(gru_).values)
        gru_corr=round(gru_corr[0,1],5)
        mlp_=df[(df['model']== 'MLP') & (df['bitcoin']== 'DOGE')].forcast.to_frame().reset_index(drop=True)
        mlp_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(mlp_).values)
        mlp_corr=round(mlp_corr[0,1],5)
        cnn_=df[(df['model']== 'CNN') & (df['bitcoin']== 'DOGE')].forcast.to_frame().reset_index(drop=True)
        cnn_corr=np.corrcoef(np.transpose(tar_).values,np.transpose(cnn_).values)
        cnn_corr=round(cnn_corr[0,1],5) 
        
    models_stds=[float(round(np.std(ada_),2)),float(round(np.std(gbm_),2)),float(round(np.std(xgb_),2)),float(round(np.std(gru_),2)),float(round(np.std(mlp_),2)),float(round(np.std(cnn_),2)),float(round(np.std(tar_),2))]
    models_corrs=[ada_corr,gbm_corr,xgb_corr,gru_corr,mlp_corr,cnn_corr]
    return models_corrs, models_stds
#-------------------------=======================--------------------------------------
# End of miscellaneous functions
#-------------------------=======================--------------------------------------

#-------------------------=======================--------------------------------------
# Graphs plotting functions
#-------------------------=======================--------------------------------------  
def make_TaylorDiag_data(data):
    
    btc_corrs, btc_stds=compute_std_corrs(data,'BTC')
    ltc_corrs, ltc_stds=compute_std_corrs(data,'LTC')
    eth_corrs, eth_stds=compute_std_corrs(data,'ETH')
    bnb_corrs, bnb_stds=compute_std_corrs(data,'BNB')
    xlm_corrs, xlm_stds=compute_std_corrs(data,'XLM')
    doge_corrs, doge_stds=compute_std_corrs(data,'DOGE')

    btc_ref = btc_stds[6]
    btc_samples = [[btc_stds[0], btc_corrs[0], 'ADA'],[btc_stds[1], btc_corrs[1], 'GBM'],[btc_stds[2], btc_corrs[2], 'XGB'],[btc_stds[3], btc_corrs[3], 'GRU'],[btc_stds[4], btc_corrs[4], 'MLP'],[btc_stds[5], btc_corrs[5], 'CNN']]
    
    eth_ref = eth_stds[6]
    eth_samples = [[eth_stds[0], eth_corrs[0], 'ADA'],[eth_stds[1], eth_corrs[1], 'GBM'],
                   [eth_stds[2], eth_corrs[2], 'XGB'],[eth_stds[3], eth_corrs[3], 'GRU'],
                   [eth_stds[4], eth_corrs[4], 'MLP'],[eth_stds[5], eth_corrs[5], 'CNN']]

    bnb_ref = bnb_stds[6]
    bnb_samples = [[bnb_stds[0], bnb_corrs[0], 'ADA'],[bnb_stds[1], bnb_corrs[1], 'GBM'],
                   [bnb_stds[2], bnb_corrs[2], 'XGB'],[bnb_stds[3], bnb_corrs[3], 'GRU'],
                   [bnb_stds[4], bnb_corrs[4], 'MLP'],[bnb_stds[5], bnb_corrs[5], 'CNN']]
    
    ltc_ref = ltc_stds[6]
    ltc_samples = [[ltc_stds[0], ltc_corrs[0], 'ADA'],[ltc_stds[1], ltc_corrs[1], 'GBM'],
                   [ltc_stds[2], ltc_corrs[2], 'XGB'],[ltc_stds[3], ltc_corrs[3], 'GRU'],
                   [ltc_stds[4], ltc_corrs[4], 'MLP'],[ltc_stds[5], ltc_corrs[5], 'CNN']]

    xlm_ref = xlm_stds[6]
    xlm_samples = [[xlm_stds[0], xlm_corrs[0], 'ADA'],[xlm_stds[1], xlm_corrs[1], 'GBM'],
                   [xlm_stds[2], xlm_corrs[2], 'XGB'],[xlm_stds[3], xlm_corrs[3], 'GRU'],
                   [xlm_stds[4], xlm_corrs[4], 'MLP'],[xlm_stds[5], xlm_corrs[5], 'CNN']]

    doge_ref = doge_stds[6]
    doge_samples = [[doge_stds[0], doge_corrs[0], 'ADA'],[doge_stds[1], doge_corrs[1], 'GBM'],
                    [doge_stds[2], doge_corrs[2], 'XGB'],[doge_stds[3], doge_corrs[3], 'GRU'],
                    [doge_stds[4], doge_corrs[4], 'MLP'],[doge_stds[5], doge_corrs[5], 'CNN']]
    
    if (scenario==1):
        btc_title='BTC-USD (Scenario A)'
        eth_title='ETH-USD (Scenario A)'
        bnb_title='BNB-USD (Scenario A)'
        ltc_title='LTC-USD (Scenario A)'
        xlm_title='XLM-USD (Scenario A)'
        doge_title='DOGE-USD (Scenario A)'
    else:
        btc_title='BTC-USD (Scenario B)'
        eth_title='ETH-USD (Scenario B)'
        bnb_title='BNB-USD (Scenario B)'
        ltc_title='LTC-USD (Scenario B)'
        xlm_title='XLM-USD (Scenario B)'
        doge_title='DOGE-USD (Scenario B)'
    
    plot_Taylor(btc_ref,  btc_samples,btc_title)
    plot_Taylor(eth_ref,  eth_samples,eth_title)
    plot_Taylor(bnb_ref,  bnb_samples,bnb_title)
    plot_Taylor(ltc_ref,  ltc_samples,ltc_title)
    plot_Taylor(xlm_ref,  xlm_samples,xlm_title)
    plot_Taylor(doge_ref,  doge_samples,doge_title)
    return


def plot_Taylor(stdref, samples, tit):
    fig = plt.figure(figsize=(7, 7))
    td = TaylorDiagram(stdref, fig=fig, label='Reference')
    td.samplePoints[0].set_color('r')  # Mark reference point as a red star

    # Add models to Taylor diagram
    for i, (stddev, corrcoef, name) in enumerate(samples):
        td.add_sample(stddev, corrcoef,marker='$%d$' % (i+1), ms=10, ls='',mfc='k', mec='b',label=name)

    # Add RMS contours, and label them
    contours = td.add_contours(levels=5, colors='0.7')  # 5 levels in grey
    plt.clabel(contours, inline=1, fontsize=14, fmt='%.0f')

    td.add_grid()                                  # Add grid
    td._ax.axis[:].major_ticks.set_tick_out(True)  # Put ticks outward

    # Add a figure legend and title
    fig.legend(td.samplePoints,[ p.get_label() for p in td.samplePoints ],numpoints=1, loc='upper right')
    fig.suptitle(tit, size='x-large')  
    mod_name=tit[0:3]
    if (scenario==1):
        if(mod_name=='BTC'):
            fname='Fig 9A.png'
        elif(mod_name=='ETH'):
            fname='Fig 9B.png'
        elif(mod_name=='BNB'):
            fname='Fig 9C.png'
        elif(mod_name=='LTC'):
            fname='Fig 9D.png'
        elif(mod_name=='XLM'):
            fname='Fig 9E.png'
        elif(mod_name=='DOG'):
            fname='Fig 9F.png'
    elif(scenario==2):
        if(mod_name=='BTC'):
            fname='Fig 10A.png'
        elif(mod_name=='ETH'):
            fname='Fig 10B.png'
        elif(mod_name=='BNB'):
            fname='Fig 10C.png'
        elif(mod_name=='LTC'):
            fname='Fig 10D.png'
        elif(mod_name=='XLM'):
            fname='Fig 10E.png'
        elif(mod_name=='DOG'):
            fname='Fig 10F.png'
    savefig('../results/'+fname)
    return

def combine_plots_tree(line1, line2,line3, line4,model,endl,tit):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
    ax[0][0].plot(line1[0][endl:],'k+', markersize=4,label='BTC')
    ax[0][0].plot(line2[0][endl:], label=model[0],color='tab:brown', linewidth=1)
    ax[0][0].plot(line3[0][endl:], label=model[1],color='r', linewidth=1)
    ax[0][0].plot(line4[0][endl:], label=model[2],color='g',linestyle='dotted', marker='o',markersize=2, linewidth=1)
    ax[0][0].set_ylabel('price ($)', fontsize=12)
    ax[0][0].set_title(tit)
    ax[0][0].legend(loc='best',frameon=False, fontsize=9)

    ax[0][1].plot(line1[1][endl:],'k+', markersize=4, label='ETH')
    ax[0][1].plot(line2[1][endl:], label=model[0],color='tab:brown', linewidth=1)
    ax[0][1].plot(line3[1][endl:], label=model[1],color='r', linewidth=1)
    ax[0][1].plot(line4[1][endl:], label=model[2],color='g',linestyle='dotted', marker='o', markersize=2,linewidth=1)
    ax[0][1].legend(loc='best',frameon=False, fontsize=9)
    ax[0][1].set_title(tit)

    ax[1][0].plot(line1[2][endl:],'k+', markersize=4, label='BNB')
    ax[1][0].plot(line2[2][endl:], label=model[0], color='tab:brown',linewidth=1)
    ax[1][0].plot(line3[2][endl:], label=model[1], color='r',linewidth=1)
    ax[1][0].plot(line4[2][endl:], label=model[2],color='g',linestyle='dotted', marker='o',markersize=2, linewidth=1)
    ax[1][0].set_ylabel('price ($)', fontsize=12)
    ax[1][0].legend(loc='best',frameon=False, fontsize=9)
    
    ax[1][1].plot(line1[3][endl:],'k+', markersize=4, label='LTC')
    ax[1][1].plot(line2[3][endl:], label=model[0],color='tab:brown', linewidth=1)
    ax[1][1].plot(line3[3][endl:], label=model[1],color= 'r',linewidth=1)
    ax[1][1].plot(line4[3][endl:], label=model[2],color='g',linestyle='dotted', marker='o',markersize=2, linewidth=1)
    ax[1][1].legend(loc='best',frameon=False, fontsize=9)
    
    ax[2][0].plot(line1[4][endl:],'k+', markersize=4, label='XLM')
    ax[2][0].plot(line2[4][endl:], label=model[0], color='tab:brown',linewidth=1)
    ax[2][0].plot(line3[4][endl:], label=model[1],color='r', linewidth=1)
    ax[2][0].plot(line4[4][endl:], label=model[2],color='g',linestyle='dotted', marker='o',markersize=2, linewidth=1)
    ax[2][0].set_ylabel('price ($)', fontsize=12)
    ax[2][0].set_xlabel('time (days)', fontsize=12)
    ax[2][0].legend(loc='best',frameon=False, fontsize=9)
    
    ax[2][1].plot(line1[5][endl:], 'k+', markersize=4,label='DOGE')
    ax[2][1].plot(line2[5][endl:], label=model[0], color='tab:brown',linewidth=1)
    ax[2][1].plot(line3[5][endl:], label=model[1], color='r',linewidth=1)
    ax[2][1].plot(line4[5][endl:], label=model[2],color='g',linestyle='dotted', marker='o',markersize=2, linewidth=1)
    ax[2][1].set_xlabel('time (days)', fontsize=12)
    #ax[2][1].set_title(tit)
    ax[2][1].legend(loc='best', frameon=False,fontsize=9)
    plt.tight_layout()
    if (scenario==1):
        savefig('../results/Fig 6.png')
    else:
       savefig('../results/Fig 8.png')
    
def combine_plots_DNN(line1, line5, line6,line7,model,endl,tit): # plot prediction graph for deep learning models
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
    ax[0][0].plot(line1[0][endl:],'k+', markersize=4,label='BTC')
    ax[0][0].plot(line5[0][endl:], label=model[3],color='b', linewidth=1)
    ax[0][0].plot(line6[0][endl:], label=model[4],color='k', linewidth=1)
    ax[0][0].plot(line7[0][endl:], label=model[5],color='c', linewidth=1)
    ax[0][0].set_ylabel('price ($)', fontsize=12)
    ax[0][0].set_title(tit)
    ax[0][0].legend(loc='best',frameon=False, fontsize=9)

    ax[0][1].plot(line1[1][endl:],'k+', markersize=4, label='ETH')
    ax[0][1].plot(line5[1][endl:], label=model[3],color='b', linewidth=1)
    ax[0][1].plot(line6[1][endl:], label=model[4],color='k', linewidth=1)
    ax[0][1].plot(line7[1][endl:], label=model[5],color='c', linewidth=1)
    ax[0][1].legend(loc='best',frameon=False, fontsize=9)
    ax[0][1].set_title(tit)

    ax[1][0].plot(line1[2][endl:],'k+', markersize=4, label='BNB')
    ax[1][0].plot(line5[2][endl:], label=model[3],color='b', linewidth=1)
    ax[1][0].plot(line6[2][endl:], label=model[4],color='k', linewidth=1)
    ax[1][0].plot(line7[2][endl:], label=model[5],color='c', linewidth=1)
    ax[1][0].set_ylabel('price ($)', fontsize=12)
    ax[1][0].legend(loc='best',frameon=False, fontsize=9)
    
    ax[1][1].plot(line1[3][endl:],'k+', markersize=4, label='LTC')
    ax[1][1].plot(line5[3][endl:], label=model[3],color='b', linewidth=1)
    ax[1][1].plot(line6[3][endl:], label=model[4],color='k', linewidth=1)
    ax[1][1].plot(line7[3][endl:], label=model[5],color='c', linewidth=1)
    ax[1][1].legend(loc='best',frameon=False, fontsize=9)
    
    ax[2][0].plot(line1[4][endl:],'k+', markersize=4, label='XLM')
    ax[2][0].plot(line5[4][endl:], label=model[3],color='b', linewidth=1)
    ax[2][0].plot(line6[4][endl:], label=model[4],color='k', linewidth=1)
    ax[2][0].plot(line7[4][endl:], label=model[5],color='c', linewidth=1)
    ax[2][0].set_ylabel('price ($)', fontsize=12)
    ax[2][0].set_xlabel('time (days)', fontsize=12)
    ax[2][0].legend(loc='best',frameon=False, fontsize=9)
    
    ax[2][1].plot(line1[5][endl:], 'k+', markersize=4,label='DOGE')
    ax[2][1].plot(line5[5][endl:], label=model[3],color='b', linewidth=1)
    ax[2][1].plot(line6[5][endl:], label=model[4],color='k',linewidth=1)
    ax[2][1].plot(line7[5][endl:], label=model[5],color='c', linewidth=1)
    ax[2][1].set_xlabel('time (days)', fontsize=12)
    ax[2][1].legend(loc='best', frameon=False,fontsize=9)
    plt.tight_layout()
    if(scenario==1):
        savefig('../results/Fig 5.png')
    else:
        savefig('../results/Fig 7.png')

def plot_residuals_tree(a1,a2,g1,g2,x1,x2,c1,c2,m1,m2,r1,r2,tit): # plot residuals for the tree-based models
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(10, 12))
    xtit='Predicted price ($)'
    axes[0][0].plot(a1[0],a2[0],'b.')
    axes[0][0].hlines(y=0, xmin=np.min(a1[0]), xmax=np.max(a1[0]), linewidth=2, color='r')
    axes[0][0].set_ylabel('BTC Residuals')  
    axes[0][0].set_title('ADA -'+tit)
    axes[0][1].plot(a1[1],a2[1],'b.')
    axes[0][1].hlines(y=0, xmin=np.min(a1[1]), xmax=np.max(a1[1]), linewidth=2, color='r') 
    axes[0][1].set_title('GBM -'+tit)
    axes[0][2].plot(a1[2],a2[2],'b.')
    axes[0][2].hlines(y=0, xmin=np.min(a1[2]), xmax=np.max(a1[2]), linewidth=2, color='r')
    axes[0][2].set_title('XGB -'+tit)

    axes[1][0].plot(g1[0],g2[0],'b.')
    axes[1][0].hlines(y=0, xmin=np.min(g1[0]), xmax=np.max(g1[0]), linewidth=2, color='r')
    axes[1][0].set_ylabel('ETH Residuals')   
    axes[1][1].plot(g1[1],g2[1],'b.')
    axes[1][1].hlines(y=0, xmin=np.min(g1[1]), xmax=np.max(g1[1]), linewidth=2, color='r')  
    axes[1][2].plot(g1[2],g2[2],'b.')
    axes[1][2].hlines(y=0, xmin=np.min(g1[2]), xmax=np.max(g1[2]), linewidth=2, color='r')
   
    axes[2][0].plot(x1[0],x2[0],'b.')
    axes[2][0].hlines(y=0, xmin=np.min(x1[0]), xmax=np.max(x1[0]), linewidth=2, color='r')
    axes[2][0].set_ylabel('BNB Residuals')    
    axes[2][1].plot(x1[1],x2[1],'b.')
    axes[2][1].hlines(y=0, xmin=np.min(x1[1]), xmax=np.max(x1[1]), linewidth=2, color='r')   
    axes[2][2].plot(x1[2],x2[2],'b.')
    axes[2][2].hlines(y=0, xmin=np.min(x1[2]), xmax=np.max(x1[2]), linewidth=2, color='r')
  
    axes[3][0].plot(c1[0],c2[0],'b.')
    axes[3][0].hlines(y=0, xmin=np.min(c1[0]), xmax=np.max(c1[0]), linewidth=2, color='r')
    axes[3][0].set_ylabel('LTC Residuals')   
    axes[3][1].plot(c1[1],c2[1],'b.')
    axes[3][1].hlines(y=0, xmin=np.min(c1[1]), xmax=np.max(c1[1]), linewidth=2, color='r')   
    axes[3][2].plot(c1[2],c2[2],'b.')
    axes[3][2].hlines(y=0, xmin=np.min(c1[2]), xmax=np.max(c1[2]), linewidth=2, color='r')
    
    axes[4][0].plot(m1[0],m2[0],'b.')
    axes[4][0].hlines(y=0, xmin=np.min(m1[0]), xmax=np.max(m1[0]), linewidth=2, color='r')
    axes[4][0].set_ylabel('XLM Residuals')   
    axes[4][1].plot(m1[1],m2[1],'b.')
    axes[4][1].hlines(y=0, xmin=np.min(m1[1]), xmax=np.max(m1[1]), linewidth=2, color='r')   
    axes[4][2].plot(m1[2],m2[2],'b.')
    axes[4][2].hlines(y=0, xmin=np.min(m1[2]), xmax=np.max(m1[2]), linewidth=2, color='r')
    
    axes[5][0].plot(r1[0],r2[0],'b.')
    axes[5][0].hlines(y=0, xmin=np.min(r1[0]), xmax=np.max(r1[0]), linewidth=2, color='r')
    axes[5][0].set_ylabel('DOGE Residuals')
    axes[5][0].set_xlabel(xtit)    
    axes[5][1].plot(r1[1],r2[1],'b.')
    axes[5][1].hlines(y=0, xmin=np.min(r1[1]), xmax=np.max(r1[1]), linewidth=2, color='r')
    axes[5][1].set_xlabel(xtit)  
    axes[5][2].plot(r1[2],r2[2],'b.')
    axes[5][2].hlines(y=0, xmin=np.min(r1[2]), xmax=np.max(r1[2]), linewidth=2, color='r')
    axes[5][2].set_xlabel(xtit)
    plt.tight_layout()
    if(scenario==1):
        savefig('../results/Fig 4.png')
    #else:
       # savefig('..results/Fig 4XXX.png')# Not shown in the paper

def plot_residuals_DNN(a1,a2,g1,g2,x1,x2,c1,c2,m1,m2,r1,r2,tit): # plot residuals for the  deep learning models
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(10, 12))
    xtit='Predicted price ($)'
    axes[0][0].plot(a1[3],a2[3],'b.')
    axes[0][0].hlines(y=0, xmin=np.min(a1[3]), xmax=np.max(a1[3]), linewidth=2, color='r')
    axes[0][0].set_ylabel('BTC Residuals') 
    axes[0][0].set_title('CNN -'+tit)
    axes[0][1].plot(a1[4],a2[4],'b.')
    axes[0][1].hlines(y=0, xmin=np.min(a1[4]), xmax=np.max(a1[4]), linewidth=2, color='r') 
    axes[0][1].set_title('DFNN -'+tit)
    axes[0][2].plot(a1[5],a2[5],'b.')
    axes[0][2].hlines(y=0, xmin=np.min(a1[5]), xmax=np.max(a1[5]), linewidth=2, color='r')
    axes[0][2].set_title('GRU -'+tit)
  
    axes[1][0].plot(g1[3],g2[3],'b.')
    axes[1][0].hlines(y=0, xmin=np.min(g1[3]), xmax=np.max(g1[3]), linewidth=2, color='r')
    axes[1][0].set_ylabel('ETH Residuals')   
    axes[1][1].plot(g1[4],g2[4],'b.')
    axes[1][1].hlines(y=0, xmin=np.min(g1[4]), xmax=np.max(g1[4]), linewidth=2, color='r')   
    axes[1][2].plot(g1[5],g2[5],'b.')
    axes[1][2].hlines(y=0, xmin=np.min(g1[5]), xmax=np.max(g1[5]), linewidth=2, color='r')    
  
    axes[2][0].plot(x1[3],x2[3],'b.')
    axes[2][0].hlines(y=0, xmin=np.min(x1[3]), xmax=np.max(x1[3]), linewidth=2, color='r')
    axes[2][0].set_ylabel('BNB Residuals')   
    axes[2][1].plot(x1[4],x2[4],'b.')
    axes[2][1].hlines(y=0, xmin=np.min(x1[4]), xmax=np.max(x1[4]), linewidth=2, color='r')  
    axes[2][2].plot(x1[5],x2[5],'b.')
    axes[2][2].hlines(y=0, xmin=np.min(x1[5]), xmax=np.max(x1[5]), linewidth=2, color='r')
    
    axes[3][0].plot(c1[3],c2[3],'b.')
    axes[3][0].hlines(y=0, xmin=np.min(c1[3]), xmax=np.max(c1[3]), linewidth=2, color='r')
    axes[3][0].set_ylabel('LTC Residuals')  
    axes[3][1].plot(c1[4],c2[4],'b.')
    axes[3][1].hlines(y=0, xmin=np.min(c1[4]), xmax=np.max(c1[4]), linewidth=2, color='r')   
    axes[3][2].plot(c1[5],c2[5],'b.')
    axes[3][2].hlines(y=0, xmin=np.min(c1[5]), xmax=np.max(c1[5]), linewidth=2, color='r')
   
    axes[4][0].plot(m1[3],m2[3],'b.')
    axes[4][0].hlines(y=0, xmin=np.min(m1[3]), xmax=np.max(m1[3]), linewidth=2, color='r')
    axes[4][0].set_ylabel('XLM Residuals')   
    axes[4][1].plot(m1[4],m2[4],'b.')
    axes[4][1].hlines(y=0, xmin=np.min(m1[4]), xmax=np.max(m1[4]), linewidth=2, color='r')   
    axes[4][2].plot(m1[5],m2[5],'b.')
    axes[4][2].hlines(y=0, xmin=np.min(m1[5]), xmax=np.max(m1[5]), linewidth=2, color='r')
    
    axes[5][0].plot(r1[3],r2[3],'b.')
    axes[5][0].hlines(y=0, xmin=np.min(r1[3]), xmax=np.max(r1[3]), linewidth=2, color='r')
    axes[5][0].set_ylabel('DOGE Residuals')
    axes[5][0].set_xlabel(xtit)   
    axes[5][1].plot(r1[4],r2[4],'b.')
    axes[5][1].hlines(y=0, xmin=np.min(r1[4]), xmax=np.max(r1[4]), linewidth=2, color='r')
    axes[5][1].set_xlabel(xtit)   
    axes[5][2].plot(r1[5],r2[5],'b.')
    axes[5][2].hlines(y=0, xmin=np.min(r1[5]), xmax=np.max(r1[5]), linewidth=2, color='r')
    axes[5][2].set_xlabel(xtit)
    plt.tight_layout()
    if(scenario==1):
        savefig('../results/Fig 3.png')
    #else:
        #savefig('../results/Fig 3XXX.png')# Not shown in the paper
    #plt.show()
#-------------------------=======================--------------------------------------
# End of Graphs plotting functions
#-------------------------=======================-------------------------------------- 

#--------------------------------------------------------------------------------------
# module prepare_results                                                              #
#--------------------------------------------------------------------------------------
def prepare_results(yahoo_BTC_XGB,yahoo_LTC_XGB,yahoo_ETH_XGB,yahoo_BNB_XGB,yahoo_DOGE_XGB,yahoo_XLM_XGB,
                yahoo_BTC_GBM,yahoo_LTC_GBM,yahoo_ETH_GBM,yahoo_BNB_GBM,yahoo_DOGE_GBM,yahoo_XLM_GBM,
                yahoo_BTC_ADA,yahoo_LTC_ADA,yahoo_ETH_ADA,yahoo_BNB_ADA,yahoo_DOGE_ADA,yahoo_XLM_ADA,
                yahoo_BTC_GRU,yahoo_LTC_GRU,yahoo_ETH_GRU,yahoo_BNB_GRU,yahoo_DOGE_GRU,yahoo_XLM_GRU,
                yahoo_BTC_MLP,yahoo_LTC_MLP,yahoo_ETH_MLP,yahoo_BNB_MLP,yahoo_DOGE_MLP,yahoo_XLM_MLP,
                yahoo_BTC_CNN,yahoo_LTC_CNN,yahoo_ETH_CNN,yahoo_BNB_CNN,yahoo_DOGE_CNN,yahoo_XLM_CNN,
                invest_BTC_XGB,invest_LTC_XGB,invest_ETH_XGB,invest_BNB_XGB,invest_DOGE_XGB,invest_XLM_XGB,
                invest_BTC_GBM,invest_LTC_GBM,invest_ETH_GBM,invest_BNB_GBM,invest_DOGE_GBM,invest_XLM_GBM,
                invest_BTC_ADA,invest_LTC_ADA,invest_ETH_ADA,invest_BNB_ADA,invest_DOGE_ADA,invest_XLM_ADA,
                invest_BTC_GRU,invest_LTC_GRU,invest_ETH_GRU,invest_BNB_GRU,invest_DOGE_GRU,invest_XLM_GRU,
                invest_BTC_MLP,invest_LTC_MLP,invest_ETH_MLP,invest_BNB_MLP,invest_DOGE_MLP,invest_XLM_MLP,
                invest_BTC_CNN,invest_LTC_CNN,invest_ETH_CNN,invest_BNB_CNN,invest_DOGE_CNN,invest_XLM_CNN,
                bitfinex_BTC_XGB,bitfinex_LTC_XGB,bitfinex_ETH_XGB,bitfinex_BNB_XGB,bitfinex_DOGE_XGB,bitfinex_XLM_XGB,
                bitfinex_BTC_GBM,bitfinex_LTC_GBM,bitfinex_ETH_GBM,bitfinex_BNB_GBM,bitfinex_DOGE_GBM,bitfinex_XLM_GBM,
                bitfinex_BTC_ADA,bitfinex_LTC_ADA,bitfinex_ETH_ADA,bitfinex_BNB_ADA,bitfinex_DOGE_ADA,bitfinex_XLM_ADA,
                bitfinex_BTC_GRU,bitfinex_LTC_GRU,bitfinex_ETH_GRU,bitfinex_BNB_GRU,bitfinex_DOGE_GRU,bitfinex_XLM_GRU,
                bitfinex_BTC_MLP,bitfinex_LTC_MLP,bitfinex_ETH_MLP,bitfinex_BNB_MLP,bitfinex_DOGE_MLP,bitfinex_XLM_MLP,
                bitfinex_BTC_CNN,bitfinex_LTC_CNN,bitfinex_ETH_CNN,bitfinex_BNB_CNN,bitfinex_DOGE_CNN,bitfinex_XLM_CNN):
    
    
    #compute explained variance score (evs), mean absolute percentage error (MAPE), t-test, and NSE for all models 
    #in the three datasets (Yahoo, UK investing and Bitfinex), store results as Table 4A and Table 4B
    # compute comparative performance of models on BTC
    btc=[['xgb','evs',round(explained_variance_score(yahoo_BTC_XGB.target, yahoo_BTC_XGB.forcast),2),round(explained_variance_score(invest_BTC_XGB.target, invest_BTC_XGB.forcast),2),round(explained_variance_score(bitfinex_BTC_XGB.target, bitfinex_BTC_XGB.forcast),2)],['xgb','mape',round(mean_absolute_percentage_error(yahoo_BTC_XGB.target, yahoo_BTC_XGB.forcast),2),round(mean_absolute_percentage_error(invest_BTC_XGB.target, invest_BTC_XGB.forcast),2),round(mean_absolute_percentage_error(bitfinex_BTC_XGB.target, bitfinex_BTC_XGB.forcast),2)],['xgb','t-test',round(mbv(yahoo_BTC_XGB.target, yahoo_BTC_XGB.forcast),2),round(mbv(invest_BTC_XGB.target, invest_BTC_XGB.forcast),2),round(mbv(bitfinex_BTC_XGB.target, bitfinex_BTC_XGB.forcast),2)],['xgb','nse',round(nse(yahoo_BTC_XGB.target, yahoo_BTC_XGB.forcast),2),round(nse(invest_BTC_XGB.target, invest_BTC_XGB.forcast),2),round(nse(bitfinex_BTC_XGB.target, bitfinex_BTC_XGB.forcast),2)],
          ['gbm','evs',round(explained_variance_score(yahoo_BTC_GBM.target, yahoo_BTC_GBM.forcast),2),round(explained_variance_score(invest_BTC_GBM.target, invest_BTC_GBM.forcast),2),round(explained_variance_score(bitfinex_BTC_GBM.target, bitfinex_BTC_GBM.forcast),2)],['gbm','mape',round(mean_absolute_percentage_error(yahoo_BTC_GBM.target, yahoo_BTC_GBM.forcast),2),round(mean_absolute_percentage_error(invest_BTC_GBM.target, invest_BTC_GBM.forcast),2),round(mean_absolute_percentage_error(bitfinex_BTC_GBM.target, bitfinex_BTC_GBM.forcast),2)],['gbm','t-test',round(mbv(yahoo_BTC_GBM.target, yahoo_BTC_GBM.forcast),2),round(mbv(invest_BTC_GBM.target, invest_BTC_GBM.forcast),2),round(mbv(bitfinex_BTC_GBM.target, bitfinex_BTC_GBM.forcast),2)],['gbm','nse',round(nse(yahoo_BTC_GBM.target, yahoo_BTC_GBM.forcast),2),round(nse(invest_BTC_GBM.target, invest_BTC_GBM.forcast),2),round(nse(bitfinex_BTC_GBM.target, bitfinex_BTC_GBM.forcast),2)],
          ['ada','evs',round(explained_variance_score(yahoo_BTC_ADA.target, yahoo_BTC_ADA.forcast),2),round(explained_variance_score(invest_BTC_ADA.target, invest_BTC_ADA.forcast),2),round(explained_variance_score(bitfinex_BTC_ADA.target, bitfinex_BTC_ADA.forcast),2)],['ada','mape',round(mean_absolute_percentage_error(yahoo_BTC_ADA.target, yahoo_BTC_ADA.forcast),2),round(mean_absolute_percentage_error(invest_BTC_ADA.target, invest_BTC_ADA.forcast),2),round(mean_absolute_percentage_error(bitfinex_BTC_ADA.target, bitfinex_BTC_ADA.forcast),2)],['ada','t-test',round(mbv(yahoo_BTC_ADA.target, yahoo_BTC_ADA.forcast),2), round(mbv(invest_BTC_ADA.target, invest_BTC_ADA.forcast),2),round(mbv(bitfinex_BTC_ADA.target, bitfinex_BTC_ADA.forcast),2)],['ada','nse',round(nse(yahoo_BTC_ADA.target, yahoo_BTC_ADA.forcast),2), round(nse(invest_BTC_ADA.target, invest_BTC_ADA.forcast),2),round(nse(bitfinex_BTC_ADA.target, bitfinex_BTC_ADA.forcast),2)],
          ['gru','evs',round(explained_variance_score(yahoo_BTC_GRU.target, yahoo_BTC_GRU.forcast),2),round(explained_variance_score(invest_BTC_GRU.target, invest_BTC_GRU.forcast),2),round(explained_variance_score(bitfinex_BTC_GRU.target, bitfinex_BTC_GRU.forcast),2)],['gru','mape',round(mean_absolute_percentage_error(yahoo_BTC_GRU.target, yahoo_BTC_GRU.forcast),2),round(mean_absolute_percentage_error(invest_BTC_GRU.target, invest_BTC_GRU.forcast),2),round(mean_absolute_percentage_error(bitfinex_BTC_GRU.target, bitfinex_BTC_GRU.forcast),2)],['gru','t-test',round(mbv(yahoo_BTC_GRU.target, yahoo_BTC_GRU.forcast),2),round(mbv(invest_BTC_GRU.target, invest_BTC_GRU.forcast),2),round(mbv(bitfinex_BTC_GRU.target, bitfinex_BTC_GRU.forcast),2)],['gru','nse',round(nse(yahoo_BTC_GRU.target, yahoo_BTC_GRU.forcast),2),round(nse(invest_BTC_GRU.target, invest_BTC_GRU.forcast),2),round(nse(bitfinex_BTC_GRU.target, bitfinex_BTC_GRU.forcast),2)],
          ['mlp','evs',round(explained_variance_score(yahoo_BTC_MLP.target, yahoo_BTC_MLP.forcast),2),round(explained_variance_score(invest_BTC_MLP.target, invest_BTC_MLP.forcast),2),round(explained_variance_score(bitfinex_BTC_MLP.target, bitfinex_BTC_MLP.forcast),2)],['mlp','mape',round(mean_absolute_percentage_error(yahoo_BTC_MLP.target, yahoo_BTC_MLP.forcast),2),round(mean_absolute_percentage_error(invest_BTC_MLP.target, invest_BTC_MLP.forcast),2),round(mean_absolute_percentage_error(bitfinex_BTC_MLP.target, bitfinex_BTC_MLP.forcast),2)],['mlp','t-test',round(mbv(yahoo_BTC_MLP.target, yahoo_BTC_MLP.forcast),2),round(mbv(invest_BTC_MLP.target, invest_BTC_MLP.forcast),2),round(mbv(bitfinex_BTC_MLP.target, bitfinex_BTC_MLP.forcast),2)],['mlp','nse',round(nse(yahoo_BTC_MLP.target, yahoo_BTC_MLP.forcast),2), round(nse(invest_BTC_MLP.target, invest_BTC_MLP.forcast),2),round(nse(bitfinex_BTC_MLP.target, bitfinex_BTC_MLP.forcast),2)],
          ['cnn','evs',round(explained_variance_score(yahoo_BTC_CNN.target, yahoo_BTC_CNN.forcast),2),round(explained_variance_score(invest_BTC_CNN.target, invest_BTC_CNN.forcast),2),round(explained_variance_score(bitfinex_BTC_CNN.target, bitfinex_BTC_CNN.forcast),2)],['cnn','mape',round(mean_absolute_percentage_error(yahoo_BTC_CNN.target, yahoo_BTC_CNN.forcast),2),round(mean_absolute_percentage_error(invest_BTC_CNN.target, invest_BTC_CNN.forcast),2),round(mean_absolute_percentage_error(bitfinex_BTC_CNN.target, bitfinex_BTC_CNN.forcast),2)],['cnn','t-test',round(mbv(yahoo_BTC_CNN.target, yahoo_BTC_CNN.forcast),2),round(mbv(invest_BTC_CNN.target, invest_BTC_CNN.forcast),2),round(mbv(bitfinex_BTC_CNN.target, bitfinex_BTC_CNN.forcast),2)],['cnn','nse',round(nse(yahoo_BTC_CNN.target, yahoo_BTC_CNN.forcast),2),round(nse(invest_BTC_CNN.target, invest_BTC_CNN.forcast),2),round(nse(bitfinex_BTC_CNN.target, bitfinex_BTC_CNN.forcast),2)]]
    # Create the pandas DataFrame
    df_btc = pd.DataFrame(btc, columns=['Model','Metric', 'YAH:BTC','INVS:BTC','BITF:BTC'])
   
     #compute explained variance score (evs), mean absolute percentage error (MAPE), t-test, and NSE for all models 
     #in the three datasets (Yahoo, UK investing and Bitfinex) for ETH
    eth=[[round(explained_variance_score(yahoo_ETH_XGB.target, yahoo_ETH_XGB.forcast),2),round(explained_variance_score(invest_ETH_XGB.target, invest_ETH_XGB.forcast),2),round(explained_variance_score(bitfinex_ETH_XGB.target, bitfinex_ETH_XGB.forcast),2)],[round(mean_absolute_percentage_error(yahoo_ETH_XGB.target, yahoo_ETH_XGB.forcast),2),round(mean_absolute_percentage_error(invest_ETH_XGB.target, invest_ETH_XGB.forcast),2),round(mean_absolute_percentage_error(bitfinex_ETH_XGB.target, bitfinex_ETH_XGB.forcast),2)],[round(mbv(yahoo_ETH_XGB.target, yahoo_ETH_XGB.forcast),2),round(mbv(invest_ETH_XGB.target, invest_ETH_XGB.forcast),2), round(mbv(bitfinex_ETH_XGB.target, bitfinex_ETH_XGB.forcast),2)],[round(nse(yahoo_ETH_XGB.target, yahoo_ETH_XGB.forcast),2),round(nse(invest_ETH_XGB.target, invest_ETH_XGB.forcast),2),round(nse(bitfinex_ETH_XGB.target, bitfinex_ETH_XGB.forcast),2)],
          [round(explained_variance_score(yahoo_ETH_GBM.target, yahoo_ETH_GBM.forcast),2),round(explained_variance_score(invest_ETH_GBM.target, invest_ETH_GBM.forcast),2),round(explained_variance_score(bitfinex_ETH_GBM.target, bitfinex_ETH_GBM.forcast),2)],[round(mean_absolute_percentage_error(yahoo_ETH_GBM.target, yahoo_ETH_GBM.forcast),2),round(mean_absolute_percentage_error(invest_ETH_GBM.target, invest_ETH_GBM.forcast),2),round(mean_absolute_percentage_error(bitfinex_ETH_GBM.target, bitfinex_ETH_GBM.forcast),2)],[round(mbv(yahoo_ETH_GBM.target, yahoo_ETH_GBM.forcast),2),round(mbv(invest_ETH_GBM.target, invest_ETH_GBM.forcast),2),round(mbv(bitfinex_ETH_GBM.target, bitfinex_ETH_GBM.forcast),2)],[round(nse(yahoo_ETH_GBM.target, yahoo_ETH_GBM.forcast),2),round(nse(invest_ETH_GBM.target, invest_ETH_GBM.forcast),2),round(nse(bitfinex_ETH_GBM.target, bitfinex_ETH_GBM.forcast),2)],
          [round(explained_variance_score(yahoo_ETH_ADA.target, yahoo_ETH_ADA.forcast),2),round(explained_variance_score(invest_ETH_ADA.target, invest_ETH_ADA.forcast),2),round(explained_variance_score(bitfinex_ETH_ADA.target, bitfinex_ETH_ADA.forcast),2)],[round(mean_absolute_percentage_error(yahoo_ETH_ADA.target, yahoo_ETH_ADA.forcast),2),round(mean_absolute_percentage_error(invest_ETH_ADA.target, invest_ETH_ADA.forcast),2),round(mean_absolute_percentage_error(bitfinex_ETH_ADA.target, bitfinex_ETH_ADA.forcast),2)],[round(mbv(yahoo_ETH_ADA.target, yahoo_ETH_ADA.forcast),2),round(mbv(invest_ETH_ADA.target, invest_ETH_ADA.forcast),2),round(mbv(bitfinex_ETH_ADA.target, bitfinex_ETH_ADA.forcast),2)],[round(nse(yahoo_ETH_ADA.target, yahoo_ETH_ADA.forcast),2),round(nse(invest_ETH_ADA.target, invest_ETH_ADA.forcast),2),round(nse(bitfinex_ETH_ADA.target, bitfinex_ETH_ADA.forcast),2)],
          [round(explained_variance_score(yahoo_ETH_GRU.target, yahoo_ETH_GRU.forcast),2),round(explained_variance_score(invest_ETH_GRU.target, invest_ETH_GRU.forcast),2),round(explained_variance_score(bitfinex_ETH_GRU.target, bitfinex_ETH_GRU.forcast),2)],[round(mean_absolute_percentage_error(yahoo_ETH_GRU.target, yahoo_ETH_GRU.forcast),2),round(mean_absolute_percentage_error(invest_ETH_GRU.target, invest_ETH_GRU.forcast),2),round(mean_absolute_percentage_error(bitfinex_ETH_GRU.target, bitfinex_ETH_GRU.forcast),2)],[round(mbv(yahoo_ETH_GRU.target, yahoo_ETH_GRU.forcast),2),round(mbv(invest_ETH_GRU.target, invest_ETH_GRU.forcast),2),round(mbv(bitfinex_ETH_GRU.target, bitfinex_ETH_GRU.forcast),2)],[round(nse(yahoo_ETH_GRU.target, yahoo_ETH_GRU.forcast),2),round(nse(invest_ETH_GRU.target, invest_ETH_GRU.forcast),2),round(nse(bitfinex_ETH_GRU.target, bitfinex_ETH_GRU.forcast),2)],
          [round(explained_variance_score(yahoo_ETH_MLP.target, yahoo_ETH_MLP.forcast),2),round(explained_variance_score(invest_ETH_MLP.target, invest_ETH_MLP.forcast),2),round(explained_variance_score(bitfinex_ETH_MLP.target, bitfinex_ETH_MLP.forcast),2)],[round(mean_absolute_percentage_error(yahoo_ETH_MLP.target, yahoo_ETH_MLP.forcast),2),round(mean_absolute_percentage_error(invest_ETH_MLP.target, invest_ETH_MLP.forcast),2),round(mean_absolute_percentage_error(bitfinex_ETH_MLP.target, bitfinex_ETH_MLP.forcast),2)],[round(mbv(yahoo_ETH_MLP.target, yahoo_ETH_MLP.forcast),2),round(mbv(invest_ETH_MLP.target, invest_ETH_MLP.forcast),2),round(mbv(bitfinex_ETH_MLP.target, bitfinex_ETH_MLP.forcast),2)],[round(nse(yahoo_ETH_MLP.target, yahoo_ETH_MLP.forcast),2),round(nse(invest_ETH_MLP.target, invest_ETH_MLP.forcast),2),round(nse(bitfinex_ETH_MLP.target, bitfinex_ETH_MLP.forcast),2)],
          [round(explained_variance_score(yahoo_ETH_CNN.target, yahoo_ETH_CNN.forcast),2),round(explained_variance_score(invest_ETH_CNN.target, invest_ETH_CNN.forcast),2),round(explained_variance_score(bitfinex_ETH_CNN.target, bitfinex_ETH_CNN.forcast),2)],[round(mean_absolute_percentage_error(yahoo_ETH_CNN.target, yahoo_ETH_CNN.forcast),2),round(mean_absolute_percentage_error(invest_ETH_CNN.target, invest_ETH_CNN.forcast),2),round(mean_absolute_percentage_error(bitfinex_ETH_CNN.target, bitfinex_ETH_CNN.forcast),2)],[round(mbv(yahoo_ETH_CNN.target, yahoo_ETH_CNN.forcast),2),round(mbv(invest_ETH_CNN.target, invest_ETH_CNN.forcast),2),round(mbv(bitfinex_ETH_CNN.target, bitfinex_ETH_CNN.forcast),2)],[round(nse(yahoo_ETH_CNN.target, yahoo_ETH_CNN.forcast),2),round(nse(invest_ETH_CNN.target, invest_ETH_CNN.forcast),2),round(nse(bitfinex_ETH_CNN.target, bitfinex_ETH_CNN.forcast),2)]]
    df_eth = pd.DataFrame(eth, columns=['YAH:ETH','INVS:ETH','BITF:ETH'])

      #compute explained variance score (evs), mean absolute percentage error (MAPE), t-test, and NSE for all models 
      #in the three datasets (Yahoo, UK investing and Bitfinex) for BNB
    bnb=[[round(explained_variance_score(yahoo_BNB_XGB.target, yahoo_BNB_XGB.forcast),2),round(explained_variance_score(invest_BNB_XGB.target, invest_BNB_XGB.forcast),2),round(explained_variance_score(bitfinex_BNB_XGB.target, bitfinex_BNB_XGB.forcast),2)],[round(mean_absolute_percentage_error(yahoo_BNB_XGB.target, yahoo_BNB_XGB.forcast),2),round(mean_absolute_percentage_error(invest_BNB_XGB.target, invest_BNB_XGB.forcast),2),round(mean_absolute_percentage_error(bitfinex_BNB_XGB.target, bitfinex_BNB_XGB.forcast),2)],[round(mbv(yahoo_BNB_XGB.target, yahoo_BNB_XGB.forcast),2),round(mbv(invest_BNB_XGB.target, invest_BNB_XGB.forcast),2),round(mbv(bitfinex_BNB_XGB.target, bitfinex_BNB_XGB.forcast),2)],[round(nse(yahoo_BNB_XGB.target, yahoo_BNB_XGB.forcast),2),round(nse(invest_BNB_XGB.target, invest_BNB_XGB.forcast),2),round(nse(bitfinex_BNB_XGB.target, bitfinex_BNB_XGB.forcast),2)],
          [round(explained_variance_score(yahoo_BNB_GBM.target, yahoo_BNB_GBM.forcast),2),round(explained_variance_score(invest_BNB_GBM.target, invest_BNB_GBM.forcast),2),round(explained_variance_score(bitfinex_BNB_GBM.target, bitfinex_BNB_GBM.forcast),2)],[round(mean_absolute_percentage_error(yahoo_BNB_GBM.target, yahoo_BNB_GBM.forcast),2),round(mean_absolute_percentage_error(invest_BNB_GBM.target, invest_BNB_GBM.forcast),2),round(mean_absolute_percentage_error(bitfinex_BNB_GBM.target, bitfinex_BNB_GBM.forcast),2)],[round(mbv(yahoo_BNB_GBM.target, yahoo_BNB_GBM.forcast),2),round(mbv(invest_BNB_GBM.target, invest_BNB_GBM.forcast),2),round(mbv(bitfinex_BNB_GBM.target, bitfinex_BNB_GBM.forcast),2)],[round(nse(yahoo_BNB_GBM.target, yahoo_BNB_GBM.forcast),2),round(nse(invest_BNB_GBM.target, invest_BNB_GBM.forcast),2),round(nse(bitfinex_BNB_GBM.target, bitfinex_BNB_GBM.forcast),2)],
          [round(explained_variance_score(yahoo_BNB_ADA.target, yahoo_BNB_ADA.forcast),2),round(explained_variance_score(invest_BNB_ADA.target, invest_BNB_ADA.forcast),2),round(explained_variance_score(bitfinex_BNB_ADA.target, bitfinex_BNB_ADA.forcast),2)],[round(mean_absolute_percentage_error(yahoo_BNB_ADA.target, yahoo_BNB_ADA.forcast),2),round(mean_absolute_percentage_error(invest_BNB_ADA.target, invest_BNB_ADA.forcast),2),round(mean_absolute_percentage_error(bitfinex_BNB_ADA.target, bitfinex_BNB_ADA.forcast),2)],[round(mbv(yahoo_BNB_ADA.target, yahoo_BNB_ADA.forcast),2),round(mbv(invest_BNB_ADA.target, invest_BNB_ADA.forcast),2),round(mbv(bitfinex_BNB_ADA.target, bitfinex_BNB_ADA.forcast),2)],[round(nse(yahoo_BNB_ADA.target, yahoo_BNB_ADA.forcast),2),round(nse(invest_BNB_ADA.target, invest_BNB_ADA.forcast),2),round(nse(bitfinex_BNB_ADA.target, bitfinex_BNB_ADA.forcast),2)],
          [round(explained_variance_score(yahoo_BNB_GRU.target, yahoo_BNB_GRU.forcast),2),round(explained_variance_score(invest_BNB_GRU.target, invest_BNB_GRU.forcast),2),round(explained_variance_score(bitfinex_BNB_GRU.target, bitfinex_BNB_GRU.forcast),2)],[round(mean_absolute_percentage_error(yahoo_BNB_GRU.target, yahoo_BNB_GRU.forcast),2),round(mean_absolute_percentage_error(invest_BNB_GRU.target, invest_BNB_GRU.forcast),2),round(mean_absolute_percentage_error(bitfinex_BNB_GRU.target, bitfinex_BNB_GRU.forcast),2)],[round(mbv(yahoo_BNB_GRU.target, yahoo_BNB_GRU.forcast),2),round(mbv(invest_BNB_GRU.target, invest_BNB_GRU.forcast),2),round(mbv(bitfinex_BNB_GRU.target, bitfinex_BNB_GRU.forcast),2)],[round(nse(yahoo_BNB_GRU.target, yahoo_BNB_GRU.forcast),2),round(nse(invest_BNB_GRU.target, invest_BNB_GRU.forcast),2),round(nse(bitfinex_BNB_GRU.target, bitfinex_BNB_GRU.forcast),2)],
          [round(explained_variance_score(yahoo_BNB_MLP.target, yahoo_BNB_MLP.forcast),2),round(explained_variance_score(invest_BNB_MLP.target, invest_BNB_MLP.forcast),2),round(explained_variance_score(bitfinex_BNB_MLP.target, bitfinex_BNB_MLP.forcast),2)],[round(mean_absolute_percentage_error(yahoo_BNB_MLP.target, yahoo_BNB_MLP.forcast),2),round(mean_absolute_percentage_error(invest_BNB_MLP.target, invest_BNB_MLP.forcast),2),round(mean_absolute_percentage_error(bitfinex_BNB_MLP.target, bitfinex_BNB_MLP.forcast),2)],[round(mbv(yahoo_BNB_MLP.target, yahoo_BNB_MLP.forcast),2),round(mbv(invest_BNB_MLP.target, invest_BNB_MLP.forcast),2),round(mbv(bitfinex_BNB_MLP.target, bitfinex_BNB_MLP.forcast),2)],[round(nse(yahoo_BNB_MLP.target, yahoo_BNB_MLP.forcast),2),round(nse(invest_BNB_MLP.target, invest_BNB_MLP.forcast),2),round(nse(bitfinex_BNB_MLP.target, bitfinex_BNB_MLP.forcast),2)],
          [round(explained_variance_score(yahoo_BNB_CNN.target, yahoo_BNB_CNN.forcast),2),round(explained_variance_score(invest_BNB_CNN.target, invest_BNB_CNN.forcast),2),round(explained_variance_score(bitfinex_BNB_CNN.target, bitfinex_BNB_CNN.forcast),2)],[round(mean_absolute_percentage_error(yahoo_BNB_CNN.target, yahoo_BNB_CNN.forcast),2),round(mean_absolute_percentage_error(invest_BNB_CNN.target, invest_BNB_CNN.forcast),2),round(mean_absolute_percentage_error(bitfinex_BNB_CNN.target, bitfinex_BNB_CNN.forcast),2)],[round(mbv(yahoo_BNB_CNN.target, yahoo_BNB_CNN.forcast),2),round(mbv(invest_BNB_CNN.target, invest_BNB_CNN.forcast),2),round(mbv(bitfinex_BNB_CNN.target, bitfinex_BNB_CNN.forcast),2)],[round(nse(yahoo_BNB_CNN.target, yahoo_BNB_CNN.forcast),2),round(nse(invest_BNB_CNN.target, invest_BNB_CNN.forcast),2),round(nse(bitfinex_BNB_CNN.target, bitfinex_BNB_CNN.forcast),2)]]
    df_bnb = pd.DataFrame(bnb, columns=['YAH:BNB','INVS:BNB','BITF:BNB'])

    #compute explained variance score (evs), mean absolute percentage error (MAPE), t-test, and NSE for all models 
    #in the three datasets (Yahoo, UK investing and Bitfinex) for LTC
    ltc=[[round(explained_variance_score(yahoo_LTC_XGB.target, yahoo_LTC_XGB.forcast),2),round(explained_variance_score(invest_LTC_XGB.target, invest_LTC_XGB.forcast),2),round(explained_variance_score(bitfinex_LTC_XGB.target, bitfinex_LTC_XGB.forcast),2)],[round(mean_absolute_percentage_error(yahoo_LTC_XGB.target, yahoo_LTC_XGB.forcast),2),round(mean_absolute_percentage_error(invest_LTC_XGB.target, invest_LTC_XGB.forcast),2),round(mean_absolute_percentage_error(bitfinex_LTC_XGB.target, bitfinex_LTC_XGB.forcast),2)],[round(mbv(yahoo_LTC_XGB.target, yahoo_LTC_XGB.forcast),2),round(mbv(invest_LTC_XGB.target, invest_LTC_XGB.forcast),2),round(mbv(bitfinex_LTC_XGB.target, bitfinex_LTC_XGB.forcast),2)],[round(nse(yahoo_LTC_XGB.target, yahoo_LTC_XGB.forcast),2),round(nse(invest_LTC_XGB.target, invest_LTC_XGB.forcast),2),round(nse(bitfinex_LTC_XGB.target, bitfinex_LTC_XGB.forcast),2)],
        [round(explained_variance_score(yahoo_LTC_GBM.target, yahoo_LTC_GBM.forcast),2),round(explained_variance_score(invest_LTC_GBM.target, invest_LTC_GBM.forcast),2),round(explained_variance_score(bitfinex_LTC_GBM.target, bitfinex_LTC_GBM.forcast),2)],[round(mean_absolute_percentage_error(yahoo_LTC_GBM.target, yahoo_LTC_GBM.forcast),2),round(mean_absolute_percentage_error(invest_LTC_GBM.target, invest_LTC_GBM.forcast),2),round(mean_absolute_percentage_error(bitfinex_LTC_GBM.target, bitfinex_LTC_GBM.forcast),2)],[round(mbv(yahoo_LTC_GBM.target, yahoo_LTC_GBM.forcast),2),round(mbv(invest_LTC_GBM.target, invest_LTC_GBM.forcast),2),round(mbv(bitfinex_LTC_GBM.target, bitfinex_LTC_GBM.forcast),2)],[round(nse(yahoo_LTC_GBM.target, yahoo_LTC_GBM.forcast),2),round(nse(invest_LTC_GBM.target, invest_LTC_GBM.forcast),2),round(nse(bitfinex_LTC_GBM.target, bitfinex_LTC_GBM.forcast),2)],
        [round(explained_variance_score(yahoo_LTC_ADA.target, yahoo_LTC_ADA.forcast),2),round(explained_variance_score(invest_LTC_ADA.target, invest_LTC_ADA.forcast),2),round(explained_variance_score(bitfinex_LTC_ADA.target, bitfinex_LTC_ADA.forcast),2)],[round(mean_absolute_percentage_error(yahoo_LTC_ADA.target, yahoo_LTC_ADA.forcast),2),round(mean_absolute_percentage_error(invest_LTC_ADA.target, invest_LTC_ADA.forcast),2),round(mean_absolute_percentage_error(bitfinex_LTC_ADA.target, bitfinex_LTC_ADA.forcast),2)],[round(mbv(yahoo_LTC_ADA.target, yahoo_LTC_ADA.forcast),2),round(mbv(invest_LTC_ADA.target, invest_LTC_ADA.forcast),2),round(mbv(bitfinex_LTC_ADA.target, bitfinex_LTC_ADA.forcast),2)],[round(nse(yahoo_LTC_ADA.target, yahoo_LTC_ADA.forcast),2),round(nse(invest_LTC_ADA.target, invest_LTC_ADA.forcast),2),round(nse(bitfinex_LTC_ADA.target, bitfinex_LTC_ADA.forcast),2)],
        [round(explained_variance_score(yahoo_LTC_GRU.target, yahoo_LTC_GRU.forcast),2),round(explained_variance_score(invest_LTC_GRU.target, invest_LTC_GRU.forcast),2),round(explained_variance_score(bitfinex_LTC_GRU.target, bitfinex_LTC_GRU.forcast),2)],[round(mean_absolute_percentage_error(yahoo_LTC_GRU.target, yahoo_LTC_GRU.forcast),2),round(mean_absolute_percentage_error(invest_LTC_GRU.target, invest_LTC_GRU.forcast),2),round(mean_absolute_percentage_error(bitfinex_LTC_GRU.target, bitfinex_LTC_GRU.forcast),2)],[round(mbv(yahoo_LTC_GRU.target, yahoo_LTC_GRU.forcast),2),round(mbv(invest_LTC_GRU.target, invest_LTC_GRU.forcast),2),round(mbv(bitfinex_LTC_GRU.target, bitfinex_LTC_GRU.forcast),2)],[round(nse(yahoo_LTC_GRU.target, yahoo_LTC_GRU.forcast),2),round(nse(invest_LTC_GRU.target, invest_LTC_GRU.forcast),2),round(nse(bitfinex_LTC_GRU.target, bitfinex_LTC_GRU.forcast),2)],
        [round(explained_variance_score(yahoo_LTC_MLP.target, yahoo_LTC_MLP.forcast),2),round(explained_variance_score(invest_LTC_MLP.target, invest_LTC_MLP.forcast),2),round(explained_variance_score(bitfinex_LTC_MLP.target, bitfinex_LTC_MLP.forcast),2)],[round(mean_absolute_percentage_error(yahoo_LTC_MLP.target, yahoo_LTC_MLP.forcast),2),round(mean_absolute_percentage_error(invest_LTC_MLP.target, invest_LTC_MLP.forcast),2),round(mean_absolute_percentage_error(bitfinex_LTC_MLP.target, bitfinex_LTC_MLP.forcast),2)],[round(mbv(yahoo_LTC_MLP.target, yahoo_LTC_MLP.forcast),2),round(mbv(invest_LTC_MLP.target, invest_LTC_MLP.forcast),2),round(mbv(bitfinex_LTC_MLP.target, bitfinex_LTC_MLP.forcast),2)],[round(nse(yahoo_LTC_MLP.target, yahoo_LTC_MLP.forcast),2),round(nse(invest_LTC_MLP.target, invest_LTC_MLP.forcast),2),round(nse(bitfinex_LTC_MLP.target, bitfinex_LTC_MLP.forcast),2)],
        [round(explained_variance_score(yahoo_LTC_CNN.target, yahoo_LTC_CNN.forcast),2),round(explained_variance_score(invest_LTC_CNN.target, invest_LTC_CNN.forcast),2),round(explained_variance_score(bitfinex_LTC_CNN.target, bitfinex_LTC_CNN.forcast),2)],[round(mean_absolute_percentage_error(yahoo_LTC_CNN.target, yahoo_LTC_CNN.forcast),2),round(mean_absolute_percentage_error(invest_LTC_CNN.target, invest_LTC_CNN.forcast),2),round(mean_absolute_percentage_error(bitfinex_LTC_CNN.target, bitfinex_LTC_CNN.forcast),2)],[round(mbv(yahoo_LTC_CNN.target, yahoo_LTC_CNN.forcast),2),round(mbv(invest_LTC_CNN.target, invest_LTC_CNN.forcast),2),round(mbv(bitfinex_LTC_CNN.target, bitfinex_LTC_CNN.forcast),2)],[round(nse(yahoo_LTC_CNN.target, yahoo_LTC_CNN.forcast),2),round(nse(invest_LTC_CNN.target, invest_LTC_CNN.forcast),2),round(nse(bitfinex_LTC_CNN.target, bitfinex_LTC_CNN.forcast),2)]]
    df_ltc = pd.DataFrame(ltc, columns=['YAH:LTC','INVS:LTC','BITF:LTC'])

      #compute explained variance score (evs), mean absolute percentage error (MAPE), t-test, and NSE for all models 
      #in the three datasets (Yahoo, UK investing and Bitfinex) for XLM
    xlm=[[round(explained_variance_score(yahoo_XLM_XGB.target, yahoo_XLM_XGB.forcast),2),round(explained_variance_score(invest_XLM_XGB.target, invest_XLM_XGB.forcast),2),round(explained_variance_score(bitfinex_XLM_XGB.target, bitfinex_XLM_XGB.forcast),2)],[round(mean_absolute_percentage_error(yahoo_XLM_XGB.target, yahoo_XLM_XGB.forcast),2),round(mean_absolute_percentage_error(invest_XLM_XGB.target, invest_XLM_XGB.forcast),2),round(mean_absolute_percentage_error(bitfinex_XLM_XGB.target, bitfinex_XLM_XGB.forcast),2)],[round(mbv(yahoo_XLM_XGB.target, yahoo_XLM_XGB.forcast),2),round(mbv(invest_XLM_XGB.target, invest_XLM_XGB.forcast),2),round(mbv(bitfinex_XLM_XGB.target, bitfinex_XLM_XGB.forcast),2)],[round(nse(yahoo_XLM_XGB.target, yahoo_XLM_XGB.forcast),2),round(nse(invest_XLM_XGB.target, invest_XLM_XGB.forcast),2),round(nse(bitfinex_XLM_XGB.target, bitfinex_XLM_XGB.forcast),2)],
          [round(explained_variance_score(yahoo_XLM_GBM.target, yahoo_XLM_GBM.forcast),2),round(explained_variance_score(invest_XLM_GBM.target, invest_XLM_GBM.forcast),2),round(explained_variance_score(bitfinex_XLM_GBM.target, bitfinex_XLM_GBM.forcast),2)],[round(mean_absolute_percentage_error(yahoo_XLM_GBM.target, yahoo_XLM_GBM.forcast),2),round(mean_absolute_percentage_error(invest_XLM_GBM.target, invest_XLM_GBM.forcast),2),round(mean_absolute_percentage_error(bitfinex_XLM_GBM.target, bitfinex_XLM_GBM.forcast),2)],[round(mbv(yahoo_XLM_GBM.target, yahoo_XLM_GBM.forcast),2),round(mbv(invest_XLM_GBM.target, invest_XLM_GBM.forcast),2),round(mbv(bitfinex_XLM_GBM.target, bitfinex_XLM_GBM.forcast),2)],[round(nse(yahoo_XLM_GBM.target, yahoo_XLM_GBM.forcast),2),round(nse(invest_XLM_GBM.target, invest_XLM_GBM.forcast),2),round(nse(bitfinex_XLM_GBM.target, bitfinex_XLM_GBM.forcast),2)],
          [round(explained_variance_score(yahoo_XLM_ADA.target, yahoo_XLM_ADA.forcast),2),round(explained_variance_score(invest_XLM_ADA.target, invest_XLM_ADA.forcast),2),round(explained_variance_score(bitfinex_XLM_ADA.target, bitfinex_XLM_ADA.forcast),2)],[round(mean_absolute_percentage_error(yahoo_XLM_ADA.target, yahoo_XLM_ADA.forcast),2),round(mean_absolute_percentage_error(invest_XLM_ADA.target, invest_XLM_ADA.forcast),2),round(mean_absolute_percentage_error(bitfinex_XLM_ADA.target, bitfinex_XLM_ADA.forcast),2)],[round(mbv(yahoo_XLM_ADA.target, yahoo_XLM_ADA.forcast),2),round(mbv(invest_XLM_ADA.target, invest_XLM_ADA.forcast),2),round(mbv(bitfinex_XLM_ADA.target, bitfinex_XLM_ADA.forcast),2)],[round(nse(yahoo_XLM_ADA.target, yahoo_XLM_ADA.forcast),2),round(nse(invest_XLM_ADA.target, invest_XLM_ADA.forcast),2),round(nse(bitfinex_XLM_ADA.target, bitfinex_XLM_ADA.forcast),2)],
          [round(explained_variance_score(yahoo_XLM_GRU.target, yahoo_XLM_GRU.forcast),2),round(explained_variance_score(invest_XLM_GRU.target, invest_XLM_GRU.forcast),2),round(explained_variance_score(bitfinex_XLM_GRU.target, bitfinex_XLM_GRU.forcast),2)],[round(mean_absolute_percentage_error(yahoo_XLM_GRU.target, yahoo_XLM_GRU.forcast),2),round(mean_absolute_percentage_error(invest_XLM_GRU.target, invest_XLM_GRU.forcast),2),round(mean_absolute_percentage_error(bitfinex_XLM_GRU.target, bitfinex_XLM_GRU.forcast),2)],[round(mbv(yahoo_XLM_GRU.target, yahoo_XLM_GRU.forcast),2),round(mbv(invest_XLM_GRU.target, invest_XLM_GRU.forcast),2),round(mbv(bitfinex_XLM_GRU.target, bitfinex_XLM_GRU.forcast),2)],[round(nse(yahoo_XLM_GRU.target, yahoo_XLM_GRU.forcast),2),round(nse(invest_XLM_GRU.target, invest_XLM_GRU.forcast),2),round(nse(bitfinex_XLM_GRU.target, bitfinex_XLM_GRU.forcast),2)],
          [round(explained_variance_score(yahoo_XLM_MLP.target, yahoo_XLM_MLP.forcast),2),round(explained_variance_score(invest_XLM_MLP.target, invest_XLM_MLP.forcast),2),round(explained_variance_score(bitfinex_XLM_MLP.target, bitfinex_XLM_MLP.forcast),2)],[round(mean_absolute_percentage_error(yahoo_XLM_MLP.target, yahoo_XLM_MLP.forcast),2),round(mean_absolute_percentage_error(invest_XLM_MLP.target, invest_XLM_MLP.forcast),2),round(mean_absolute_percentage_error(bitfinex_XLM_MLP.target, bitfinex_XLM_MLP.forcast),2)],[round(mbv(yahoo_XLM_MLP.target, yahoo_XLM_MLP.forcast),2),round(mbv(invest_XLM_MLP.target, invest_XLM_MLP.forcast),2),round(mbv(bitfinex_XLM_MLP.target, bitfinex_XLM_MLP.forcast),2)],[round(nse(yahoo_XLM_MLP.target, yahoo_XLM_MLP.forcast),2),round(nse(invest_XLM_MLP.target, invest_XLM_MLP.forcast),2),round(nse(bitfinex_XLM_MLP.target, bitfinex_XLM_MLP.forcast),2)],
          [round(explained_variance_score(yahoo_XLM_CNN.target, yahoo_XLM_CNN.forcast),2),round(explained_variance_score(invest_XLM_CNN.target, invest_XLM_CNN.forcast),2),round(explained_variance_score(bitfinex_XLM_CNN.target, bitfinex_XLM_CNN.forcast),2)],[round(mean_absolute_percentage_error(yahoo_XLM_CNN.target, yahoo_XLM_CNN.forcast),2),round(mean_absolute_percentage_error(invest_XLM_CNN.target, invest_XLM_CNN.forcast),2),round(mean_absolute_percentage_error(bitfinex_XLM_CNN.target, bitfinex_XLM_CNN.forcast),2)],[round(mbv(yahoo_XLM_CNN.target, yahoo_XLM_CNN.forcast),2),round(mbv(invest_XLM_CNN.target, invest_XLM_CNN.forcast),2),round(mbv(bitfinex_XLM_CNN.target, bitfinex_XLM_CNN.forcast),2)],[round(nse(yahoo_XLM_CNN.target, yahoo_XLM_CNN.forcast),2),round(nse(invest_XLM_CNN.target, invest_XLM_CNN.forcast),2),round(nse(bitfinex_XLM_CNN.target, bitfinex_XLM_CNN.forcast),2)]]
    df_xlm= pd.DataFrame(xlm, columns=['YAH:XLM','INVS:XLM','BITF:XLM'])

      #compute explained variance score (evs), mean absolute percentage error (MAPE), t-test, and NSE for all models 
      #in the three datasets (Yahoo, UK investing and Bitfinex) for DOGE
    doge=[[round(explained_variance_score(yahoo_DOGE_XGB.target, yahoo_DOGE_XGB.forcast),2),round(explained_variance_score(invest_DOGE_XGB.target, invest_DOGE_XGB.forcast),2),round(explained_variance_score(bitfinex_DOGE_XGB.target, bitfinex_DOGE_XGB.forcast),2)],[round(mean_absolute_percentage_error(yahoo_DOGE_XGB.target, yahoo_DOGE_XGB.forcast),2),round(mean_absolute_percentage_error(invest_DOGE_XGB.target, invest_DOGE_XGB.forcast),2),round(mean_absolute_percentage_error(bitfinex_DOGE_XGB.target, bitfinex_DOGE_XGB.forcast),2)],[round(mbv(yahoo_DOGE_XGB.target, yahoo_DOGE_XGB.forcast),2),round(mbv(invest_DOGE_XGB.target, invest_DOGE_XGB.forcast),2),round(mbv(bitfinex_DOGE_XGB.target, bitfinex_DOGE_XGB.forcast),2)],[round(nse(yahoo_DOGE_XGB.target, yahoo_DOGE_XGB.forcast),2),round(nse(invest_DOGE_XGB.target, invest_DOGE_XGB.forcast),2),round(nse(bitfinex_DOGE_XGB.target, bitfinex_DOGE_XGB.forcast),2)],
          [round(explained_variance_score(yahoo_DOGE_GBM.target, yahoo_DOGE_GBM.forcast),2),round(explained_variance_score(invest_DOGE_GBM.target, invest_DOGE_GBM.forcast),2),round(explained_variance_score(bitfinex_DOGE_GBM.target, bitfinex_DOGE_GBM.forcast),2)],[round(mean_absolute_percentage_error(yahoo_DOGE_GBM.target, yahoo_DOGE_GBM.forcast),2),round(mean_absolute_percentage_error(invest_DOGE_GBM.target, invest_DOGE_GBM.forcast),2),round(mean_absolute_percentage_error(bitfinex_DOGE_GBM.target, bitfinex_DOGE_GBM.forcast),2)],[round(mbv(yahoo_DOGE_GBM.target, yahoo_DOGE_GBM.forcast),2),round(mbv(invest_DOGE_GBM.target, invest_DOGE_GBM.forcast),2),round(mbv(bitfinex_DOGE_GBM.target, bitfinex_DOGE_GBM.forcast),2)],[round(nse(yahoo_DOGE_GBM.target, yahoo_DOGE_GBM.forcast),2),round(nse(invest_DOGE_GBM.target, invest_DOGE_GBM.forcast),2),round(nse(bitfinex_DOGE_GBM.target, bitfinex_DOGE_GBM.forcast),2)],
          [round(explained_variance_score(yahoo_DOGE_ADA.target, yahoo_DOGE_ADA.forcast),2),round(explained_variance_score(invest_DOGE_ADA.target, invest_DOGE_ADA.forcast),2),round(explained_variance_score(bitfinex_DOGE_ADA.target, bitfinex_DOGE_ADA.forcast),2)],[round(mean_absolute_percentage_error(yahoo_DOGE_ADA.target, yahoo_DOGE_ADA.forcast),2),round(mean_absolute_percentage_error(invest_DOGE_ADA.target, invest_DOGE_ADA.forcast),2),round(mean_absolute_percentage_error(bitfinex_DOGE_ADA.target, bitfinex_DOGE_ADA.forcast),2)],[round(mbv(yahoo_DOGE_ADA.target, yahoo_DOGE_ADA.forcast),2),round(mbv(invest_DOGE_ADA.target, invest_DOGE_ADA.forcast),2),round(mbv(bitfinex_DOGE_ADA.target, bitfinex_DOGE_ADA.forcast),2)],[round(nse(yahoo_DOGE_ADA.target, yahoo_DOGE_ADA.forcast),2),round(nse(invest_DOGE_ADA.target, invest_DOGE_ADA.forcast),2),round(nse(bitfinex_DOGE_ADA.target, bitfinex_DOGE_ADA.forcast),2)],
          [round(explained_variance_score(yahoo_DOGE_GRU.target, yahoo_DOGE_GRU.forcast),2),round(explained_variance_score(invest_DOGE_GRU.target, invest_DOGE_GRU.forcast),2),round(explained_variance_score(bitfinex_DOGE_GRU.target, bitfinex_DOGE_GRU.forcast),2)],[round(mean_absolute_percentage_error(yahoo_DOGE_GRU.target, yahoo_DOGE_GRU.forcast),2),round(mean_absolute_percentage_error(invest_DOGE_GRU.target, invest_DOGE_GRU.forcast),2),round(mean_absolute_percentage_error(bitfinex_DOGE_GRU.target, bitfinex_DOGE_GRU.forcast),2)],[round(mbv(yahoo_DOGE_GRU.target, yahoo_DOGE_GRU.forcast),2),round(mbv(invest_DOGE_GRU.target, invest_DOGE_GRU.forcast),2),round(mbv(bitfinex_DOGE_GRU.target, bitfinex_DOGE_GRU.forcast),2)],[round(nse(yahoo_DOGE_GRU.target, yahoo_DOGE_GRU.forcast),2),round(nse(invest_DOGE_GRU.target, invest_DOGE_GRU.forcast),2),round(nse(bitfinex_DOGE_GRU.target, bitfinex_DOGE_GRU.forcast),2)],
          [round(explained_variance_score(yahoo_DOGE_MLP.target, yahoo_DOGE_MLP.forcast),2),round(explained_variance_score(invest_DOGE_MLP.target, invest_DOGE_MLP.forcast),2),round(explained_variance_score(bitfinex_DOGE_MLP.target, bitfinex_DOGE_MLP.forcast),2)],[round(mean_absolute_percentage_error(yahoo_DOGE_MLP.target, yahoo_DOGE_MLP.forcast),2),round(mean_absolute_percentage_error(invest_DOGE_MLP.target, invest_DOGE_MLP.forcast),2),round(mean_absolute_percentage_error(bitfinex_DOGE_MLP.target, bitfinex_DOGE_MLP.forcast),2)],[round(mbv(yahoo_DOGE_MLP.target, yahoo_DOGE_MLP.forcast),2),round(mbv(invest_DOGE_MLP.target, invest_DOGE_MLP.forcast),2),round(mbv(bitfinex_DOGE_MLP.target, bitfinex_DOGE_MLP.forcast),2)],[round(nse(yahoo_DOGE_MLP.target, yahoo_DOGE_MLP.forcast),2),round(nse(invest_DOGE_MLP.target, invest_DOGE_MLP.forcast),2),round(nse(bitfinex_DOGE_MLP.target, bitfinex_DOGE_MLP.forcast),2)],
          [round(explained_variance_score(yahoo_DOGE_CNN.target, yahoo_DOGE_CNN.forcast),2),round(explained_variance_score(invest_DOGE_CNN.target, invest_DOGE_CNN.forcast),2),round(explained_variance_score(bitfinex_DOGE_CNN.target, bitfinex_DOGE_CNN.forcast),2)],[round(mean_absolute_percentage_error(yahoo_DOGE_CNN.target, yahoo_DOGE_CNN.forcast),2),round(mean_absolute_percentage_error(invest_DOGE_CNN.target, invest_DOGE_CNN.forcast),2),round(mean_absolute_percentage_error(bitfinex_DOGE_CNN.target, bitfinex_DOGE_CNN.forcast),2)],[round(mbv(yahoo_DOGE_CNN.target, yahoo_DOGE_CNN.forcast),2),round(mbv(invest_DOGE_CNN.target, invest_DOGE_CNN.forcast),2),round(mbv(bitfinex_DOGE_CNN.target, bitfinex_DOGE_CNN.forcast),2)],[round(nse(yahoo_DOGE_CNN.target, yahoo_DOGE_CNN.forcast),2),round(nse(invest_DOGE_CNN.target, invest_DOGE_CNN.forcast),2),round(nse(bitfinex_DOGE_CNN.target, bitfinex_DOGE_CNN.forcast),2)]]
    df_doge = pd.DataFrame(doge, columns=['YAH:DOGE','INVS:DOGE','BITF:DOGE'])
   
    performance_table=pd.concat([df_btc,df_eth,df_bnb,df_ltc,df_xlm,df_doge], axis=1)
    print("=========Performance Comparison of Models===========\n")
    performance_table["Mean"] = round(performance_table.iloc[:, [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].mean(axis=1),2)
    if(scenario==1):
        performance_table.to_csv('../results/Table4A.csv', index=False)
        print("Comparative Performance results (Scenario A) stored in result/Table 4A.csv")
    else:
        performance_table.to_csv('../results/Table4B.csv', index=False)
        print("Comparative Performance results (Scenario B) stored in result/Table 4B.csv")

    # add model and cryptocurrency type columns to predtions' dataframes 
    yahoo_BTC_XGB=add_columns_results(yahoo_BTC_XGB,1,3)
    yahoo_LTC_XGB=add_columns_results(yahoo_LTC_XGB,4,3)
    yahoo_ETH_XGB=add_columns_results(yahoo_ETH_XGB,2,3)
    yahoo_BNB_XGB=add_columns_results(yahoo_BNB_XGB,3,3)
    yahoo_XLM_XGB=add_columns_results(yahoo_XLM_XGB,5,3)
    yahoo_DOGE_XGB=add_columns_results(yahoo_DOGE_XGB,6,3)
    
    yahoo_BTC_GBM=add_columns_results(yahoo_BTC_GBM,1,2)
    yahoo_LTC_GBM=add_columns_results(yahoo_LTC_GBM,4,2)
    yahoo_ETH_GBM=add_columns_results(yahoo_ETH_GBM,2,2)
    yahoo_BNB_GBM=add_columns_results(yahoo_BNB_GBM,3,2)
    yahoo_XLM_GBM=add_columns_results(yahoo_XLM_GBM,5,2)
    yahoo_DOGE_GBM=add_columns_results(yahoo_DOGE_GBM,6,2)

    yahoo_BTC_ADA=add_columns_results(yahoo_BTC_ADA,1,1)
    yahoo_LTC_ADA=add_columns_results(yahoo_LTC_ADA,4,1)
    yahoo_ETH_ADA=add_columns_results(yahoo_ETH_ADA,2,1)
    yahoo_BNB_ADA=add_columns_results(yahoo_BNB_ADA,3,1)
    yahoo_XLM_ADA=add_columns_results(yahoo_XLM_ADA,5,1)
    yahoo_DOGE_ADA=add_columns_results(yahoo_DOGE_ADA,6,1)

    yahoo_BTC_GRU=add_columns_results(yahoo_BTC_GRU,1,4)
    yahoo_LTC_GRU=add_columns_results(yahoo_LTC_GRU,4,4)
    yahoo_ETH_GRU=add_columns_results(yahoo_ETH_GRU,2,4)
    yahoo_BNB_GRU=add_columns_results(yahoo_BNB_GRU,3,4)
    yahoo_XLM_GRU=add_columns_results(yahoo_XLM_GRU,5,4)
    yahoo_DOGE_GRU=add_columns_results(yahoo_DOGE_GRU,6,4)
    
    yahoo_BTC_MLP=add_columns_results(yahoo_BTC_MLP,1,5)
    yahoo_LTC_MLP=add_columns_results(yahoo_LTC_MLP,4,5)
    yahoo_ETH_MLP=add_columns_results(yahoo_ETH_MLP,2,5)
    yahoo_BNB_MLP=add_columns_results(yahoo_BNB_MLP,3,5)
    yahoo_XLM_MLP=add_columns_results(yahoo_XLM_MLP,5,5)
    yahoo_DOGE_MLP=add_columns_results(yahoo_DOGE_MLP,6,5)
    
    yahoo_BTC_CNN=add_columns_results(yahoo_BTC_CNN,1,6)
    yahoo_LTC_CNN=add_columns_results(yahoo_LTC_CNN,4,6)
    yahoo_ETH_CNN=add_columns_results(yahoo_ETH_CNN,2,6)
    yahoo_BNB_CNN=add_columns_results(yahoo_BNB_CNN,3,6)
    yahoo_XLM_CNN=add_columns_results(yahoo_XLM_CNN,5,6)
    yahoo_DOGE_CNN=add_columns_results(yahoo_DOGE_CNN,6,6)
    
    invest_BTC_XGB=add_columns_results(invest_BTC_XGB,1,3)
    invest_LTC_XGB=add_columns_results(invest_LTC_XGB,4,3)
    invest_ETH_XGB=add_columns_results(invest_ETH_XGB,2,3)
    invest_BNB_XGB=add_columns_results(invest_BNB_XGB,3,3)
    invest_XLM_XGB=add_columns_results(invest_XLM_XGB,5,3)
    invest_DOGE_XGB=add_columns_results(invest_DOGE_XGB,6,3)
    
    invest_BTC_GBM=add_columns_results(invest_BTC_GBM,1,2)
    invest_LTC_GBM=add_columns_results(invest_LTC_GBM,4,2)
    invest_ETH_GBM=add_columns_results(invest_ETH_GBM,2,2)
    invest_BNB_GBM=add_columns_results(invest_BNB_GBM,3,2)
    invest_XLM_GBM=add_columns_results(invest_XLM_GBM,5,2)
    invest_DOGE_GBM=add_columns_results(invest_DOGE_GBM,6,2)
    
    invest_BTC_ADA=add_columns_results(invest_BTC_ADA,1,1)
    invest_LTC_ADA=add_columns_results(invest_LTC_ADA,4,1)
    invest_ETH_ADA=add_columns_results(invest_ETH_ADA,2,1)
    invest_BNB_ADA=add_columns_results(invest_BNB_ADA,3,1)
    invest_XLM_ADA=add_columns_results(invest_XLM_ADA,5,1)
    invest_DOGE_ADA=add_columns_results(invest_DOGE_ADA,6,1)

    invest_BTC_GRU=add_columns_results(invest_BTC_GRU,1,4)
    invest_LTC_GRU=add_columns_results(invest_LTC_GRU,4,4)
    invest_ETH_GRU=add_columns_results(invest_ETH_GRU,2,4)
    invest_BNB_GRU=add_columns_results(invest_BNB_GRU,3,4)
    invest_XLM_GRU=add_columns_results(invest_XLM_GRU,5,4)
    invest_DOGE_GRU=add_columns_results(invest_DOGE_GRU,6,4)
    
    invest_BTC_MLP=add_columns_results(invest_BTC_MLP,1,5)
    invest_LTC_MLP=add_columns_results(invest_LTC_MLP,4,5)
    invest_ETH_MLP=add_columns_results(invest_ETH_MLP,2,5)
    invest_BNB_MLP=add_columns_results(invest_BNB_MLP,3,5)
    invest_XLM_MLP=add_columns_results(invest_XLM_MLP,5,5)
    invest_DOGE_MLP=add_columns_results(invest_DOGE_MLP,6,5)

    invest_BTC_CNN=add_columns_results(invest_BTC_CNN,1,6)
    invest_LTC_CNN=add_columns_results(invest_LTC_CNN,4,6)
    invest_ETH_CNN=add_columns_results(invest_ETH_CNN,2,6)
    invest_BNB_CNN=add_columns_results(invest_BNB_CNN,3,6)
    invest_XLM_CNN=add_columns_results(invest_XLM_CNN,5,6)
    invest_DOGE_CNN=add_columns_results(invest_DOGE_CNN,6,6)
    
    bitfinex_BTC_XGB=add_columns_results(bitfinex_BTC_XGB,1,3)
    bitfinex_LTC_XGB=add_columns_results(bitfinex_LTC_XGB,4,3)
    bitfinex_ETH_XGB=add_columns_results(bitfinex_ETH_XGB,2,3)
    bitfinex_BNB_XGB=add_columns_results(bitfinex_BNB_XGB,3,3)
    bitfinex_XLM_XGB=add_columns_results(bitfinex_XLM_XGB,5,3)
    bitfinex_DOGE_XGB=add_columns_results(bitfinex_DOGE_XGB,6,3)
    
    bitfinex_BTC_GBM=add_columns_results(bitfinex_BTC_GBM,1,2)
    bitfinex_LTC_GBM=add_columns_results(bitfinex_LTC_GBM,4,2)
    bitfinex_ETH_GBM=add_columns_results(bitfinex_ETH_GBM,2,2)
    bitfinex_BNB_GBM=add_columns_results(bitfinex_BNB_GBM,3,2)
    bitfinex_XLM_GBM=add_columns_results(bitfinex_XLM_GBM,5,2)
    bitfinex_DOGE_GBM=add_columns_results(bitfinex_DOGE_GBM,6,2)

    bitfinex_BTC_ADA=add_columns_results(bitfinex_BTC_ADA,1,1)
    bitfinex_LTC_ADA=add_columns_results(bitfinex_LTC_ADA,4,1)
    bitfinex_ETH_ADA=add_columns_results(bitfinex_ETH_ADA,2,1)
    bitfinex_BNB_ADA=add_columns_results(bitfinex_BNB_ADA,3,1)
    bitfinex_XLM_ADA=add_columns_results(bitfinex_XLM_ADA,5,1)
    bitfinex_DOGE_ADA=add_columns_results(bitfinex_DOGE_ADA,6,1)
    
    bitfinex_BTC_GRU=add_columns_results(bitfinex_BTC_GRU,1,4)
    bitfinex_LTC_GRU=add_columns_results(bitfinex_LTC_GRU,4,4)
    bitfinex_ETH_GRU=add_columns_results(bitfinex_ETH_GRU,2,4)
    bitfinex_BNB_GRU=add_columns_results(bitfinex_BNB_GRU,3,4)
    bitfinex_XLM_GRU=add_columns_results(bitfinex_XLM_GRU,5,4)
    bitfinex_DOGE_GRU=add_columns_results(bitfinex_DOGE_GRU,6,4)

    bitfinex_BTC_MLP=add_columns_results(bitfinex_BTC_MLP,1,5)
    bitfinex_LTC_MLP=add_columns_results(bitfinex_LTC_MLP,4,5)
    bitfinex_ETH_MLP=add_columns_results(bitfinex_ETH_MLP,2,5)
    bitfinex_BNB_MLP=add_columns_results(bitfinex_BNB_MLP,3,5)
    bitfinex_XLM_MLP=add_columns_results(bitfinex_XLM_MLP,5,5)
    bitfinex_DOGE_MLP=add_columns_results(bitfinex_DOGE_MLP,6,5)
    
    bitfinex_BTC_CNN=add_columns_results(bitfinex_BTC_CNN,1,6)
    bitfinex_LTC_CNN=add_columns_results(bitfinex_LTC_CNN,4,6)
    bitfinex_ETH_CNN=add_columns_results(bitfinex_ETH_CNN,2,6)
    bitfinex_BNB_CNN=add_columns_results(bitfinex_BNB_CNN,3,6)
    bitfinex_XLM_CNN=add_columns_results(bitfinex_XLM_CNN,5,6)
    bitfinex_DOGE_CNN=add_columns_results(bitfinex_DOGE_CNN,6,6)

    # combine all dataframes containing results from the yahoo Finance datasets into yahoo_result
    yahoo_result=concat_dataframes(yahoo_BTC_XGB,  yahoo_BTC_GBM, yahoo_BTC_ADA, yahoo_BTC_GRU, yahoo_BTC_MLP, yahoo_BTC_CNN,yahoo_LTC_XGB,  yahoo_LTC_GBM, yahoo_LTC_ADA, yahoo_LTC_GRU, yahoo_LTC_MLP, yahoo_LTC_CNN,yahoo_ETH_XGB,  yahoo_ETH_GBM, yahoo_ETH_ADA, yahoo_ETH_GRU, yahoo_ETH_MLP, yahoo_ETH_CNN,yahoo_BNB_XGB,  yahoo_BNB_GBM, yahoo_BNB_ADA, yahoo_BNB_GRU, yahoo_BNB_MLP, yahoo_BNB_CNN,yahoo_DOGE_XGB,yahoo_DOGE_GBM,yahoo_DOGE_ADA,yahoo_DOGE_GRU,yahoo_DOGE_MLP,yahoo_DOGE_CNN,yahoo_XLM_XGB,  yahoo_XLM_GBM, yahoo_XLM_ADA, yahoo_XLM_GRU, yahoo_XLM_MLP,yahoo_XLM_CNN)

    # combine all dataframes containing results from the UK investing datasets into invest_result
    invest_result =concat_dataframes(invest_BTC_XGB,  invest_BTC_GBM, invest_BTC_ADA, invest_BTC_GRU, invest_BTC_MLP, invest_BTC_CNN,invest_LTC_XGB,  invest_LTC_GBM, invest_LTC_ADA, invest_LTC_GRU, invest_LTC_MLP, invest_LTC_CNN,invest_ETH_XGB,  invest_ETH_GBM, invest_ETH_ADA, invest_ETH_GRU, invest_ETH_MLP, invest_ETH_CNN,invest_BNB_XGB,  invest_BNB_GBM, invest_BNB_ADA, invest_BNB_GRU, invest_BNB_MLP, invest_BNB_CNN,invest_DOGE_XGB,invest_DOGE_GBM,invest_DOGE_ADA,invest_DOGE_GRU,invest_DOGE_MLP,invest_DOGE_CNN,invest_XLM_XGB,  invest_XLM_GBM, invest_XLM_ADA, invest_XLM_GRU, invest_XLM_MLP, invest_XLM_CNN)
    
    # combine all dataframes containing results from the Bitfinex datasets into bitfinex_result
    bitfinex_result =concat_dataframes(bitfinex_BTC_XGB,  bitfinex_BTC_GBM, bitfinex_BTC_ADA, bitfinex_BTC_GRU, bitfinex_BTC_MLP, bitfinex_BTC_CNN,bitfinex_LTC_XGB,  bitfinex_LTC_GBM, bitfinex_LTC_ADA, bitfinex_LTC_GRU, bitfinex_LTC_MLP, bitfinex_LTC_CNN,bitfinex_ETH_XGB,  bitfinex_ETH_GBM, bitfinex_ETH_ADA, bitfinex_ETH_GRU, bitfinex_ETH_MLP, bitfinex_ETH_CNN,bitfinex_BNB_XGB,  bitfinex_BNB_GBM, bitfinex_BNB_ADA, bitfinex_BNB_GRU, bitfinex_BNB_MLP, bitfinex_BNB_CNN,bitfinex_DOGE_XGB,bitfinex_DOGE_GBM,bitfinex_DOGE_ADA,bitfinex_DOGE_GRU,bitfinex_DOGE_MLP,bitfinex_DOGE_CNN,bitfinex_XLM_XGB,  bitfinex_XLM_GBM, bitfinex_XLM_ADA, bitfinex_XLM_GRU, bitfinex_XLM_MLP, bitfinex_XLM_CNN)

    # compute error residual
    for index, row in yahoo_result.iterrows():
        yahoo_result.at[index,'residual']=round(yahoo_result.at[index,'target']-yahoo_result.at[index,'forcast'],5)
    for index, row in invest_result.iterrows():
        invest_result.at[index,'residual']=round(invest_result.at[index,'target']-invest_result.at[index,'forcast'],5)
    for index, row in bitfinex_result.iterrows():
        bitfinex_result.at[index,'residual']=round(bitfinex_result.at[index,'target']-bitfinex_result.at[index,'forcast'],5)
    
    display_graph(yahoo_result, invest_result, bitfinex_result)   #display graphs

    return yahoo_result, invest_result, bitfinex_result  # return dataframes of prediction results

#--------------------------------------------------------------------------------------
# module display_graph                                                                #
#--------------------------------------------------------------------------------------
def display_graph(res1,res2,res3):
    if (scenario==1):
        data=res1.copy()
        dset='Yahoo Finance'
    elif(scenario==2):
        data=res2.copy()
        dset='UK Investing'
    # intialize multiple lists to store temporary results from predictive models
    gbm_mbth,gbm_meth,gbm_mbnb,gbm_mltc,gbm_mxlm,gbm_mdoge,ada_mbth,ada_meth,ada_mbnb,ada_mltc,ada_mxlm,ada_mdoge,xgb_mbth,xgb_meth,xgb_mbnb,xgb_mltc,xgb_mxlm,xgb_mdoge,cnn_mbth,cnn_meth,cnn_mbnb,cnn_mltc,cnn_mxlm,cnn_mdoge,mlp_mbth,mlp_meth,mlp_mbnb,mlp_mltc,mlp_mxlm,mlp_mdoge,gru_mbth,gru_meth,gru_mbnb,gru_mltc,gru_mxlm,gru_mdoge = ([] for i in range(36))
    
    ada_mbth.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'BTC')].forcast.to_numpy())
    ada_meth.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'ETH')].forcast.to_numpy())
    ada_mbnb.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'BNB')].forcast.to_numpy())
    ada_mltc.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'LTC')].forcast.to_numpy())
    ada_mxlm.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'XLM')].forcast.to_numpy())
    ada_mdoge.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'DOGE')].forcast.to_numpy())
    gbm_mbth.append(data[(data['model']== 'GBM') & (data['bitcoin']== 'BTC')].forcast.to_numpy())
    gbm_meth.append(data[(data['model']== 'GBM') & (data['bitcoin']== 'ETH')].forcast.to_numpy())
    gbm_mbnb.append(data[(data['model']== 'GBM') & (data['bitcoin']== 'BNB')].forcast.to_numpy())
    gbm_mltc.append(data[(data['model']== 'GBM') & (data['bitcoin']== 'LTC')].forcast.to_numpy())
    gbm_mxlm.append(data[(data['model']== 'GBM') & (data['bitcoin']== 'XLM')].forcast.to_numpy())
    gbm_mdoge.append(data[(data['model']== 'GBM') & (data['bitcoin']== 'DOGE')].forcast.to_numpy())
    xgb_mbth.append(data[(data['model']== 'XGB') & (data['bitcoin']== 'BTC')].forcast.to_numpy())
    xgb_meth.append(data[(data['model']== 'XGB') & (data['bitcoin']== 'ETH')].forcast.to_numpy())
    xgb_mbnb.append(data[(data['model']== 'XGB') & (data['bitcoin']== 'BNB')].forcast.to_numpy())
    xgb_mltc.append(data[(data['model']== 'XGB') & (data['bitcoin']== 'LTC')].forcast.to_numpy())
    xgb_mxlm.append(data[(data['model']== 'XGB') & (data['bitcoin']== 'XLM')].forcast.to_numpy())
    xgb_mdoge.append(data[(data['model']== 'XGB') & (data['bitcoin']== 'DOGE')].forcast.to_numpy())
    cnn_mbth.append(data[(data['model']== 'CNN') & (data['bitcoin']== 'BTC')].forcast.to_numpy())
    cnn_meth.append(data[(data['model']== 'CNN') & (data['bitcoin']== 'ETH')].forcast.to_numpy())
    cnn_mbnb.append(data[(data['model']== 'CNN') & (data['bitcoin']== 'BNB')].forcast.to_numpy())
    cnn_mltc.append(data[(data['model']== 'CNN') & (data['bitcoin']== 'LTC')].forcast.to_numpy())
    cnn_mxlm.append(data[(data['model']== 'CNN') & (data['bitcoin']== 'XLM')].forcast.to_numpy())
    cnn_mdoge.append(data[(data['model']== 'CNN') & (data['bitcoin']== 'DOGE')].forcast.to_numpy())
    mlp_mbth.append(data[(data['model']== 'MLP') & (data['bitcoin']== 'BTC')].forcast.to_numpy())
    mlp_meth.append(data[(data['model']== 'MLP') & (data['bitcoin']== 'ETH')].forcast.to_numpy())
    mlp_mbnb.append(data[(data['model']== 'MLP') & (data['bitcoin']== 'BNB')].forcast.to_numpy())
    mlp_mltc.append(data[(data['model']== 'MLP') & (data['bitcoin']== 'LTC')].forcast.to_numpy())
    mlp_mxlm.append(data[(data['model']== 'MLP') & (data['bitcoin']== 'XLM')].forcast.to_numpy())
    mlp_mdoge.append(data[(data['model']== 'MLP') & (data['bitcoin']== 'DOGE')].forcast.to_numpy())
    gru_mbth.append(data[(data['model']== 'GRU') & (data['bitcoin']== 'BTC')].forcast.to_numpy())
    gru_meth.append(data[(data['model']== 'GRU') & (data['bitcoin']== 'ETH')].forcast.to_numpy())
    gru_mbnb.append(data[(data['model']== 'GRU') & (data['bitcoin']== 'BNB')].forcast.to_numpy())
    gru_mltc.append(data[(data['model']== 'GRU') & (data['bitcoin']== 'LTC')].forcast.to_numpy())
    gru_mxlm.append(data[(data['model']== 'GRU') & (data['bitcoin']== 'XLM')].forcast.to_numpy())
    gru_mdoge.append(data[(data['model']== 'GRU') & (data['bitcoin']== 'DOGE')].forcast.to_numpy())
        
    obtc,oeth,obnb,oltc,oxlm,odoge,observations=([] for i in range(7)) #initialize lists to store the measured values for all cryptocurrencies

    obtc.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'BTC')].target.to_numpy())
    oeth.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'ETH')].target.to_numpy())
    obnb.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'BNB')].target.to_numpy())
    oltc.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'LTC')].target.to_numpy())
    oxlm.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'XLM')].target.to_numpy())
    odoge.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'DOGE')].target.to_numpy())

    observations.extend([pd.DataFrame(data=np.transpose(obtc[0][:]), columns=['target']), pd.DataFrame(data=np.transpose(oeth[0][:]), columns=['target']), 
                         pd.DataFrame(data=np.transpose(obnb[0][:]), columns=['target']), pd.DataFrame(data=np.transpose(oltc[0][:]), columns=['target']), 
                         pd.DataFrame(data=np.transpose(oxlm[0][:]), columns=['target']), pd.DataFrame(data=np.transpose(odoge[0][:]), columns=['target'])])
 
    ada_predictions,gbm_predictions,xgb_predictions,cnn_predictions,mlp_predictions,gru_predictions=([] for i in range(6)) #initialize lists

    ada_predictions=insert_list(ada_predictions, ada_mbth[0][:], ada_meth[0][:], ada_mbnb[0][:], ada_mltc[0][:], ada_mxlm[0][:], ada_mdoge[0][:])
    gbm_predictions=insert_list(gbm_predictions, gbm_mbth[0][:], gbm_meth[0][:], gbm_mbnb[0][:], gbm_mltc[0][:], gbm_mxlm[0][:], gbm_mdoge[0][:])
    xgb_predictions=insert_list(xgb_predictions, xgb_mbth[0][:], xgb_meth[0][:], xgb_mbnb[0][:], xgb_mltc[0][:], xgb_mxlm[0][:], xgb_mdoge[0][:])
    gru_predictions=insert_list(gru_predictions, gru_mbth[0][:], gru_meth[0][:], gru_mbnb[0][:], gru_mltc[0][:], gru_mxlm[0][:], gru_mdoge[0][:])
    mlp_predictions=insert_list(mlp_predictions, mlp_mbth[0][:], mlp_meth[0][:], mlp_mbnb[0][:], mlp_mltc[0][:], mlp_mxlm[0][:], mlp_mdoge[0][:])
    cnn_predictions=insert_list(cnn_predictions, cnn_mbth[0][:], cnn_meth[0][:], cnn_mbnb[0][:], cnn_mltc[0][:], cnn_mxlm[0][:], cnn_mdoge[0][:])
     
    
    combine_plots_tree(observations,ada_predictions,gbm_predictions,xgb_predictions,model_name,1,dset)
    combine_plots_DNN(observations,cnn_predictions,mlp_predictions,gru_predictions,model_name,1,dset)
        
    # Prepare data to plot residuals ----------------------------------------------
    #initialize multiple lists to store the models' residuals
    gbm_mbth1,gbm_meth1,gbm_mbnb1,gbm_mltc1,gbm_mxlm1,gbm_mdoge1,ada_mbth1,ada_meth1,ada_mbnb1,ada_mltc1,ada_mxlm1,ada_mdoge1,xgb_mbth1,xgb_meth1,xgb_mbnb1,xgb_mltc1,xgb_mxlm1,xgb_mdoge1,cnn_mbth1,cnn_meth1,cnn_mbnb1,cnn_mltc1,cnn_mxlm1,cnn_mdoge1,mlp_mbth1,mlp_meth1,mlp_mbnb1,mlp_mltc1,mlp_mxlm1,mlp_mdoge1,gru_mbth1,gru_meth1,gru_mbnb1,gru_mltc1,gru_mxlm1,gru_mdoge1 = ([] for i in range(36))
    
    ada_mbth1.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'BTC')].residual)
    ada_meth1.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'ETH')].residual)
    ada_mbnb1.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'BNB')].residual)
    ada_mltc1.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'LTC')].residual)
    ada_mxlm1.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'XLM')].residual)
    ada_mdoge1.append(data[(data['model']== 'ADA') & (data['bitcoin']== 'DOGE')].residual)
    gbm_mbth1.append(data[(data['model']== 'GBM') & (data['bitcoin']== 'BTC')].residual)
    gbm_meth1.append(data[(data['model']== 'GBM') & (data['bitcoin']== 'ETH')].residual)
    gbm_mbnb1.append(data[(data['model']== 'GBM') & (data['bitcoin']== 'BNB')].residual)
    gbm_mltc1.append(data[(data['model']== 'GBM') & (data['bitcoin']== 'LTC')].residual)
    gbm_mxlm1.append(data[(data['model']== 'GBM') & (data['bitcoin']== 'XLM')].residual)
    gbm_mdoge1.append(data[(data['model']== 'GBM') & (data['bitcoin']== 'DOGE')].residual)
    xgb_mbth1.append(data[(data['model']== 'XGB') & (data['bitcoin']== 'BTC')].residual)
    xgb_meth1.append(data[(data['model']== 'XGB') & (data['bitcoin']== 'ETH')].residual)
    xgb_mbnb1.append(data[(data['model']== 'XGB') & (data['bitcoin']== 'BNB')].residual)
    xgb_mltc1.append(data[(data['model']== 'XGB') & (data['bitcoin']== 'LTC')].residual)
    xgb_mxlm1.append(data[(data['model']== 'XGB') & (data['bitcoin']== 'XLM')].residual)
    xgb_mdoge1.append(data[(data['model']== 'XGB') & (data['bitcoin']== 'DOGE')].residual)
    cnn_mbth1.append(data[(data['model']== 'CNN') & (data['bitcoin']== 'BTC')].residual)
    cnn_meth1.append(data[(data['model']== 'CNN') & (data['bitcoin']== 'ETH')].residual)
    cnn_mbnb1.append(data[(data['model']== 'CNN') & (data['bitcoin']== 'BNB')].residual)
    cnn_mltc1.append(data[(data['model']== 'CNN') & (data['bitcoin']== 'LTC')].residual)
    cnn_mxlm1.append(data[(data['model']== 'CNN') & (data['bitcoin']== 'XLM')].residual)
    cnn_mdoge1.append(data[(data['model']== 'CNN') & (data['bitcoin']== 'DOGE')].residual)
    mlp_mbth1.append(data[(data['model']== 'MLP') & (data['bitcoin']== 'BTC')].residual)
    mlp_meth1.append(data[(data['model']== 'MLP') & (data['bitcoin']== 'ETH')].residual)
    mlp_mbnb1.append(data[(data['model']== 'MLP') & (data['bitcoin']== 'BNB')].residual)
    mlp_mltc1.append(data[(data['model']== 'MLP') & (data['bitcoin']== 'LTC')].residual)
    mlp_mxlm1.append(data[(data['model']== 'MLP') & (data['bitcoin']== 'XLM')].residual)
    mlp_mdoge1.append(data[(data['model']== 'MLP') & (data['bitcoin']== 'DOGE')].residual)
    gru_mbth1.append(data[(data['model']== 'GRU') & (data['bitcoin']== 'BTC')].residual)
    gru_meth1.append(data[(data['model']== 'GRU') & (data['bitcoin']== 'ETH')].residual)
    gru_mbnb1.append(data[(data['model']== 'GRU') & (data['bitcoin']== 'BNB')].residual)
    gru_mltc1.append(data[(data['model']== 'GRU') & (data['bitcoin']== 'LTC')].residual)
    gru_mxlm1.append(data[(data['model']== 'GRU') & (data['bitcoin']== 'XLM')].residual)
    gru_mdoge1.append(data[(data['model']== 'GRU') & (data['bitcoin']== 'DOGE')].residual)
        
    bth_forcast, bth_residual=combine_lists(ada_mbth,gbm_mbth,xgb_mbth,cnn_mbth,mlp_mbth,gru_mbth,ada_mbth1,gbm_mbth1,xgb_mbth1,cnn_mbth1,mlp_mbth1,gru_mbth1)
    eth_forcast, eth_residual=combine_lists(ada_meth,gbm_meth,xgb_meth,cnn_meth,mlp_meth,gru_meth,ada_meth1,gbm_meth1,xgb_meth1,cnn_meth1,mlp_meth1,gru_meth1)
    bnb_forcast, bnb_residual=combine_lists(ada_mbnb,gbm_mbnb,xgb_mbnb,cnn_mbnb,mlp_mbnb,gru_mbnb,ada_mbnb1,gbm_mbnb1,xgb_mbnb1,cnn_mbnb1,mlp_mbnb1,gru_mbnb1)
    ltc_forcast, ltc_residual=combine_lists(ada_mltc,gbm_mltc,xgb_mltc,cnn_mltc,mlp_mltc,gru_mltc,ada_mltc1,gbm_mltc1,xgb_mltc1,cnn_mltc1,mlp_mltc1,gru_mltc1)
    xlm_forcast, xlm_residual=combine_lists(ada_mxlm,gbm_mxlm,xgb_mxlm,cnn_mxlm, mlp_mxlm,gru_mxlm,ada_mxlm1,gbm_mxlm1,xgb_mxlm1,cnn_mxlm1,mlp_mxlm1,gru_mxlm1)
    doge_forcast, doge_residual=combine_lists(ada_mdoge,gbm_mdoge,xgb_mdoge,cnn_mdoge,mlp_mdoge,gru_mdoge,ada_mdoge1,gbm_mdoge1,xgb_mdoge1,cnn_mdoge1,mlp_mdoge1, gru_mdoge1)

    #plot residuals for the tree-based models
    plot_residuals_tree(bth_forcast, bth_residual, eth_forcast,eth_residual, bnb_forcast,bnb_residual,ltc_forcast, ltc_residual, xlm_forcast, xlm_residual, doge_forcast, doge_residual,dset)
    #plot residuals for the deep learning models
    plot_residuals_DNN(bth_forcast, bth_residual, eth_forcast,eth_residual, bnb_forcast,bnb_residual,ltc_forcast, ltc_residual, xlm_forcast, xlm_residual, doge_forcast, doge_residual,dset) 
    
    make_TaylorDiag_data(data) #plot Taylor's diagrams
    return

#--------------------------------------------------------------------------------------
# module create_model                                                                 #
#--------------------------------------------------------------------------------------
def create_model(data1,data2,data3, dparam,typ):
    #dparam = dataframe for optimized parameters
    #typ = an integer representing the model type 1-XGB, 2- GBM, 3-ADA, 4-GRU etc
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    temp_data = scaler.fit_transform(data1.values) #training and testing set Yahoo Finance
    train = temp_data[:n_train_length, :]
    test = temp_data[ n_train_length:, :]
    train_features, train_target = train[:, 1:11], train[:, 0] # train_features (features), train_target (target for training set)
    test_features, test_target = test[:, 1:11], test[:, 0] # test_features (features), test_target (target for testing set)
    
    if(typ==1):
        mdl = xgb.XGBRegressor(learning_rate = round(np.mean(dparam.learning_rate),5),max_depth=int(round(np.mean(dparam.max_depth),0)), n_estimators = int(round(np.mean(dparam.n_estimators),0)))
        mdl.fit(train_features, train_target)
    elif(typ==2): 
        mdl = GradientBoostingRegressor(learning_rate = round(np.mean(dparam.learning_rate),5),max_depth = int(round(np.mean(dparam.max_depth),0)), n_estimators = int(round(np.mean(dparam.n_estimators),0)))
        mdl.fit(train_features, train_target)
    elif(typ==3):
        mdl = AdaBoostRegressor(learning_rate = round(np.mean(dparam.learning_rate),5), n_estimators = int(round(np.mean(dparam.n_estimators),0)), loss = 'square') 
        mdl.fit(train_features, train_target)
    elif(typ==4):
        mdl=create_gru_model(int(round(np.mean(dparam.units),0)))
        mdl.fit(train_features.reshape(train_features.shape[0], 1, train_features.shape[1]), train_target, epochs=int(round(np.mean(dparam.epochs),0)), batch_size=int(round(np.mean(dparam.batch_size),0)), validation_data=(test_features.reshape(test_features.shape[0], 1, test_features.shape[1]), test_target), verbose=0, shuffle=False)
    elif(typ==5):
        mdl=create_mlp_model(int(round(np.mean(dparam.units),0)))
        mdl.fit(train_features, train_target,epochs=int(round(np.mean(dparam.epochs),0)), batch_size=int(round(np.mean(dparam.batch_size),0)))
    elif(typ==6):
        mdl=create_cnn_model(int(round(np.mean(dparam.filters),0)),int(round(np.mean(dparam.units),0)))
        mdl.fit(train_features.reshape(train_features.shape[0], train_features.shape[1],1), train_target, epochs=int(round(np.mean(dparam.epochs),0)), batch_size=int(round(np.mean(dparam.batch_size),0)), validation_data=(test_features.reshape(test_features.shape[0], test_features.shape[1],1), test_target), verbose=0, shuffle=False) 
    
    def test_validate_model(data2, data3, mdl):
        scaled_data2 = scaler.fit_transform(data2.values) # validation investing
        tst_2 = scaled_data2[:len(data2), :]
        tst_2_features, tst_2_target = tst_2[:, 1:11], tst_2[:, 0]
    
        scaled_data3 = scaler.fit_transform(data3.values)  # validation Bitfinex
        tst_3 = scaled_data3[:len(data3), :]
        tst_3_features, tst_3_target = tst_3[:, 1:11], tst_3[:, 0]
    
        # forcast close price using the test data
      
        
        x2=pd.DataFrame(test_features, columns=['Open','High','Low','Vol','Year','Weighted_price','SMA3','SMA10','SMA20','EMA10'])
        if (typ== 4):# GRU convert input data to 3D
            x2.insert(0,'Price',pd.DataFrame(mdl.predict(test_features.reshape(test_features.shape[0], 1, test_features.shape[1])), columns=['Price']).Price)
        elif(typ==6): # CNN convert inout data to 3D
            x2.insert(0,'Price',pd.DataFrame(mdl.predict(test_features.reshape(test_features.shape[0], test_features.shape[1],1)), columns=['Price']).Price)
        else:
            x2.insert(0,'Price',pd.DataFrame(mdl.predict(test_features), columns=['Price']).Price)
        x3=pd.DataFrame(train_features, columns=['Open','High','Low','Vol','Year','Weighted_price','SMA3','SMA10','SMA20','EMA10'])
        x3.insert(0,'Price',pd.DataFrame(train_target, columns=['Price']).Price)
        inv_yhat= scaler.inverse_transform(pd.concat([x3,x2], axis=0))
        inv_yhat=inv_yhat[n_train_length:,:]
        inv_yhat = inv_yhat[:,0]
    
        x2=pd.DataFrame(test_features, columns=['Open','High','Low','Vol','Year','Weighted_price','SMA3','SMA10','SMA20','EMA10'])
        x2.insert(0,'Price',pd.DataFrame(test_target, columns=['Price']).Price)
        x3=pd.DataFrame(train_features, columns=['Open','High','Low','Vol','Year','Weighted_price','SMA3','SMA10','SMA20','EMA10'])
        x3.insert(0,'Price',pd.DataFrame(train_target, columns=['Price']).Price)
        inv_y = scaler.inverse_transform(pd.concat([x3,x2], axis=0))
        inv_y =inv_y[n_train_length:,:]
        inv_y =inv_y[:,0]
        data1_prediction=pd.concat([pd.DataFrame(inv_y, columns=['target']),pd.DataFrame(np.round(inv_yhat,5), columns=['forcast'])], axis=1)
        
        # validate the model with the UK Investing dataset
        x2=pd.DataFrame(tst_2_features, columns=['Open','High','Low','Vol','Year','Weighted_price','SMA3','SMA10','SMA20','EMA10'])
        if (typ==4): # GRU convert input data to 3D
            x2.insert(0,'Price',pd.DataFrame(mdl.predict(tst_2_features.reshape(tst_2_features.shape[0], 1, tst_2_features.shape[1])), columns=['Price']).Price)
        elif(typ==6): # CNN convert inout data to 3D
            x2.insert(0,'Price',pd.DataFrame(mdl.predict(tst_2_features.reshape(tst_2_features.shape[0], tst_2_features.shape[1],1)), columns=['Price']).Price) 
        else:
            x2.insert(0,'Price',pd.DataFrame(mdl.predict(tst_2_features), columns=['Price']).Price)
        inv_val= scaler.inverse_transform(x2)
        inv_val=inv_val[:len(tst_2_features),:]
        inv_val = inv_val[:,0]
    
        x2=pd.DataFrame(tst_2_features, columns=['Open','High','Low','Vol','Year','Weighted_price','SMA3','SMA10','SMA20','EMA10'])
        x2.insert(0,'Price',pd.DataFrame(tst_2_target, columns=['Price']).Price)
        inv_vala = scaler.inverse_transform(x2)
        inv_vala =inv_vala[:len(tst_2_features),:]
        inv_vala =inv_vala[:,0]
        data2_prediction=pd.concat([pd.DataFrame(inv_vala, columns=['target']),pd.DataFrame(np.round(inv_val,5), columns=['forcast'])], axis=1)

        # validate the model with the Bitfinex dataset
        x2=pd.DataFrame(tst_3_features, columns=['Open','High','Low','Vol','Year','Weighted_price','SMA3','SMA10','SMA20','EMA10'])
        if (typ==4): # GRU convert input data to 3D
            x2.insert(0,'Price',pd.DataFrame(mdl.predict(tst_3_features.reshape(tst_3_features.shape[0], 1, tst_3_features.shape[1])), columns=['Price']).Price)
        elif(typ==6): # CNN convert inout data to 3D
            x2.insert(0,'Price',pd.DataFrame(mdl.predict(tst_3_features.reshape(tst_3_features.shape[0], tst_3_features.shape[1],1)), columns=['Price']).Price)
        else:
            x2.insert(0,'Price',pd.DataFrame(mdl.predict(tst_3_features), columns=['Price']).Price)
        inv_val= scaler.inverse_transform(x2)
        inv_val=inv_val[:len(tst_3_features),:]
        inv_val = inv_val[:,0]
    
        x2=pd.DataFrame(tst_3_features, columns=['Open','High','Low','Vol','Year','Weighted_price','SMA3','SMA10','SMA20','EMA10'])
        x2.insert(0,'Price',pd.DataFrame(tst_3_target, columns=['Price']).Price)
        inv_vala = scaler.inverse_transform(x2)
        inv_vala =inv_vala[:len(tst_3_features),:]
        inv_vala =inv_vala[:,0]
        data3_prediction=pd.concat([pd.DataFrame(inv_vala, columns=['target']),pd.DataFrame(np.round(inv_val,5), columns=['forcast'])], axis=1)
        return data1_prediction, data2_prediction, data3_prediction #return dataframes containing prediction results 
    return test_validate_model(data2,data3,mdl)

#------------------------------------------------------------------------------------------
# module genetic_parameters_tune - uses GA for hyperparameters tuning and models training  #
#------------------------------------------------------------------------------------------
def genetic_parameters_tune(yahoo_btc, yahoo_ltc, yahoo_eth, yahoo_bnb, yahoo_doge,yahoo_xlm): 
    '''
    Parameters
    ----------
    yahoo_btc : TYPE - DATAFRAME
        DESCRIPTION. A Yahoo finance crypto dataset for BTC with 10 columns, i.e., Price, Open, High, Low, Vol, year, w_Price, SMA3, SMA10,SMA20,EMA10
    yahoo_eth : TYPE
        DESCRIPTIONA - Yahoo finance crypto dataset for ETH with 10 columns, i.e., Price, Open, High, Low, Vol, year, w_Price, SMA3, SMA10,SMA20,EMA10.
    yahoo_bnb : TYPE
        DESCRIPTION -A Yahoo finance crypto dataset for BNB with 10 columns, i.e., Price, Open, High, Low, Vol, year, w_Price, SMA3, SMA10,SMA20,EMA10.
    yahoo_ltc : TYPE
        DESCRIPTION -A Yahoo finance crypto dataset for LTC with 10 columns, i.e., Price, Open, High, Low, Vol, year, w_Price, SMA3, SMA10,SMA20,EMA10.
    yahoo_xlm : TYPE
        DESCRIPTION -A Yahoo finance crypto dataset for XLM with 10 columns, i.e., Price, Open, High, Low, Vol, year, w_Price, SMA3, SMA10,SMA20,EMA10.
    yahoo_doge : TYPE
        DESCRIPTION -A Yahoo finance crypto dataset for DOGE with 10 columns, i.e., Price, Open, High, Low, Vol, year, w_Price, SMA3, SMA10,SMA20,EMA10.
    
    Returns
    -------
    res1 : TYPE - Dataframe 
        DESCRIPTION. Consists of observartions and predictions from all models (XGB, GBM,ADA, GRU, MLP, CNN) using the Yahoo Finance datasets, i.e., 
                     the column names are target, forcast
    res2 : TYPE
        DESCRIPTION - Consists of observartions and predictions from all models (XGB, GBM,ADA, GRU, MLP, CNN) using the UK Investing datasets, i.e., 
                     the column names are target, forcast.
    res3 : TYPE
        DESCRIPTION - Consists of observartions and predictions from all models (XGB, GBM,ADA, GRU, MLP, CNN) using the Bitfinex datasets, i.e., 
                     the column names are target, forcast.
    '''

    xgb_reg = xgb.XGBRegressor()
    gbm_reg = GradientBoostingRegressor()
    ada_reg = AdaBoostRegressor()
    model_gru = KerasRegressor(model=create_gru_model, units=18,  epochs=4, batch_size=32, verbose=0)
    model_mlp = KerasRegressor(model=create_mlp_model, units=18, epochs=4, batch_size=32, verbose=0)
    model_cnn = KerasRegressor(model=create_cnn_model, filters=8, units=8,  epochs=4, batch_size=32, verbose=0)

    xgb_dict,gbm_dict,ada_dict,mlp_dict,gru_dict,cnn_dict=({} for i in range(6))
    
    xgb_grid = {'max_depth': Integer(3, 6),'learning_rate': Continuous(0.8, 1.3),'n_estimators': Integer(60, 70)}
    gbm_grid = {'max_depth': Integer(3, 6),'learning_rate': Continuous(0.8, 1.3),'n_estimators': Integer(65, 75)}
    ada_grid = {'learning_rate': Continuous(0.8, 1.3),'n_estimators': Integer(65, 75),'loss': Categorical(['square'])}
    gru_grid = {'units': Integer(11, 14),'batch_size': Integer(42, 45), 'epochs': Integer(76, 84)}
    mlp_grid = {'units': Integer(14, 19),'batch_size': Integer(40, 50), 'epochs': Integer(80, 90)}
    cnn_grid = {'filters': Integer(9, 12),'units': Integer(10, 14), 'batch_size': Integer(44, 54), 'epochs': Integer(75, 85)}


    xgb_estimator = GASearchCV(estimator=xgb_reg,cv=3,scoring='neg_mean_squared_error',population_size=10,generations=5,param_grid=xgb_grid,n_jobs=-1,crossover_probability=0.9,mutation_probability=0.05,verbose=False,keep_top_k=1)
    gbm_estimator = GASearchCV(estimator=gbm_reg,cv=3,scoring='neg_mean_squared_error',population_size=10,generations=5,param_grid=gbm_grid,n_jobs=-1,crossover_probability=0.9,mutation_probability=0.05,verbose=False,keep_top_k=1)
    ada_estimator = GASearchCV(estimator=ada_reg,cv=3,scoring='neg_mean_squared_error',population_size=10,generations=5,param_grid=ada_grid,n_jobs=-1,crossover_probability=0.9,mutation_probability=0.05,verbose=False,keep_top_k=1)
    gru_estimator = GASearchCV(estimator=model_gru,cv=3,scoring='neg_mean_squared_error',population_size=10,generations=5,param_grid=gru_grid,n_jobs=-1,crossover_probability=0.9,mutation_probability=0.05,verbose=False,keep_top_k=1)
    mlp_estimator = GASearchCV(estimator=model_mlp,cv=3,scoring='neg_mean_squared_error',population_size=10,generations=5,param_grid=mlp_grid,n_jobs=-1,crossover_probability=0.9,mutation_probability=0.05,verbose=False,keep_top_k=1)
    cnn_estimator = GASearchCV(estimator=model_cnn,cv=3,scoring='neg_mean_squared_error',population_size=10,generations=5,param_grid=cnn_grid,n_jobs=-1,crossover_probability=0.9,mutation_probability=0.05,verbose=False,keep_top_k=1)
    '''
    if (scenario==1):
        print('Scenario A: Wait.... tuning parameters of all predictive models')
    elif(scenario==2):
        print('Scenario B: Wait.... tuning parameters of all predictive models')
    '''
    def tune_parameters(data):
        train,train_target,test,test_target=divide_data(data)   
        xgb_estimator.fit(train, train_target)
        gbm_estimator.fit(train, train_target)
        ada_estimator.fit(train, train_target)
        gru_estimator.fit(train.reshape(train.shape[0], 1, train.shape[1]), train_target)
        mlp_estimator.fit(train, train_target)
        cnn_estimator.fit(train.reshape(train.shape[0], train.shape[1],1), train_target)
    ## Store the best parameters used to fit models
        xgb_dict[(xgb_estimator.best_params_['max_depth'],xgb_estimator.best_params_['learning_rate'],xgb_estimator.best_params_['n_estimators'])] = {'MSE':round(mean_squared_error(test_target, xgb_estimator.predict(test)),4)} 
        gbm_dict[(gbm_estimator.best_params_['max_depth'],gbm_estimator.best_params_['learning_rate'],gbm_estimator.best_params_['n_estimators'])] = {'MSE':round(mean_squared_error(test_target, gbm_estimator.predict(test)),4)}
        ada_dict[(ada_estimator.best_params_['learning_rate'],ada_estimator.best_params_['n_estimators'],ada_estimator.best_params_['loss'])] = {'MSE':round(mean_squared_error(test_target, ada_estimator.predict(test)),4)}
        gru_dict[(gru_estimator.best_params_['units'],gru_estimator.best_params_['batch_size'],gru_estimator.best_params_['epochs'])] = {'MSE':round(mean_squared_error(test_target, gru_estimator.predict(test.reshape(test.shape[0], 1, test.shape[1]))),4)}
        mlp_dict[(mlp_estimator.best_params_['units'],mlp_estimator.best_params_['batch_size'],mlp_estimator.best_params_['epochs'])] = {'MSE':round(mean_squared_error(test_target, mlp_estimator.predict(test)),4)}
        cnn_dict[(cnn_estimator.best_params_['filters'],cnn_estimator.best_params_['units'],cnn_estimator.best_params_['batch_size'],cnn_estimator.best_params_['epochs'])] = {'MSE':round(mean_squared_error(test_target, cnn_estimator.predict(test.reshape(test.shape[0], test.shape[1],1))),4)}
        return
    
    tune_parameters(yahoo_btc) #BTC
    tune_parameters(yahoo_eth) #ETH
    tune_parameters(yahoo_bnb) #BNB
    tune_parameters(yahoo_ltc) #LTC
    tune_parameters(yahoo_xlm) #XLM
    tune_parameters(yahoo_doge) #DOGE
    
    #convert dictionary storing tuned (optimal) parameters of models to dataframes
    df_xgb = pd.DataFrame(xgb_dict).T.reset_index() 
    df_xgb.columns = ['max_depth','learning_rate','n_estimators','MSE']
    df_gbm = pd.DataFrame(gbm_dict).T.reset_index() 
    df_gbm.columns = ['max_depth','learning_rate','n_estimators','MSE']
    df_ada = pd.DataFrame(ada_dict).T.reset_index() 
    df_ada.columns = ['learning_rate','n_estimators','loss','MSE']
    df_gru = pd.DataFrame(gru_dict).T.reset_index() 
    df_gru.columns = ['units','batch_size','epochs','MSE']
    df_mlp = pd.DataFrame(mlp_dict).T.reset_index()
    df_mlp.columns = ['units','batch_size','epochs','MSE']
    df_cnn = pd.DataFrame(cnn_dict).T.reset_index()
    df_cnn.columns = ['filters','units','batch_size','epochs','MSE']
    # Store optimal parameters to the disk
    if (scenario==1):
        pd.concat([df_xgb, df_gbm, df_ada, df_gru, df_mlp, df_cnn], axis=1).to_csv('../results/tune_parameters.csv')    
    return df_xgb, df_gbm, df_ada, df_gru, df_mlp, df_cnn
   
def build_models(df_xgb, df_gbm, df_ada, df_gru, df_mlp, df_cnn):
        #--------------------------------------
        # Training and validating ADA, GBM, XGB, GRU, MLP, CNN models  =========(BTC - Yahoo finance,  UK Investing and Bitfinex)=============
        # forcast close price using the test and validation datasets
        yahoo_BTC_XGB,invest_BTC_XGB, bitfinex_BTC_XGB = create_model(yahoo_btc,invest_btc,bitfinex_btc,df_xgb,1)
        yahoo_BTC_GBM,invest_BTC_GBM, bitfinex_BTC_GBM = create_model(yahoo_btc,invest_btc,bitfinex_btc,df_gbm,2)
        yahoo_BTC_ADA,invest_BTC_ADA, bitfinex_BTC_ADA = create_model(yahoo_btc,invest_btc,bitfinex_btc,df_ada,3)
        yahoo_BTC_GRU,invest_BTC_GRU, bitfinex_BTC_GRU = create_model(yahoo_btc,invest_btc,bitfinex_btc,df_gru,4)
        yahoo_BTC_MLP,invest_BTC_MLP, bitfinex_BTC_MLP = create_model(yahoo_btc,invest_btc,bitfinex_btc,df_mlp,5)
        yahoo_BTC_CNN,invest_BTC_CNN, bitfinex_BTC_CNN = create_model(yahoo_btc,invest_btc,bitfinex_btc,df_cnn,6)

        # Training andvalidating ADA, GBM, XGB, GRU, MLP, CNN models  =========(ETH - Yahoo finance,  UK Investing and Bitfinex)=============
        # forcast close price using the test and validation datasets
        yahoo_ETH_XGB,invest_ETH_XGB, bitfinex_ETH_XGB = create_model(yahoo_eth,invest_eth,bitfinex_eth,df_xgb,1)
        yahoo_ETH_GBM,invest_ETH_GBM, bitfinex_ETH_GBM = create_model(yahoo_eth,invest_eth,bitfinex_eth,df_gbm,2)
        yahoo_ETH_ADA,invest_ETH_ADA, bitfinex_ETH_ADA = create_model(yahoo_eth,invest_eth,bitfinex_eth,df_ada,3)
        yahoo_ETH_GRU,invest_ETH_GRU, bitfinex_ETH_GRU = create_model(yahoo_eth,invest_eth,bitfinex_eth,df_gru,4)
        yahoo_ETH_MLP,invest_ETH_MLP, bitfinex_ETH_MLP = create_model(yahoo_eth,invest_eth,bitfinex_eth,df_mlp,5)
        yahoo_ETH_CNN,invest_ETH_CNN, bitfinex_ETH_CNN = create_model(yahoo_eth,invest_eth,bitfinex_eth,df_cnn,6)


        # Training and validating ADA, GBM, XGB, GRU, MLP, CNN models  =========(BNB - Yahoo finance,  UK Investing and Bitfinex)=============
        # forcast close price using the test and validation datasets
        yahoo_BNB_XGB,invest_BNB_XGB, bitfinex_BNB_XGB = create_model(yahoo_bnb,invest_bnb,bitfinex_bnb,df_xgb,1)
        yahoo_BNB_GBM,invest_BNB_GBM, bitfinex_BNB_GBM = create_model(yahoo_bnb,invest_bnb,bitfinex_bnb,df_gbm,2)
        yahoo_BNB_ADA,invest_BNB_ADA, bitfinex_BNB_ADA = create_model(yahoo_bnb,invest_bnb,bitfinex_bnb,df_ada,3)
        yahoo_BNB_GRU,invest_BNB_GRU, bitfinex_BNB_GRU = create_model(yahoo_bnb,invest_bnb,bitfinex_bnb,df_gru,4)
        yahoo_BNB_MLP,invest_BNB_MLP, bitfinex_BNB_MLP = create_model(yahoo_bnb,invest_bnb,bitfinex_bnb,df_mlp,5)
        yahoo_BNB_CNN,invest_BNB_CNN, bitfinex_BNB_CNN = create_model(yahoo_bnb,invest_bnb,bitfinex_bnb,df_cnn,6)

        # Training andvalidating ADA, GBM, XGB, GRU, MLP, CNN models  =========(LTC - Yahoo finance,  UK Investing and Bitfinex)=============
        # forcast close price using the test and validation datasets
        yahoo_LTC_XGB,invest_LTC_XGB, bitfinex_LTC_XGB = create_model(yahoo_ltc,invest_ltc,bitfinex_ltc,df_xgb,1)
        yahoo_LTC_GBM,invest_LTC_GBM, bitfinex_LTC_GBM = create_model(yahoo_ltc,invest_ltc,bitfinex_ltc,df_gbm,2)
        yahoo_LTC_ADA,invest_LTC_ADA, bitfinex_LTC_ADA = create_model(yahoo_ltc,invest_ltc,bitfinex_ltc,df_ada,3)
        yahoo_LTC_GRU,invest_LTC_GRU, bitfinex_LTC_GRU = create_model(yahoo_ltc,invest_ltc,bitfinex_ltc,df_gru,4)
        yahoo_LTC_MLP,invest_LTC_MLP, bitfinex_LTC_MLP = create_model(yahoo_ltc,invest_ltc,bitfinex_ltc,df_mlp,5)
        yahoo_LTC_CNN,invest_LTC_CNN, bitfinex_LTC_CNN = create_model(yahoo_ltc,invest_ltc,bitfinex_ltc,df_cnn,6)

        #Training and validating ADA, GBM, XGB, GRU, MLP, CNN models  =========(XLM - Yahoo finance,  UK Investing and Bitfinex)=============
        # forcast close price using the test and validation datasets
        yahoo_XLM_XGB,invest_XLM_XGB, bitfinex_XLM_XGB = create_model(yahoo_xlm,invest_xlm,bitfinex_xlm,df_xgb,1)
        yahoo_XLM_GBM,invest_XLM_GBM, bitfinex_XLM_GBM = create_model(yahoo_xlm,invest_xlm,bitfinex_xlm,df_gbm,2)
        yahoo_XLM_ADA,invest_XLM_ADA, bitfinex_XLM_ADA = create_model(yahoo_xlm,invest_xlm,bitfinex_xlm,df_ada,3)
        yahoo_XLM_GRU,invest_XLM_GRU, bitfinex_XLM_GRU = create_model(yahoo_xlm,invest_xlm,bitfinex_xlm,df_gru,4)
        yahoo_XLM_MLP,invest_XLM_MLP, bitfinex_XLM_MLP = create_model(yahoo_xlm,invest_xlm,bitfinex_xlm,df_mlp,5)
        yahoo_XLM_CNN,invest_XLM_CNN, bitfinex_XLM_CNN = create_model(yahoo_xlm,invest_xlm,bitfinex_xlm,df_cnn,6)


        # Training and validating ADA, GBM, XGB, GRU, MLP, CNN models  =========(DOGE - Yahoo finance,  UK Investing and Bitfinex)=============
        # forcast close price using the test and validation datasets
        yahoo_DOGE_XGB,invest_DOGE_XGB, bitfinex_DOGE_XGB = create_model(yahoo_doge,invest_doge,bitfinex_doge,df_xgb,1)
        yahoo_DOGE_GBM,invest_DOGE_GBM, bitfinex_DOGE_GBM = create_model(yahoo_doge,invest_doge,bitfinex_doge,df_gbm,2)
        yahoo_DOGE_ADA,invest_DOGE_ADA, bitfinex_DOGE_ADA = create_model(yahoo_doge,invest_doge,bitfinex_doge,df_ada,3)
        yahoo_DOGE_GRU,invest_DOGE_GRU, bitfinex_DOGE_GRU = create_model(yahoo_doge,invest_doge,bitfinex_doge,df_gru,4)
        yahoo_DOGE_MLP,invest_DOGE_MLP, bitfinex_DOGE_MLP = create_model(yahoo_doge,invest_doge,bitfinex_doge,df_mlp,5)
        yahoo_DOGE_CNN,invest_DOGE_CNN, bitfinex_DOGE_CNN = create_model(yahoo_doge,invest_doge,bitfinex_doge,df_cnn,6)
    
    
        res1, res2, res3 = prepare_results(yahoo_BTC_XGB,yahoo_LTC_XGB,yahoo_ETH_XGB,yahoo_BNB_XGB,yahoo_DOGE_XGB,yahoo_XLM_XGB,
                    yahoo_BTC_GBM,yahoo_LTC_GBM,yahoo_ETH_GBM,yahoo_BNB_GBM,yahoo_DOGE_GBM,yahoo_XLM_GBM,
                    yahoo_BTC_ADA,yahoo_LTC_ADA,yahoo_ETH_ADA,yahoo_BNB_ADA,yahoo_DOGE_ADA,yahoo_XLM_ADA,
                    yahoo_BTC_GRU,yahoo_LTC_GRU,yahoo_ETH_GRU,yahoo_BNB_GRU,yahoo_DOGE_GRU,yahoo_XLM_GRU,
                    yahoo_BTC_MLP,yahoo_LTC_MLP,yahoo_ETH_MLP,yahoo_BNB_MLP,yahoo_DOGE_MLP,yahoo_XLM_MLP,
                    yahoo_BTC_CNN,yahoo_LTC_CNN,yahoo_ETH_CNN,yahoo_BNB_CNN,yahoo_DOGE_CNN,yahoo_XLM_CNN,
                    invest_BTC_XGB,invest_LTC_XGB,invest_ETH_XGB,invest_BNB_XGB,invest_DOGE_XGB,invest_XLM_XGB,
                    invest_BTC_GBM,invest_LTC_GBM,invest_ETH_GBM,invest_BNB_GBM,invest_DOGE_GBM,invest_XLM_GBM,
                    invest_BTC_ADA,invest_LTC_ADA,invest_ETH_ADA,invest_BNB_ADA,invest_DOGE_ADA,invest_XLM_ADA,
                    invest_BTC_GRU,invest_LTC_GRU,invest_ETH_GRU,invest_BNB_GRU,invest_DOGE_GRU,invest_XLM_GRU,
                    invest_BTC_MLP,invest_LTC_MLP,invest_ETH_MLP,invest_BNB_MLP,invest_DOGE_MLP,invest_XLM_MLP,
                    invest_BTC_CNN,invest_LTC_CNN,invest_ETH_CNN,invest_BNB_CNN,invest_DOGE_CNN,invest_XLM_CNN,
                    bitfinex_BTC_XGB,bitfinex_LTC_XGB,bitfinex_ETH_XGB,bitfinex_BNB_XGB,bitfinex_DOGE_XGB,bitfinex_XLM_XGB,
                    bitfinex_BTC_GBM,bitfinex_LTC_GBM,bitfinex_ETH_GBM,bitfinex_BNB_GBM,bitfinex_DOGE_GBM,bitfinex_XLM_GBM,
                    bitfinex_BTC_ADA,bitfinex_LTC_ADA,bitfinex_ETH_ADA,bitfinex_BNB_ADA,bitfinex_DOGE_ADA,bitfinex_XLM_ADA,
                    bitfinex_BTC_GRU,bitfinex_LTC_GRU,bitfinex_ETH_GRU,bitfinex_BNB_GRU,bitfinex_DOGE_GRU,bitfinex_XLM_GRU,
                    bitfinex_BTC_MLP,bitfinex_LTC_MLP,bitfinex_ETH_MLP,bitfinex_BNB_MLP,bitfinex_DOGE_MLP,bitfinex_XLM_MLP,
                    bitfinex_BTC_CNN,bitfinex_LTC_CNN,bitfinex_ETH_CNN,bitfinex_BNB_CNN,bitfinex_DOGE_CNN,bitfinex_XLM_CNN)
        return res1, res2, res3 

#--------------------------------------------------------------------------------------
# module to load csv files                                                            #
#--------------------------------------------------------------------------------------
def load_files():
    # load cryptocurrency data from yahoo finance
    yahoo_btc = pd.read_csv('../data/BTC_yahoo.csv')
    yahoo_ltc = pd.read_csv('../data/LTC_yahoo.csv')
    yahoo_eth = pd.read_csv('../data/ETH_yahoo.csv')
    yahoo_bnb = pd.read_csv('../data/BNB_yahoo.csv')
    yahoo_doge = pd.read_csv('../data/DOGE_yahoo.csv')
    yahoo_xlm = pd.read_csv('../data/XLM_yahoo.csv')

    # add additional features, i.e., SMA3, SMA10 ,SMA20, EMA10
    yahoo_btc = add_features(yahoo_btc)
    yahoo_ltc = add_features(yahoo_ltc)
    yahoo_eth = add_features(yahoo_eth)
    yahoo_bnb = add_features(yahoo_bnb)
    yahoo_doge = add_features(yahoo_doge)
    yahoo_xlm = add_features(yahoo_xlm)

    # load cryptocurrency data from UK invest
    invest_btc = pd.read_csv('../data/BTC_Invest.csv')
    invest_ltc = pd.read_csv('../data/LTC_Invest.csv')
    invest_eth = pd.read_csv('../data/ETH_Invest.csv')
    invest_bnb = pd.read_csv('../data/BNB_Invest.csv')
    invest_doge = pd.read_csv('../data/Doge_Invest.csv')
    invest_xlm = pd.read_csv('../data/XLM_Invest.csv')

    # add additional features, i.e., SMA3, SMA10 ,SMA20, EMA10
    invest_btc = add_features(invest_btc)
    invest_ltc = add_features(invest_ltc)
    invest_eth = add_features(invest_eth)
    invest_bnb = add_features(invest_bnb)
    invest_doge = add_features(invest_doge)
    invest_xlm = add_features(invest_xlm)

    # load cryptocurrency data from Bitfinex
    bitfinex_btc = pd.read_csv('../data/BTC_Bitfinex.csv')
    bitfinex_ltc = pd.read_csv('../data/LTC_Bitfinex.csv')
    bitfinex_eth = pd.read_csv('../data/ETH_Bitfinex.csv')
    bitfinex_bnb = pd.read_csv('../data/BNB_Bitfinex.csv')
    bitfinex_doge = pd.read_csv('../data/Doge_Invest.csv')
    bitfinex_xlm = pd.read_csv('../data/XLM_Bitfinex.csv')

    # add additional features, i.e., SMA3, SMA10 ,SMA20, EMA10
    bitfinex_btc = add_features(bitfinex_btc)
    bitfinex_ltc = add_features(bitfinex_ltc)
    bitfinex_eth = add_features(bitfinex_eth)
    bitfinex_bnb = add_features(bitfinex_bnb)
    bitfinex_doge = add_features(bitfinex_doge)
    bitfinex_xlm = add_features(bitfinex_xlm)
    #
    return    yahoo_btc, yahoo_ltc, yahoo_eth, yahoo_bnb, yahoo_doge,yahoo_xlm,invest_btc, invest_ltc, invest_eth, invest_bnb, invest_doge, invest_xlm,bitfinex_btc,bitfinex_ltc,bitfinex_eth,bitfinex_bnb,bitfinex_doge,bitfinex_xlm


#

def main():
    st = time.time()
    global scenario,n_train_length,model_name,title_str
    global yahoo_btc, yahoo_ltc, yahoo_eth, yahoo_bnb, yahoo_doge,yahoo_xlm,invest_btc, invest_ltc, invest_eth, invest_bnb, invest_doge, invest_xlm,bitfinex_btc,bitfinex_ltc,bitfinex_eth,bitfinex_bnb,bitfinex_doge,bitfinex_xlm

    
    model_name=['ADAB', 'GBM', 'XGB', 'CNN', 'DFNN', 'GRU']
    title_str=['BTC', 'ETH', 'BNB', 'LTC', 'XLM', 'DOGE']
    yahoo_btc, yahoo_ltc, yahoo_eth, yahoo_bnb, yahoo_doge,yahoo_xlm,invest_btc, invest_ltc, invest_eth, invest_bnb, invest_doge, invest_xlm,bitfinex_btc,bitfinex_ltc,bitfinex_eth,bitfinex_bnb,bitfinex_doge,bitfinex_xlm= load_files()# read datasets, create and add additional columns
    
    scenario =1
    n_train_length=int(round(yahoo_btc.shape[0]*.8,0))
    df_xgb, df_gbm, df_ada, df_gru, df_mlp, df_cnn = genetic_parameters_tune(yahoo_btc, yahoo_eth,yahoo_bnb, yahoo_ltc, yahoo_xlm,yahoo_doge) #use GA for parameters tuning
     
    for k in range(1,3): # run the two scenarios
        if (k==2):
            scenario=2
            n_train_length=int(round(yahoo_btc.shape[0]*.86,0))
    
        #create models and compute performance evaluation and store results as a csv table
        yahoo_result, invest_result, bitfinex_result = build_models(df_xgb, df_gbm, df_ada, df_gru, df_mlp, df_cnn)# use tuned parameters for building models
        #plots graphs (prediction and residual plots), then store graphs as pNG files
    
    et = time.time()
    print('Execution time:',(et-st), 'seconds')
            
    return

if __name__ == '__main__':
    main() 


