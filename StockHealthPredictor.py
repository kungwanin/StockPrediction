import numpy as np
import pandas as pd
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import  Dropout
from keras.models import model_from_json
from keras.models import load_model
from keras import regularizers

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
fmt = '$%.0f'
tick = mtick.FormatStrFormatter(fmt)

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class StockHealthPredictor:

    def __init__(self, stock_name, weight_location = None):

        self.sia = SentimentIntensityAnalyzer()
        self.stock_name = stock_name
        self.weight_location = weight_location
        
        if pickle_location == None:
            self.prep_data()
            self.train()

        else:
            self.prep_data()
            self.predict()

        


    def prep_data(self):
        
        #read from API
        data = requests.get("http://virtual.cmuvefdf4m.ap-south-1.elasticbeanstalk.com/stockinfo/" + self.stock_name
                  ).json()["data"]
        
        df = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        
        for dict1 in data:
            df.append({"Date":dict1["Date"],"Open":dict1["Open"], "High":dict1["High"], "Low":dict1["Low"], "Close":dict1["Close"], "Volume":dict1["Volume"]}, ignore_index=True)

        data = requests.get("http://virtual.cmuvefdf4m.ap-south-1.elasticbeanstalk.com/stocknews/" + self.stock_name
                  ).json()["data"]

        temp = pd.DataFrame(columns=["Date","Headline"])

        for dict1 in data:
            temp.append({"Date":dict1["Date"], "Headline":dict1["Headline"]}, ignore_index=True)

        df_merged = df.merge(temp,on='Date')
        
        df_merged['compund']=''
        df_merged['neg']=''
        df_merged['neu']=''
        df_merged['pos']=''


        for index,sentence in enumerate(df_merged['headlines']):
            ps=sia.polarity_scores(sentence)
            df_merged['compund'][index]=ps['compound']
            df_merged['neg'][index]=ps['neg']
            df_merged['neu'][index]=ps['neu']
            df_merged['pos'][index]=ps['pos']

        final_df = df_merged[['Date','compund','neg','neu','pos','Open','High','Low','Close','Volume']]

        self.final_df = final_df


    def train(self):

        percentage_of_data = 1.0
        data_to_use = int(percentage_of_data*(len(self.final_df)-1))

        # 80% of data will be of training
        train_end = int(data_to_use*0.8)

        total_data = len(self.final_df)
        print("total_data:", total_data)
        start = total_data - data_to_use
        
        # Currently doing prediction only for 1 step ahead
        steps_to_predict = 1

        #close, compund, neg, neu, pos, open, high, low, volume
        # Order -> 8,1,2,3,4,5,6,7,9
        yt = self.final_df.iloc[start:total_data,8] #close
        yt1 = self.final_df.iloc[start:total_data,1] #compund
        yt2 = self.final_df.iloc[start:total_data,2] #neg
        yt3 = self.final_df.iloc[start:total_data,3] #neu
        yt4 = self.final_df.iloc[start:total_data,4] #pos
        yt5 = self.final_df.iloc[start:total_data,5] #open
        yt6 = self.final_df.iloc[start:total_data,6] #high
        yt7 = self.final_df.iloc[start:total_data,7] #low
        vt = self.final_df.iloc[start:total_data,9] #volume

        #shift next day close and next day compund
        yt_ = yt.shift(-1) #shifted close
        yt1_ = yt1.shift(-1) #shifted compund

        #taking only: close, next_close, compund, next_compund, volume, open, high, low
        data = pd.concat([yt, yt_, yt1, yt1_, vt, yt5, yt6, yt7], axis=1)
        data.columns = ['yt', 'yt_', 'yt1', 'yt1_','vt', 'yt5', 'yt6', 'yt7']

        data = data.dropna()

        # target variable - closed price
        # after shifting
        y = data['yt_'] #next_close

        # close, compund, next_compund, volume, open, high, low   
        cols = ['yt', 'yt1', 'yt1_', 'vt', 'yt5', 'yt6', 'yt7']
        x = data[cols]

        scaler_x = preprocessing.MinMaxScaler (feature_range=(-1, 1))
        x = np.array(x).reshape((len(x) ,len(cols)))
        x = scaler_x.fit_transform(x)

        scaler_y = preprocessing.MinMaxScaler (feature_range=(-1, 1))
        y = np.array (y).reshape ((len( y), 1))
        y = scaler_y.fit_transform (y)

        X_train = x[0 : train_end,]
        X_test = x[train_end+1 : len(x),]    
        y_train = y[0 : train_end] 
        y_test = y[train_end+1 : len(y)]  

        X_train = X_train.reshape (X_train. shape + (1,)) 
        X_test = X_test.reshape(X_test.shape + (1,))

        batch_size = 32
        nb_epoch = 100
        neurons = 25
        dropout = 0.1

        seed = 2016
        np.random.seed(seed)

        model = Sequential ()
        model.add(LSTM(neurons, return_sequences=True, activation='tanh', inner_activation='hard_sigmoid', input_shape=(len(cols), 1)))
        model.add(Dropout(dropout))
        model.add(LSTM(neurons, return_sequences=True,  activation='tanh'))
        model.add(Dropout(dropout))
        model.add(LSTM(neurons, activation='tanh'))
        model.add(Dropout(dropout))

        model.add(Dense(activity_regularizer=regularizers.l1(0.00001), output_dim=1, activation='linear'))
        model.add(Activation('tanh'))

        print(model.summary())

        model.compile(loss='mean_squared_error' , optimizer='RMSprop')
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.2)

        score_train = model.evaluate(X_train, y_train, batch_size =1)
        score_test = model.evaluate(X_test, y_test, batch_size =1)

        model_json = model.to_json()
        with open(self.stock_name+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(stock_name+".h5")
        self.weight_location = stock_name+".h5"

        pred = model.predict(X_test) 

        prediction_data = pred[-1]     

        X_test = scaler_x.inverse_transform(np.array(X_test).reshape((len(X_test), len(cols))))
        plt.plot(pred, label="predictions")

        y_test = scaler_y.inverse_transform(np.array(y_test).reshape((len( y_test), 1)))
        plt.plot([row[0] for row in y_test], label="actual")

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)

        ax = plt.axes()
        ax.yaxis.set_major_formatter(tick)
        plt.savefig(stock_name+".png")


    def predict(self):
        x = pd.DataFrame(columns=['yt', 'yt_', 'yt1', 'yt1_','vt', 'yt5', 'yt6', 'yt7'])
        x.append({'yt':self.final_df["Close"][-1], 'yt_':self.final_df["Close"][-2], 'yt1':self.final_df["Compount"], 'yt1_':self.final_df["Close"][-3],'vt':self.final_df["Low"][-1], 'yt5':self.final_df["Open"][-1], 'yt6':self.final_df["High"][-1], 'yt7':self.final_df["Low"][-1]}, ignore_index=True)
        scaler_x = preprocessing.MinMaxScaler (feature_range=(-1, 1))
        x = np.array(x).reshape((len(x) ,len(cols)))
        x = scaler_x.fit_transform(x)
        X_test = x.reshape (x.shape + (1,)) 
        model = load_model(self.weight_location)
        pred = model.predict(X_test) 
        scaler_y = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        pred = scaler_y.inverse_transform(np.array(pred).reshape((len(pred), 1)))
        prediction_data = pred[-1]     
        if X_test[-1,0] > predicted_data:
            self.health = 0
        else:
            self.health = 1

        
        

