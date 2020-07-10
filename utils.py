from scipy import stats
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.fftpack import fft
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import os
from xgboost import XGBClassifier

############################ [Sentiment Analysis start] ############################################
data = pd.read_csv('Sentiment_Clean_Tweet.csv')
data = data.replace(4, 1)
data = data.dropna()
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data['clean_tweet'], data['target'], random_state=0)
X_train = X_train.apply(lambda x: np.str_(x))
X_test = X_test.apply(lambda x: np.str_(x))
# Train and evaluate the model
vect = CountVectorizer().fit(X_train)
X_train_vectorized = vect.transform(X_train)
clfrNB = MultinomialNB(alpha=0.1)
clfrNB.fit(X_train_vectorized, y_train)
# tweet clener


def clean_tweet(tweet):
    ''' 
    Utility function to clean tweet text by removing links, special characters 
    using simple regex statements. 
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
# predict single value


def predictsenti(text):
    text = clean_tweet(text)
    pre = clfrNB.predict(vect.transform([text]))
    if pre[0] == 0:
        return 0
    else:
        return 1
# predict bulk data


def sentipredictbulk(df):
    clean = []
    for parsed_tweet in df['tweet']:
        clean.append(clean_tweet(parsed_tweet))
    result = clfrNB.predict(vect.transform(clean))
    df['target'] = result
    df = df.replace(1, "Normal")
    df = df.replace(0, "Depressive")
    df.to_csv(r'static/senti_result.csv', index=False, header=True)
    return df
############################ [Sentiment Analysis End] ############################################

############################ [Actigraph Analysis start] ############################################


model = pickle.load(open('model_XG', 'rb'))


def SS(df):
    X = df['activity'].values
    X = X.reshape((len(X), 1))
    X = StandardScaler().fit_transform(X)
    df['activity'] = X
    return(df)


def FetureExtraction(df):
    fdf = df
    tdf = df
    # frq data
    ff_df = fft(fdf['activity'].values)
    fdf["activity"] = ff_df.real
    fdf = SS(fdf)
    fdf = fdf['activity'].resample('60T')
    # Frq Features list
    fmean = fdf.mean()
    fstd = fdf.std()
    fkurtosis = fdf.apply(pd.DataFrame.kurt)
    fskewness = fdf.apply(pd.DataFrame.skew)
    fcofvar = fstd/fmean
    finvcofvar = fmean/fstd
    # temporal data
    tdf = SS(tdf)
    tdf = tdf['activity'].resample('60T')
    # Temporal Features list
    mean = tdf.mean()
    std = tdf.std()
    skewness = tdf.apply(pd.DataFrame.skew)
    quantile1 = tdf.quantile(q=0.01)
    invcofvar = mean/std
    IQR = tdf.apply(stats.iqr, interpolation='midpoint')
    # data merging
    frames = [mean, skewness, quantile1, invcofvar, IQR,
              fstd, fkurtosis, fskewness, fcofvar, finvcofvar]
    final_data = pd.concat(frames, axis=1, sort=False)
    final_data.columns = ['mean', 'skewness', 'quantile1', 'invcofvar',
                          'IQR', 'fstd', 'fkurtosis', 'fskewness', 'fcofvar', 'finvcofvar']
    final_data = final_data.reset_index(drop=True)
    return(final_data)


def predictActigraph(df):
    df = df.head(60)
    df = FetureExtraction(df)
    #df = df.fillna(0)
    print(df)
    pre = model.predict(df.values)
    if pre[0] == 1:
        return 1
    else:
        return 0

# predict bulk data


def actibulkpredic(file):
    files = []
    result = []
    for filename in file:
        files.append(os.path.basename(filename))
        df = pd.read_csv(filename, parse_dates=[
                         "timestamp"], index_col="timestamp")
        df = df.head(60)
        r = predictActigraph(df)
        result.append(r)
    df = pd.DataFrame(list(zip(files, result)), columns=[
                      'File_Name', 'Predication_Class'])
    df = df.replace(0, "Normal")
    df = df.replace(1, "Depressive")
    df.to_csv(r'static/acti_result.csv', index=False, header=True)
    return df

############################ [Actigraph Analysis End] ############################################