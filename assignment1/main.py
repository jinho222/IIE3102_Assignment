import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from statsmodels.tsa.ar_model import AutoReg 

DAY = '일시'
AVERAGE_TEMPERATURE = '평균기온(°C)'
LOWEST_TEMPERATURE = '최저기온(°C)'
HIGHEST_TEMPERATURE = '최고기온(°C)'
# DAILY_PERCIPITATION = '일강수량(mm)' # 일강수량은 결측치가 많아서 사용하지 않는다. 
HIGHEST_WIND_SPEED = '최대 풍속(m/s)'
AVERAGE_WIND_SPEED = '평균 풍속(m/s)'
AVERAGE_RELATIVE_HUMIDITY = '평균 상대습도(%)'

def get_dataframe():
    """
    data source:
    - [기상청 기상자료개방포털](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36&tabNo=1)
    """
    ENCODING = 'cp949' # utf8 makes encoding issue. @see https://zephyrus1111.tistory.com/39
    df_2013 = pd.read_csv('./dataset/SURFACE_ASOS_108_DAY_2013_2013_2015.csv', encoding=ENCODING)
    df_2014 = pd.read_csv('./dataset/SURFACE_ASOS_108_DAY_2014_2014_2015.csv', encoding=ENCODING)
    df_2015 = pd.read_csv('./dataset/SURFACE_ASOS_108_DAY_2015_2015_2016.csv', encoding=ENCODING)
    df_2016 = pd.read_csv('./dataset/SURFACE_ASOS_108_DAY_2016_2016_2017.csv', encoding=ENCODING)
    df_2017 = pd.read_csv('./dataset/SURFACE_ASOS_108_DAY_2017_2017_2018.csv', encoding=ENCODING)
    df_2018 = pd.read_csv('./dataset/SURFACE_ASOS_108_DAY_2018_2018_2019.csv', encoding=ENCODING)
    df_2019 = pd.read_csv('./dataset/SURFACE_ASOS_108_DAY_2019_2019_2020.csv', encoding=ENCODING)
    df_2020 = pd.read_csv('./dataset/SURFACE_ASOS_108_DAY_2020_2020_2021.csv', encoding=ENCODING)
    df_2021 = pd.read_csv('./dataset/SURFACE_ASOS_108_DAY_2021_2021_2022.csv', encoding=ENCODING)
    df_2022 = pd.read_csv('./dataset/SURFACE_ASOS_108_DAY_2022_2022_2023.csv', encoding=ENCODING)

    df = pd.concat([
        df_2013,
        df_2014,
        df_2015,
        df_2016,
        df_2017,
        df_2018,
        df_2019,
        df_2020,
        df_2021,
        df_2022,
    ], axis=0)

    df = df[[
        DAY,
        AVERAGE_TEMPERATURE,
        LOWEST_TEMPERATURE,
        HIGHEST_TEMPERATURE,
        HIGHEST_WIND_SPEED,
        AVERAGE_WIND_SPEED,
        AVERAGE_RELATIVE_HUMIDITY
    ]]

    df = df.loc[df[DAY].isin(
        [
            '2013-09-27',
            '2014-09-27',
            '2015-09-27',
            '2016-09-27',
            '2017-09-27',
            '2018-09-27',
            '2019-09-27',
            '2020-09-27',
            '2021-09-27',
            '2022-09-27',
        ]
    )]

    return df

def predict(dataframe, target):
    '''
    @see https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/ 
    '''
    model = AutoReg(np.asarray(dataframe[target]), lags=1)
    model_fit = model.fit()

    predicted = model_fit.predict(start=len(dataframe), end=len(dataframe))
    return predicted

def app():
    df = get_dataframe()
    print('---- observed metrics ----')
    print(df)

    predicted = pd.DataFrame({
        DAY: '2023-09-27',
        AVERAGE_TEMPERATURE: predict(df, AVERAGE_TEMPERATURE),
        LOWEST_TEMPERATURE: predict(df, LOWEST_TEMPERATURE),
        HIGHEST_TEMPERATURE: predict(df, HIGHEST_TEMPERATURE),
        HIGHEST_WIND_SPEED: predict(df, HIGHEST_WIND_SPEED),
        AVERAGE_WIND_SPEED: predict(df, AVERAGE_WIND_SPEED),
        AVERAGE_RELATIVE_HUMIDITY: predict(df, AVERAGE_RELATIVE_HUMIDITY),
    })

    print('---- predicted ----')
    print(predicted)

if __name__ == '__main__':
    app()