"""

data source:
 - [기상청 기상자료개방포털](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36&tabNo=1)

"""

import pandas as pd
from matplotlib import pyplot as plt

DAY = {
    'ko': '일시',
    'en': 'day'
}

AVERAGE_TEMPERATURE = {
    'ko': '평균기온(°C)',
    'en': 'average_temperature'
}

LOWEST_TEMPERATURE = {
    'ko': '최저기온(°C)',
    'en': 'lowest_temperature'
}

HIGHEST_TEMPERATURE = {
    'ko':"최고기온(°C)",
    'en': 'highest_temperature'
}

DAILY_PERCIPITATION = {
    'ko': '일강수량(mm)',
    'en': 'daily_percipitation'
}

HIGHEST_WIND_SPEED = {
    'ko': '최대 풍속(m/s)',
    'en': 'highest_wind_speed'
}

AVERAGE_WIND_SPEED = {
    'ko': '평균 풍속(m/s)',
    'en': 'average_with_speed'
}

AVERAGE_RELATIVE_HUMIDITY = {
    'ko': '평균 상대습도(%)',
    'en': 'average_relative_humidity'
}

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

df = df.loc[:, [
    DAY['ko'],
    AVERAGE_TEMPERATURE['ko'],
    LOWEST_TEMPERATURE['ko'],
    HIGHEST_TEMPERATURE['ko'],
    DAILY_PERCIPITATION['ko'],
    HIGHEST_WIND_SPEED['ko'],
    AVERAGE_WIND_SPEED['ko'],
    AVERAGE_RELATIVE_HUMIDITY['ko']
]]

df.columns = [
    DAY['en'],
    AVERAGE_TEMPERATURE['en'],
    LOWEST_TEMPERATURE['en'],
    HIGHEST_TEMPERATURE['en'],
    DAILY_PERCIPITATION['en'],
    HIGHEST_WIND_SPEED['en'],
    AVERAGE_WIND_SPEED['en'],
    AVERAGE_RELATIVE_HUMIDITY['en']
]

print(df)

'''
TODO
- handle NaN
- predict use AR or MA model
'''


# axs = plt.gca()

# selected_df_2013.plot(kind='line', x=DAY['en'], y=AVERAGE_TEMPERATURE['en'], ax=axs)

# # plt.show()