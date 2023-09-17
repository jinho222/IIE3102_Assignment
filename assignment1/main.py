import pandas as pd

df_2015 = pd.read_csv('./dataset/SURFACE_ASOS_108_DAY_2013_2013_2015.csv', encoding="cp949") # encoding issue. @see https://zephyrus1111.tistory.com/39
print(df_2015.dtypes)
print(df_2015.info())