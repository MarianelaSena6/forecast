import pandas as pd


# Preparación del dato
def clean_df(df):
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y/%m/%d')
    df = df.set_index('DATE')
    df = df.asfreq('D')
    df = df.sort_index()
    df = df.drop('CATEGORY', axis=1)
    return df


def delete_nulls(df):
    df_nulls = df[df['UNITS_SOLD'].isnull()]
    null_finish_date = df_nulls.index.max()
    df = df.loc[df.index > null_finish_date]
    return df
