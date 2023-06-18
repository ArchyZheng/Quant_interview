# this file is created for get the mean and std for the four data IC, IF, IH, IM
# %%
import pandas as pd
import os


def get_mean_and_std(data_source: str, data_filename_list: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """

    :param data_source: the dictionary of data
    :param data_filename_list: the list of name of data, which should be same columns
    :return the mean and std of each column
    """

    dfs = []

    for data_filename in data_filename_list:
        datafile = os.path.join(data_source, data_filename)
        dfs.append(pd.read_csv(datafile))

    combined_df = pd.concat(dfs)
    multi_indexed_df = combined_df.set_index(['datetime', 'dominant_id'])
    multi_indexed_df.sort_index(inplace=True)
    feature_name_list = ['open', 'close', 'high', 'low', 'volume']
    df_mean = multi_indexed_df.groupby(level=[0])[feature_name_list].mean()
    df_std = multi_indexed_df.groupby(level=[0])[feature_name_list].std()

    return df_mean, df_std
