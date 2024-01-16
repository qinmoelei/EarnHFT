import pandas as pd
import numpy as np
import os
def concat_result(df_name,data_path_list,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df_list=[]
    for df_path in data_path_list:
        df=pd.read_feather(os.path.join(df_path,df_name))
        df_list.append(df)
    all_df=pd.concat(df_list,axis=0)
    sorteall_dfd_df = all_df.sort_values(by='timestamp')
    all_df.reset_index(drop=True,inplace=True)
    all_df.to_feather(os.path.join(save_path,df_name))
    return all_df




############################################### new preprocess#########################################################


if __name__ == "__main__":
    dates_list=os.listdir("/data1/sunshuo/tardis_data_multi_level/preprocess/concat_clean/BTCUSDT")
    dates_list.sort()
    dates_list=[os.path.join("/data1/sunshuo/tardis_data_multi_level/preprocess/concat_clean/BTCUSDT",dates) for dates in dates_list]
    concat_result(df_name="orderbook.feather",data_path_list=dates_list,save_path="preprocess/concat_clean/BTCUSDT/2021-06-01-2023-11-30")
    concat_result(df_name="trade_minitue.feather",data_path_list=dates_list,save_path="preprocess/concat_clean/BTCUSDT/2021-06-01-2023-11-30")
    concat_result(df_name="trade_second.feather",data_path_list=dates_list,save_path="preprocess/concat_clean/BTCUSDT/2021-06-01-2023-11-30")