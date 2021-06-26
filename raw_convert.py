import pandas as pd
import numpy as np
import os # doesnt have to be installed with pip
import json # doesnt have to be installed with pip

def json_to_col(dataframe, col):
    '''
    Takes column 'col' of a data frame and converts its content from a json format into tabular form and joins the newly created columns to the existing data frame. The one column with the json data ges dropped.
    '''
    dfj = pd.DataFrame(json.loads(dataframe[col].iloc[0]))
    for i in range(1, dataframe.shape[0]):
        try:
            dfj = dfj.append( pd.DataFrame(json.loads(dataframe[col].iloc[i])) )
        except:
            dfj = dfj.append(pd.Series(), ignore_index=True)
    dataframe = dataframe.reset_index(drop=True).join(dfj.reset_index(drop=True), rsuffix='_'+col)
    dataframe = dataframe.drop(col, axis=1)
    return dataframe



def json_cols(dataframe, col_list):
    '''
    Applies the json_to_col function for all columns that are specified in col_list.
    '''
    for i in range(len(col_list)):
        dataframe = json_to_col(dataframe, col_list[i])
        print('Converted column:', col_list[i])
    return dataframe


def mult_csv_w_json_to_one_df(directory: str, col_list, axis=0):
    '''
    Searches a directory (absolute or relative path) for csv files (they need to have the csv ending, comma as delimiter).
    Returns a dataframe of concatenated tables. Axis as in Pandas.
    Dependencies: pandas and os
    '''
    first = True
    for filename in os.listdir(directory):
        if first == True:
            first = False
            df = pd.read_csv(os.path.join(directory, filename))
            df = json_cols(df, col_list)

        else:
            if filename.endswith(".csv"):
                file_directory = os.path.join(directory, filename)
                df_next = pd.read_csv(file_directory)
                df_next = json_cols(df_next, col_list)
                df = pd.concat([df,df_next], axis=axis)
                
        print('Imported and preprocessed file:', filename)
        
    return df


df_all = mult_csv_w_json_to_one_df('data/', ['category', 'creator', 'location', 'profile'])

print('Writing the dataframe to a file')
df_all.to_csv('Kickstarter_preprocessed.csv')
