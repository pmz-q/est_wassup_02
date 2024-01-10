def create_new_column_names(df, suffix, columns_no_change):
    '''Change column names by given suffix, keep columns_no_change, and return back the data'''
    df.columns = [
      col + suffix 
      if col not in columns_no_change
      else col
      for col in df.columns
    ]
    return df

def flatten_multi_index_columns(df):
    df.columns = ['_'.join([col for col in multi_col if len(col)>0]) 
                  for multi_col in df.columns]
    return df