import pandas as pd
import sqlalchemy
import re
import numpy as np

# function that selects all the data from alpaca database

def fetch_data():

    engine=sqlalchemy.create_engine("postgresql+psycopg2://postgres:admin@localhost/alpaca")
    with engine.begin() as conn:
        query = sqlalchemy.text("""SELECT * FROM alpaca
                                ORDER BY "Run_Number Run_Number __value" DESC
                                """)
        df = pd.read_sql_query(query, conn)

    df.set_index('Run_Number Run_Number __value', inplace=True)

    # need to convert arrays saved as bytes back to arrays
    byte_columns = [column for column in df.columns if column.endswith("_shape") is False and df[column].dtype == object]
    shape_columns = [f'{column}_shape' for column in byte_columns]
    one_dimensional_columns=[]
    two_dimensional_columns=[]

    for shape_column in shape_columns:
        column_name = shape_column[:-6]  # Extract the original column name
        df[shape_column] = df[shape_column].apply(lambda x: tuple(map(int, re.findall(r'\d+', x))) if pd.notnull(x) else None)  # Extract the shape values
        lenght = len(df[shape_column].dropna()[df[shape_column].dropna() != ()].iloc[0])
        if lenght == 1:
            one_dimensional_columns.append(column_name)
        elif lenght ==2:
            two_dimensional_columns.append(column_name)

    for column in byte_columns:
        shape_column = f'{column}_shape'
        df[column] = df.apply(lambda row: np.frombuffer(row[column]).reshape(row[shape_column]) if pd.notnull(row[column]) and pd.notnull(row[shape_column]) else None, axis=1)

    numerical_columns = df.select_dtypes(include='number').columns

    return df, numerical_columns, one_dimensional_columns, two_dimensional_columns
