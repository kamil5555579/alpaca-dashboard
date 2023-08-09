import pandas as pd
import sqlalchemy
import re
import numpy as np
import subprocess
import time
import psycopg2 as ps

# function that selects all the data from alpaca database

def fetch_data(first_run=None, last_run=None):

    engine=sqlalchemy.create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/alpaca") # db instead of localhost for docker
    with engine.begin() as conn:
        if first_run is not None and last_run is not None:
            query = sqlalchemy.text("""SELECT * FROM alpaca
                            WHERE "Run_Number Run_Number __value" BETWEEN :first_run AND :last_run
                            ORDER BY "Run_Number Run_Number __value" DESC
                            """)
            df = pd.read_sql_query(query, conn, params={'first_run': first_run, 'last_run': last_run})
        else:
            query = sqlalchemy.text("""SELECT * FROM alpaca
                            ORDER BY "Run_Number Run_Number __value" DESC
                            """)
            df = pd.read_sql_query(query, conn)

    df.set_index('Run_Number Run_Number __value', inplace=True)

    # need to convert arrays saved as bytes back to arrays
    byte_columns = [column[:-6] for column in df.columns if column.endswith("_shape") is True and df[column].dtype == object]
    shape_columns = [f'{column}_shape' for column in byte_columns]
    one_dimensional_columns=[]
    two_dimensional_columns=[]

    for shape_column in shape_columns:
        column_name = shape_column[:-6]  # Extract the original column name
        df[shape_column] = df[shape_column].apply(lambda x: tuple(map(int, re.findall(r'\d+', x))) if pd.notnull(x) else None)  # Extract the shape values
        shape_column_values = df[shape_column].dropna()[df[shape_column].dropna() != ()]
        if not shape_column_values.empty:
            length = len(shape_column_values.iloc[0])
        else:
            length = 0
        if length == 1:
            one_dimensional_columns.append(column_name)
        elif length ==2:
            two_dimensional_columns.append(column_name)

    for column in byte_columns:
        shape_column = f'{column}_shape'
        df[column] = df.apply(lambda row: np.frombuffer(row[column]).reshape(row[shape_column]) if pd.notnull(row[column]) and pd.notnull(row[shape_column]) else None, axis=1)

    numerical_columns = df.select_dtypes(include='number').columns

    column_dic={'numerical_columns':numerical_columns,
            'one_dimensional_columns': one_dimensional_columns,
            'two_dimensional_columns': two_dimensional_columns}

    return df, column_dic

def get_column_type(element, columns_dic):
    if element in columns_dic['numerical_columns']:
        return "numerical_columns"
    elif element in columns_dic['one_dimensional_columns']:
        return "one_dimensional_columns"
    elif element in columns_dic['two_dimensional_columns']:
        return "two_dimensional_columns"
    
def execute_alpaca_script(first_run, last_run):
    script_path = "/app/python-analyses/ALPACA/applications/alpaca_to_database.py"

    command = [
        "/app/python-analyses/venv/bin/python",
        script_path,
        "--first_run",
        str(first_run),
        "--last_run",
        str(last_run)
    ]

    #local command
    command = f'c:/programowanie/alpaca-dashboard/python-analyses/venv/Scripts/python.exe c:/programowanie/alpaca-dashboard/python-analyses/ALPACA/applications/alpaca_to_database.py --first_run {first_run} --last_run {last_run}'

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(result.stdout.decode())
        print(result.stderr.decode())
    except subprocess.CalledProcessError as e:
        # Handle subprocess error
        print(f'Error: Subprocess returned non-zero exit code: {e.returncode}')

def wait_for_database(table_name, max_attempts=10, retry_interval=5):
    attempts = 0
    conn_params = {
    "host": "localhost", # db or localhost
    "dbname": "alpaca",
    "user": "postgres",
    "password": "admin",
    "port": "5432"
}
    while attempts < max_attempts:
        try:
            conn = ps.connect(**conn_params)
            cursor = conn.cursor()
            cursor.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
            cursor.close()
            conn.close()
            return True
        except ps.OperationalError:
            print(f"Database not ready, attempt {attempts+1}/{max_attempts}")
            time.sleep(retry_interval)
            attempts += 1
        except ps.ProgrammingError:
            print(f"Database not ready, attempt {attempts+1}/{max_attempts}")
            time.sleep(retry_interval)
            attempts += 1
    return False
