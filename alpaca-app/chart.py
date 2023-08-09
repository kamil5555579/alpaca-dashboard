import plotly.graph_objects as go
from dash import html, dcc
import plotly.express as px
import numpy as np
from plot import scatter_fig, histogram_fig
import dash_bootstrap_components as dbc
import pandas as pd


def create_chart_options(type, x_val=None, y_val=None,
                          numerical_options=None, one_dimensional_options=None, image_options=None):

    if type=='histogram': 
        new_element = html.Div(
            children=[
                html.Div('Data:'), dcc.Dropdown(id='x-selection', options=one_dimensional_options, value=x_val, className='dropdown'),
                dcc.Dropdown(id='y-selection', options=numerical_options, value=y_val, className='dropdown', style={'display':'none'})
            ],
            className='options-div'
        )
    if type=='image-plot': 
        new_element = html.Div(
            children=[
                html.Div('Image:'), dcc.Dropdown(id='x-selection', options=image_options, value=x_val, className='dropdown'),
                dcc.Dropdown(id='y-selection', options=numerical_options, value=y_val, className='dropdown', style={'display':'none'})
            ],
            className='options-div'
        )
    if type=='scatter-plot': 
        new_element = html.Div(
            children=[
                html.Div('x:'), dcc.Dropdown(id='x-selection', options=one_dimensional_options, value=x_val, className='dropdown'),
                html.Div('y:'), dcc.Dropdown(id='y-selection', options=one_dimensional_options, value=y_val, className='dropdown')
            ],
            className='options-div'
        )
    if type=='number': 
        new_element = html.Div(
            children=[
                html.Div('Observable:'), dcc.Dropdown(id='x-selection', options=numerical_options, value=x_val, className='dropdown'),
                dcc.Dropdown(id='y-selection', options=numerical_options, value=y_val, className='dropdown', style={'display':'none'})
            ],
            className='options-div'
        )
    return new_element


def create_chart(chart_type, title, df, n_clicks, selected_x, selected_y):

    if chart_type=='image-plot':
        img = px.imshow([[0]], color_continuous_scale='gray', title=title)
        if selected_x:
            data = df[selected_x].iloc[0]
            if data is not None and data.any() and not np.isnan(data).any():
                img = px.imshow(df[selected_x].iloc[0], title=title, zmin=0, zmax=np.max(df[selected_x].iloc[0]) / 3,  #
                                    color_continuous_scale='GnBu')

        new_element = html.Div(
            children=[dbc.Button('X', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button', color='light'),
                                 dcc.Graph(id={'type':'image-plot',
                                 'index':n_clicks}, className='graph', figure=img)],
            className='graph-div'
        )
        chart_dic={'type':'image-plot', 'index':n_clicks, 'title':title, 'x_val': selected_x}

    if chart_type=='histogram':
        if selected_x:
            fig = histogram_fig(df, selected_x, title)
        else:
            fig = go.Figure()
            fig.update_layout(title=title)

        new_element = html.Div(
            children=[dbc.Button('X', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button', color='light'),
                                 dcc.Graph(id={'type':'histogram',
                                 'index':n_clicks}, className='graph', figure=fig)],
            className='graph-div'
        )
        chart_dic={'type':'histogram', 'index':n_clicks, 'title':title, 'x_val': selected_x}

    if chart_type=='scatter-plot':
        if selected_x and selected_y:
            fig = scatter_fig(df, selected_x, selected_y, title)
        else:
            fig = go.Figure()
            fig.update_layout(title=title)

        new_element = html.Div(
            children=[dbc.Button('X', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button', color='light'),
                                 dcc.Graph(id={'type':'scatter-plot',
                                 'index':n_clicks}, className='graph', figure=fig)],
            className='graph-div'
        )
        chart_dic={'type':'scatter-plot', 'index':n_clicks, 'title':title, 'x_val': selected_x, 'y_val': selected_y}

    if chart_type=='number':
        if selected_x:
            number = df[selected_x].iloc[0]
        else:
            number = None

        new_element = html.Div(
            children=[dbc.Button('X', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button', color='light'),
                                 html.Div(id={'type':'number',
                                 'index':n_clicks}, className='graph', children=[title, ': ', number])],
            className='number-div'
        )
        chart_dic={'type':'number', 'index':n_clicks, 'title':title, 'x_val': selected_x}

    
    return new_element, chart_dic
    
# convert df to json for dcc.Store

def serialize_df(df):
    # Convert NumPy arrays with nan to lists
    for column in df.columns:
        if df[column].dtype == object and df[column].apply(type).eq(np.ndarray).any():
            df[column] = df[column].apply(lambda arr: arr.tolist() if isinstance(arr, np.ndarray) else arr)

    # Convert DataFrame to JSON string
    json_str = df.to_json(orient='split')
    return json_str

# convert from json back to df

def deserialize_df(json_str):
    # Convert JSON string back to DataFrame
    df = pd.read_json(json_str, orient='split')

    # Convert lists back to NumPy arrays
    for column in df.columns:
        df[column] = df[column].apply(np.array)

    return df