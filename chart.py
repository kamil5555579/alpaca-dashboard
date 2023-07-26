import plotly.graph_objects as go
from dash import html, dcc
import plotly.express as px
import numpy as np
from plot import scatter_fig, histogram_fig


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


def create_chart(chart_type, title, df, n_clicks, selected_x, selected_y, run_number):

    if chart_type=='image-plot':
        img = px.imshow([[0]], color_continuous_scale='gray', title=title)
        if selected_x:
            data = df[selected_x][run_number]
            if data is not None and data.any() and not np.isnan(data).any():
                img = px.imshow(df[selected_x][run_number], title=title, zmin=0, zmax=np.max(df[selected_x][run_number]) / 3,  #
                                    color_continuous_scale='GnBu')

        new_element = html.Div(
            children=[html.Button('x', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button'),
                                 dcc.Graph(id={'type':'image-plot',
                                 'index':n_clicks}, className='graph', figure=img)],
            className='graph-div'
        )
        chart_dic={'type':'image-plot', 'index':n_clicks, 'title':title, 'x_val': selected_x}

    if chart_type=='histogram':
        if selected_x:
            fig = histogram_fig(df, selected_x, run_number, title)
        else:
            fig = go.Figure()
            fig.update_layout(title=title)

        new_element = html.Div(
            children=[html.Button('x', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button'),
                                 dcc.Graph(id={'type':'histogram',
                                 'index':n_clicks}, className='graph', figure=fig)],
            className='graph-div'
        )
        chart_dic={'type':'histogram', 'index':n_clicks, 'title':title, 'x_val': selected_x}

    if chart_type=='scatter-plot':
        if selected_x and selected_y:
            fig = scatter_fig(df, selected_x, selected_y, run_number, title)
        else:
            fig = go.Figure()
            fig.update_layout(title=title)

        new_element = html.Div(
            children=[html.Button('x', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button'),
                                 dcc.Graph(id={'type':'scatter-plot',
                                 'index':n_clicks}, className='graph', figure=fig)],
            className='graph-div'
        )
        chart_dic={'type':'scatter-plot', 'index':n_clicks, 'title':title, 'x_val': selected_x, 'y_val': selected_y}

    if chart_type=='number':
        if selected_x:
            number = df[selected_x][run_number]
        else:
            number = None

        new_element = html.Div(
            children=[html.Button('x', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button'),
                                 html.Div(id={'type':'number',
                                 'index':n_clicks}, className='graph', children=[title, ': ', number])],
            className='number-div'
        )
        chart_dic={'type':'number', 'index':n_clicks, 'title':title, 'x_val': selected_x}

    
    return new_element, chart_dic
    
