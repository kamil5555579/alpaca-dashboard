import plotly.graph_objects as go
from dash import html, dcc
import plotly.express as px
import numpy as np
from plot import scatter_fig, histogram_fig, surface_fig


def create_chart_options(type, x_val=None, y_val=None, z_val=None, image=None,
                          numerical_variable_options=None, histogram_options=None, image_options=None):
    """if type=='3d-plot': 
        new_element = html.Div(
            children=[
                dcc.Dropdown(id='3d-x-selection', options=numerical_variable_options, value=x_val, className='dropdown'),
                dcc.Dropdown(id='3d-y-selection', options=numerical_variable_options, value=y_val, className='dropdown'),
                dcc.Dropdown(id='3d-z-selection', options=numerical_variable_options, value=z_val, className='dropdown')
            ],
            className='graph-div'
        )"""
    if type=='histogram': 
        new_element = html.Div(
            children=[
                dcc.Dropdown(id='x-selection', options=histogram_options, value=x_val, className='dropdown'),
                dcc.Dropdown(id='y-selection', options=numerical_variable_options, value=y_val, className='dropdown', style={'display':'none'})
            ],
            className='options-div'
        )
    if type=='image-plot': 
        new_element = html.Div(
            children=[
                dcc.Dropdown(id='x-selection', options=image_options, value=image, className='dropdown'),
                dcc.Dropdown(id='y-selection', options=numerical_variable_options, value=y_val, className='dropdown', style={'display':'none'})
            ],
            className='options-div'
        )
    if type=='scatter-plot': 
        new_element = html.Div(
            children=[
                dcc.Dropdown(id='x-selection', options=histogram_options, value=x_val, className='dropdown'),
                dcc.Dropdown(id='y-selection', options=histogram_options, value=y_val, className='dropdown')
            ],
            className='options-div'
        )
    return new_element


def create_chart(chart_type, title, df, n_clicks, selected_x, selected_y, run_number):

    if chart_type=='image-plot':
        img = None
        if selected_x:
            data = df[selected_x][run_number]
            if data is not None:
                if data.any():
                    img = px.imshow(df[selected_x][run_number], title=title, zmin=0, zmax=np.max(df[selected_x][run_number]) / 3,  #
                                    color_continuous_scale='GnBu')
        else:
            img = px.imshow([[0]], color_continuous_scale='gray')

        new_element = html.Div(
            children=[html.Button('x', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button'),
                                 dcc.Graph(id={'type':'image-plot',
                                 'index':n_clicks}, className='graph', figure=img)],
            className='graph-div'
        )
        chart_dic={'type':'image-plot', 'index':n_clicks, 'x_val': selected_x}

    if chart_type=='histogram':
        if selected_x:
            fig = histogram_fig(df, selected_x, run_number, title)
        else:
            fig = go.Figure()

        new_element = html.Div(
            children=[html.Button('x', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button'),
                                 dcc.Graph(id={'type':'histogram',
                                 'index':n_clicks}, className='graph', figure=fig)],
            className='graph-div'
        )
        chart_dic={'type':'histogram', 'index':n_clicks, 'x_val': selected_x}

    if chart_type=='scatter-plot':
        if selected_x and selected_y:
            fig = scatter_fig(df, selected_x, selected_y, run_number, title)
        else:
            fig = go.Figure()

        new_element = html.Div(
            children=[html.Button('x', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button'),
                                 dcc.Graph(id={'type':'scatter-plot',
                                 'index':n_clicks}, className='graph', figure=fig)],
            className='graph-div'
        )
        chart_dic={'type':'scatter-plot', 'index':n_clicks, 'x_val': selected_x, 'y_val': selected_y}
    
    return new_element, chart_dic
    
