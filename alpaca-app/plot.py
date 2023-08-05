import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# plotting functions taken from alpaca

def scatter_fig(df, x_col, y_col, run, title):
        
    if run not in df[x_col].dropna().index:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig
    
    if run not in df[y_col].dropna().index:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig
    # make arrays

    x = df[x_col][run]
    y = df[y_col][run]

    fig = dict({
            "data": [{"type": 'scattergl',
                      "x": x,
                      "y": y,
                      "mode": 'markers',  # lines, markers
                      "marker": {"color": '#FFFFFF', "symbol": 'circle', "size": 10, "opacity": 0.5,
                                 # default marker
                                 "line": {"color": 'Black', "width": 1.5}},
                      "name": y_col,
                      "showlegend": False}  # default: False
                     ],
            "layout": {"barmode": "stack", "title": {"text": title, "font": {"size": 20}},
                       "legend": {"bgcolor": '#FFFFFF', "bordercolor": '#ff0000', "font": {"size": 25},
                                  "orientation": 'v'},
                       "xaxis": {"title": {"text": x_col, "font": {"size": 15}},
                                 "tickfont": {"size": 15},
                                 "autorange": True, "fixedrange": False, "type": 'linear', "gridcolor": "black",
                                 "linecolor": "black", "linewidth": 4, "ticks": "inside", "tickwidth": 5,
                                 "nticks": 20,
                                 "ticklabelstep": 2, "ticklen": 10, "position": 0, "mirror": "all"},
                       "yaxis": {"title": {"text": y_col, "font": {"size": 15}},
                                 "tickfont": {"size": 15},
                                 "autorange": True, "fixedrange": False, "exponentformat": "power",
                                 "gridcolor": "black", "linecolor": "black", "linewidth": 4, "ticks": "inside",
                                 "tickwidth": 5, "nticks": 20, "ticklabelstep": 2, "ticklen": 10,
                                 "mirror": "allticks"}}
        })

    return fig

def surface_fig(df, x_col, y_col, z_col):

    # erase NaN values
    df = df[df[z_col].notnull()]

    # make arrays
    x = np.array(df[x_col])
    y = np.array(df[y_col])
    z = np.array(df[z_col])

    # make linear spaces
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)

    # make meshgrid
    X, Y = np.meshgrid(xi, yi)

    # add interpolation
    Z = griddata((x, y), z, (X, Y), method='linear')  # {‘linear’, ‘nearest’, ‘cubic’}

    # create the dictionary for the surface
    fig = dict({
            "data": [{"type": 'surface',
                      "x": X,
                      "y": Y,
                      "z": Z,
                      "connectgaps": True,
                      "opacity": 1,
                      # "colorscale": 'Blues',
                      "colorbar": {"title": {"text": "MCP integral" + '<br>' + "[arb. units]", "side": 'top'},
                                   "tickfont": {"size": 16}, 'lenmode': 'fraction', 'len': 0.6,
                                   'showexponent': 'none'},
                      "text": [x_col, y_col, z_col],
                      "contours_z": dict(show=True, usecolormap=True, highlightcolor="black", project_z=True)
                      }],

            "layout": {"scene": {
                "xaxis": {"title": {"text": x_col, "font": {"size": 10}}, "tickfont": {"size": 14},
                          "linewidth": 0, "ticks": 'outside'},
                "yaxis": {"title": {"text": y_col, "font": {"size": 10}}, "tickfont": {"size": 14},
                          "linewidth": 0, "ticks": 'outside'},
                # 'showexponent': 'last', 'exponentformat': 'none', "ticksuffix": '', range: [-16, 8]
                "zaxis": {'showexponent': 'none', "ticks": 'outside', "tickangle": 0,
                          "title": {"text": z_col, "font": {"size": 10}}, "tickfont": {"size": 14},
                          "linewidth": 0}},  # "title": self.title,
                       "margin": dict(l=20, r=20, t=20, b=20),
                       "legend": {"bgcolor": '#FFFFFF', "bordercolor": '#ff0000', "font": {"size": 25},
                                  "orientation": 'v'},
                       "showlegend": True}
        })
    return fig

def histogram_fig(df, x_col, run, title):

    if run not in df[x_col].dropna().index:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig
    
    # make arrays

    x = df[x_col][run]
    y=None

    fig = dict({
            "data": [{"type": 'histogram',
                      "x": x,
                      "y": y,
                      "histfunc": "count",
                      "name": run,
                      "orientation": "v",
                      "textfont": {"size": 45},
                      "xbins": {"size": 0.1}  # Scintillator time resolution: 0.0000001
                      }],
            "layout": {"xaxis": {"title": r"$t /  s$"},
                       "yaxis": {"title": "events / count"},
                       # "yaxis_range": [0, 300],
                       # "xaxis_range": [17, 33],
                       "title": {"text": title, "font": {"size": 20}},
                       "legend": {"bgcolor": '#FFFFFF', "bordercolor": '#ff0000', "font": {"size": 25},
                                  "orientation": 'v'},
                       "showlegend": False}
        })
    return fig