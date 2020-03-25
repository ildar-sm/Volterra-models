import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from plotly.graph_objs import Scatter, Layout

# # plotly.offline.plot({
# #     "data_train": [Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
# #     "layout": Layout(title="hello world")
# # })
#
# trace1 = go.Scatter(
#     x=['00:15', '00:30', '28.04.2017 00:45', '01:00', '01:15', '01:30'],
#     y=[1.5, 1, 1.3, 0.7, 0.8, 0.9],
#     name='Line1',
#     line=dict(color='#00B5B5')
# )
# trace2 = go.Bar(
#     x=['00:15', '00:30', '28.04.2017 00:45', '01:00', '01:15', '01:30'],
#     y=[1, 0.5, 0.7, -1.2, 0.3, 0.4]
# )
#
# trace3 = go.Scatter(
#     x=['00:15', '00:30', '28.04.2017 00:45', '01:00', '01:15', '01:30'],
#     y=[11, 10.5, 10.7, -11.2, 10.3, 10.4]
# )
#
#
# data_train = [trace1, trace2, trace3]
# # py.offline.plot(data_train, filename='bar-line')
# plotly.offline.plot(data_train, image_width=1800, image_height=1800)
#
#
#
#


import pandas as pd

add_path = ''
add_legend = ''

# sem = 'SEM SI'
# sem = 'SEM Alaska'
sem = 'Якутии'

df = pd.read_csv('docs\\data_alaska'+add_path+'.csv', sep=';')
df = df[:43800]
# df = df[:8760]
# df = df[:720]
# df = df[:168]
# df = df[:24]

# df['Load'] = pd.to_numeric(df['Load'].str.replace(',', '.'), errors='coerce')
# df['Levelled'] = pd.to_numeric(df['Levelled'].str.replace(',', '.'), errors='coerce')
# df['Random Forest'] = pd.to_numeric(df['Random Forest'].str.replace(',', '.'), errors='coerce')
# df['Trees Gradient Boosting'] = pd.to_numeric(df['Trees Gradient Boosting'].str.replace(',', '.'), errors='coerce')
# df['SVM with Radial kernel'] = pd.to_numeric(df['SVM with Radial kernel'].str.replace(',', '.'), errors='coerce')
df['SoC, kW*h'] = pd.to_numeric(df['SoC, kW*h'].str.replace(',', '.'), errors='coerce')
df['SoC, %'] = np.round(pd.to_numeric(df['SoC, %'].str.replace(',', '.'), errors='coerce'), 4)*100
df['Charge/Discharge strategy, kW/h'] = pd.to_numeric(df['Charge/Discharge strategy, kW/h'].str.replace(',', '.'), errors='coerce')
df['Demand, kW'] = pd.to_numeric(df['Demand, kW'].str.replace(',', '.'), errors='coerce')
# df['Diesel, kW'] = pd.to_numeric(df['Diesel, kW'].str.replace(',', '.'), errors='coerce')


N = df['Time, h'].size
start = pd.to_datetime('04-01-2012 10:00')
dates = pd.date_range(start='26-04-2016 20:15', end='28-04-2016 07:45', freq='15min')
dates = df['Time, h']

def load_leveling(dates, df):
    load = go.Scatter(
        x=dates,
        y=df['Load'],
        name='Load without Batteries',
        # line=dict(color='#00B5B5'),
    )

    levelled = go.Scatter(
        x=dates,
        y=df['Levelled'],
        name='Levelled Load',
        # line=dict(color='#00B5B5'),
    )

    zero = go.Scatter(
        x=dates,
        y=np.zeros([N]),
        line=dict(color='black'),
        name=''
    )

    layout = go.Layout(
        title='Load Leveling for Data from '+sem,
        hovermode='closest',
        xaxis=dict(
            title='Time',
            # ticklen=5,
            zeroline=False,
            # gridwidth=2,
        ),
        yaxis=dict(
            title='Power, W',
            # ticklen=5,
            # gridwidth=2,
        ),
        # showlegend=False,
    )

    data = [load, levelled]#, zero]
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(figure_or_data=fig, filename='load-leveling.html')
    pass


def charge_discharge(dates, df):

    load = go.Bar(
        x=dates,
        y=df['Charge/Discharge strategy, kW/h'],
        name='Charge/Discharge strategy, kW/h',
        # line=dict(color='#00B5B5'),
    )

    layout = go.Layout(
        # title='Charge/Discharge Strategy for Load Leveling for Data from '+sem+add_legend,
        title='Функция изменения мощности АБ для реальных данных нагрузки из ' + sem + add_legend,
        hovermode='closest',
        xaxis=dict(
            # title='Time, h',
            title='Время, ч',
            # ticklen=5,
            zeroline=False,
            # gridwidth=2,
        ),
        yaxis=dict(
            # title='Power, kW',
            title='Мощность, кВт',
            # ticklen=5,
            # gridwidth=2,
        ),
        # showlegend=False,
    )

    data = [load]
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(figure_or_data=fig, filename='charge-discharge'+add_path+'.html')
    pass


def state_of_charge(dates, df):

    load = go.Bar(
        x=dates,
        y=df['SoC'],
        name='State of Charge',
        marker=dict(
            color='#00B5B5',
            # line=dict(color='#00B5B5', width=1.5)
        ),
    )

    layout = go.Layout(
        title='State of Charge for Batteries'+add_legend,
        hovermode='closest',
        xaxis=dict(
            title='Time',
            # ticklen=5,
            zeroline=False,
            # gridwidth=2,
        ),
        yaxis=dict(
            title='State of Charge, Wh',
            # ticklen=5,
            # gridwidth=2,
        ),
        showlegend=False,
    )

    data = [load]
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(figure_or_data=fig, filename='soc'+add_path+'.html')
    pass


def state_of_charge_percent(dates, df):

    load = go.Bar(
        x=dates,
        y=df['SoC, %'],
        name='State of Charge',
        # mode='markers',
        marker=dict(
            color='#00B5B5',
            # line=dict(color='#00B5B5', width=1.5)
        ),
    )

    layout = go.Layout(
        # title='State of Charge for Batteries'+add_legend,
        title='Уровень заряда аккумуляторных батарей' + add_legend,
        hovermode='closest',
        xaxis=dict(
            # title='Time, h',
            title='Время, ч',
            # ticklen=5,
            zeroline=False,
            # gridwidth=2,
        ),
        yaxis=dict(
            # title='State of Charge, %',
            title='SoC, %',
            # ticklen=5,
            # gridwidth=2,
        ),
        # showlegend=True,
    )

    data = [load]
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(figure_or_data=fig, filename='soc_percent'+add_path+'.html')
    pass


def prognosis_load(dates, df):
    load = go.Scatter(
        x=dates,
        y=df['Demand, kW'],
        name='Load, kW',
        # line=dict(
        #     # color='#00B5B5',
        #     width=5
        # )
    )

    levelled = go.Scatter(
        x=dates,
        y=df['Diesel, kW'],
        name='Diesel, kW',
        # mode='markers',
        # line=dict(
        #     # color='#00B5B5',
        #     width=5
        # )
    )

    zero = go.Scatter(
        x=dates,
        y=np.zeros([N]),
        line=dict(color='black'),
        name=''
    )

    layout = go.Layout(
        title='Load Leveling with Prognosis for Data from '+sem,
        hovermode='closest',
        xaxis=dict(
            title='Time',
            # ticklen=5,
            zeroline=False,
            # gridwidth=2,
        ),
        yaxis=dict(
            title='Power, W',
            # ticklen=5,
            # gridwidth=2,
        ),
        # showlegend=False,
    )

    data = [load]#, zero]
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(figure_or_data=fig, filename='prognosis_load.html')
    pass


# load_leveling(dates, df)
charge_discharge(dates, df)
# state_of_charge(dates, df)
state_of_charge_percent(dates, df)
# prognosis_load(dates, df)
