import csv
import numpy as np
from plotly.graph_objs import Scatter,Layout
import plotly
import plotly.offline as py
import plotly.graph_objs as go

#画箱型图
def plot_box(data_array, save_dir):
    data_num = data_array.shape[0]-1
    x = ['Dice']*data_num + ['Sensitivity']*data_num + ['Specificity']*data_num
    trace1 = go.Box(
    x = x,
    y = list(data_array[1:,1].astype(np.float32))+list(data_array[1:,2].astype(np.float32))+list(data_array[1:,3].astype(np.float32)),
    name = 'Whole Tumor',
    boxpoints = 'all',
    jitter = 0.25, # 点在x轴上的随机抖动距离
    pointpos = -1.2, # 点在y轴上的偏移距离
    marker = dict(
    color = 'green',
    line = dict(
        color = 'green',
        width = 0.1
        )
    )
    )
    trace2 = go.Box(
    x = x,
    y = list(data_array[1:,4].astype(np.float32))+list(data_array[1:,5].astype(np.float32))+list(data_array[1:,6].astype(np.float32)),
    name = 'Tumor Core',
    boxpoints = 'all',
    jitter = 0.25, # 点在x轴上的随机抖动距离
    pointpos = -1.2, # 点在y轴上的偏移距离
    marker = dict(
    color = 'blue',
    line = dict(
        color = 'blue',
        width = 0.1
        )
    )
    )
    trace3 = go.Box(
    x = x,
    y = list(data_array[1:,7].astype(np.float32))+list(data_array[1:,8].astype(np.float32))+list(data_array[1:,9].astype(np.float32)),
    name = 'Enhance Tumor',
    boxpoints = 'all',
    jitter = 0.25, # 点在x轴上的随机抖动距离
    pointpos = -1.2, # 点在y轴上的偏移距离
    marker = dict(
    color = 'red',
    line = dict(
        color = 'red',
        width = 0.1
        )
    )
    )
    plot_data = [trace1,trace2,trace3]
    layout = go.Layout(title = dict(text='Result on Brats19 Data', x=0.5,y=0.92),  legend=dict(x=0,y=1,orientation="h"),boxmode='group')
    fig = go.Figure(data=plot_data, layout=layout)
    py.plot(fig, filename='{}/result_plot.html'.format(save_dir))


if __name__ == '__main__':
    sFileName='./data/output/result.csv'
    save_dir = './data/output'
    data = []
    with open(sFileName,newline='',encoding='UTF-8') as csvfile:
        rows=csv.reader(csvfile)
        for row in rows:
            data.append(row)
    data_array = np.array(data)
    plot_box(data_array, save_dir)