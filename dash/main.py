from collections import Counter
import math 

import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource, DataTable, NumberFormatter,
                          RangeTool, StringFormatter, TableColumn,
                          GraphRenderer, Oval, StaticLayoutProvider,
                          TextInput, Button, Paragraph)
from bokeh.palettes import Spectral11, Spectral8
from bokeh.plotting import figure
from bokeh.sampledata.autompg2 import autompg2 as mpg
from bokeh.sampledata.stocks import AAPL
from bokeh.transform import cumsum

# Timeseries

dates = np.array(AAPL['date'], dtype=np.datetime64)
source = ColumnDataSource(data=dict(date=dates, close=AAPL['adj_close']))

p = figure(plot_height=160, tools="", toolbar_location=None, #name="line",
           x_axis_type="datetime", x_range=(dates[1500], dates[2500]), sizing_mode="scale_width")

p.line('date', 'close', source=source, line_width=2, alpha=0.7)
p.yaxis.axis_label = 'Densidad'
p.background_fill_color="#f5f5f5"
p.grid.grid_line_color="white"

select = figure(plot_height=50, plot_width=600, y_range=p.y_range,
                x_axis_type="datetime", y_axis_type=None,
                tools="", toolbar_location=None, sizing_mode="scale_width")

range_rool = RangeTool(x_range=p.x_range)
range_rool.overlay.fill_color = "navy"
range_rool.overlay.fill_alpha = 0.2

select.line('date', 'close', source=source)
select.ygrid.grid_line_color = None
select.add_tools(range_rool)
select.toolbar.active_multi = range_rool
select.background_fill_color="#f5f5f5"
select.grid.grid_line_color="white"
select.x_range.range_padding = 0.01

layout = column(p, select, sizing_mode="scale_width", name="line")

curdoc().add_root(layout)

# Donut chart

x = Counter({ 'United States': 157, 'United Kingdom': 93, 'Japan': 89, 'China': 63,
              'Germany': 44, 'India': 42, 'Italy': 40, 'Australia': 35, 'Brazil': 32,
              'France': 31, 'Taiwan': 31  })

data = pd.DataFrame.from_dict(dict(x), orient='index').reset_index().rename(index=str, columns={0:'value', 'index':'country'})
data['angle'] = data['value']/sum(x.values()) * 2*math.pi
data['color'] = Spectral11

region = figure(plot_height=350, toolbar_location=None, outline_line_color=None, sizing_mode="scale_both", name="region", x_range=(-0.4, 1))

region.annular_wedge(x=-0, y=1, inner_radius=0.2, outer_radius=0.32,
                  start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                  line_color="white", fill_color='color', legend_group='country', source=data)

region.axis.axis_label=None
region.axis.visible=False
region.grid.grid_line_color = None
region.legend.label_text_font_size = "0.7em"
region.legend.spacing = 1
region.legend.glyph_height = 15
region.legend.label_height = 15

curdoc().add_root(region)

# Bar chart

plats = ("IOS", "Android", "OSX", "Windows", "Other")
values = (35, 22, 13, 26, 4)
platform = figure(plot_height=350, toolbar_location=None, outline_line_color=None, sizing_mode="scale_both", name="platform",
                  y_range=list(reversed(plats)), x_axis_location="above")
platform.x_range.start = 0
platform.ygrid.grid_line_color = None
platform.axis.minor_tick_line_color = None
platform.outline_line_color = None

platform.hbar(left=0, right=values, y=plats, height=0.8)

curdoc().add_root(platform)


# Table

source = ColumnDataSource(data=mpg[:6])
columns = [
    TableColumn(field="cyl", title="Counts"),
    TableColumn(field="cty", title="Uniques",
                formatter=StringFormatter(text_align="center")),
    TableColumn(field="hwy", title="Rating",
                formatter=NumberFormatter(text_align="right")),
]
table = DataTable(source=source, columns=columns, height=210, width=330, name="table", sizing_mode="scale_both")

curdoc().add_root(table)



# networkx

N = 8
node_indices = list(range(N))
plot = figure(x_range=(-1.1,1.1), y_range=(-1.1,1.1), tools='', toolbar_location=None,
                plot_height=390, sizing_mode="scale_both", name="net1")
graph = GraphRenderer()
graph.node_renderer.data_source.add(node_indices, 'index')
graph.node_renderer.data_source.add(Spectral8, 'color')
#graph.node_renderer.glyph = Oval(height=0.1, width=0.2, fill_color='color')
graph.edge_renderer.data_source.data = dict(
    start=[0]*N,
    end=node_indices)
### start of layout code
circ = [i*2*math.pi/8 for i in node_indices]
x = [math.cos(i) for i in circ]
y = [math.sin(i) for i in circ]
graph_layout = dict(zip(node_indices, zip(x, y)))
graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
plot.renderers.append(graph)

curdoc().add_root(plot)


# studies selection
button = Button(label="Cargar")
input = TextInput(value="BokehStudio")
output = Paragraph()

# add a callback to a widget
def update_me():
    output.text = "Hello, " + input.value
button.on_click(update_me)
# create a layout for everything
layout1 = column(row(input, button), output, sizing_mode="scale_width", name="studies")
# add the layout to curdoc
curdoc().add_root(layout1)



# Setup

curdoc().title = "Dashboard Analisis Movimientos"
curdoc().template_variables['stats_names'] = ['long', 'secs', 'movement', 'calories']
curdoc().template_variables['stats'] = {
    'long'      : {'icon': 'play-circle-o', 'value': 11200, 'change':  4   , 'label': 'Duración (frames)'},
    'secs'      : {'icon': 'clock-o',       'value': 350,   'change':  1.2 , 'label': 'Tiempo (segs)'},
    'movement'  : {'icon': 'child',         'value': 5.6,   'change': -2.3 , 'label': 'Movim-x (m)'},
    'calories'  : {'icon': 'heart-o',       'value': 27300, 'change':  0.5 , 'label': 'Total Calorias'},
}
