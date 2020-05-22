import math 
import glob
import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (Button, ColumnDataSource, DataTable, Div, GraphRenderer,
                        NumberFormatter, Oval, Paragraph, RangeTool, Select, StaticLayoutProvider, 
                        StringFormatter, TableColumn, TextInput  )
from bokeh.palettes import Spectral11, Spectral8
from bokeh.plotting import figure


from collections import Counter
from bokeh.sampledata.autompg2 import autompg2 as mpg
from bokeh.sampledata.stocks import AAPL
from bokeh.transform import cumsum

StudiesPath="../videos/"
VideosPath="../videos/"

# Read list of files in default path
def readFilesList(filesType):
    if filesType=='studies':
        searchpath=StudiesPath+'*.csv'
    elif filesType=='videos':
        searchpath=StudiesPath+'*.mp4'
    else:
        return []
    studies=glob.glob(searchpath)
    for st in range(len(studies)):
        studies[st]=studies[st].split("/")[-1].split(".")[0]
    return studies

# Get filename.ext and returns filename
def pathToName(path = "out.mp4"):
    x = path.split("/")
    return x[-1].split(".")[0]

# Converts coords in string to python tuples
def cleanStrCoords(coords):
    if isinstance(coords, str):
        res=coords[1:-1].split(",")
        coords=(int(res[0]) , int(res[1]))
    return coords

# Imports csv study
def importStudy(study_csv):
    videoCoords=pd.read_csv(study_csv,sep=";",index_col=0) 
    for col in mimo.columns:
        videoCoords[col]=videoCoords[col].apply(cleanStrCoords)
    return videoCoords

class statKpi:
    def __init__(self, kpiName, icon, title):
        self.name=kpiName
        self.value=0
        self.icon=icon
        self.title=title
        self.bticon="circle-o"
        self.btcolor="brown"
        self.bttext="Tendencia"
        self.render=Div(text=str(self.value), name=kpiName, margin=(0,0,0,0), css_classes=["count","text-center"])
        self.render.tags=[self.icon, self.title, 0, self.bticon, self.btcolor, self.bttext]
    def setKpi(self, value):
        if value>self.value:
            self.bticon="sort-asc"
            self.btcolor="green"
        elif value<self.value:
            self.bticon="sort-desc"
            self.btcolor="green"
        else:
            self.bticon="dot-circle-o"
            self.btcolor="brown"
        self.render.tags=[self.icon, self.title, value-self.value, self.bticon, self.btcolor, self.bttext]
        self.value=value
        self.render.text=str(round(value,2))

# Stats

kpiLongVal=statKpi('kpi_long', 'clock-o', 'Duración(frames)')
kpiMovementVal=statKpi('kpi_mov', 'child', 'Movim-x (m)')
kpiCaloriesVal=statKpi('kpi_cal', 'heart-o', 'Total Calorias')

curdoc().add_root(kpiLongVal.render)
curdoc().add_root(kpiMovementVal.render)
curdoc().add_root(kpiCaloriesVal.render)

curdoc().title = "Dashboard Analisis Movimientos"
curdoc().template_variables['stats_names'] = ['long', 'secs', 'movement', 'calories']
statsDict={
    'long'      : {'icon': 'play-circle-o', 'value': 11200, 'change':  4   , 'label': 'Duración (frames)'},
    'secs'      : {'icon': 'clock-o',       'value': 350,   'change':  0 , 'label': 'Tiempo (segs)'},
    'movement'  : {'icon': 'child',         'value': 5.6,   'change': -2.3 , 'label': 'Movim-x (m)'},
    'calories'  : {'icon': 'heart-o',       'value': 27300, 'change':  0.5 , 'label': 'Total Calorias'},
}
curdoc().template_variables['stats'] = statsDict

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
                plot_height=410, sizing_mode="scale_both", name="net1")
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
#input = TextInput(value="BokehStudio")
#Div(text=".", width=100)

buttonSave = Button(label="Guardar Estudio", button_type="success", disabled=True)
outputStatus = Paragraph(text="Seleccione estudio para continuar", width=80, css_classes=['text-success'])
selectStudy = Select(title='Cargar Estudio', value='', options=readFilesList('studies'))
selectVideo = Select(title='Abrir Video', value='', options=readFilesList('videos'))
# add a callback to a widget
def update_me():
    outputStatus.text = selectVideo.value + ".csv Saved"
    kpiLongVal.setKpi(73)
    kpiCaloriesVal.setKpi(340)
    kpiMovementVal.setKpi(1.34)
def openStudioCbk(attr, old, new):
    if attr=='value':
        outputStatus.text = "Opening Studio " + selectStudy.value+".csv"
def openVideoCbk(attr,old,new):
    buttonSave.disabled = False
    outputStatus.text = "Opening Video " + selectVideo.value+".mp4"
buttonSave.on_click(update_me)
selectStudy.on_change('value', openStudioCbk)
selectVideo.on_change('value', openVideoCbk)

# create a layout for everything
layout1 = row(selectStudy, selectVideo, column(outputStatus, buttonSave, width=120), 
        sizing_mode="scale_width", name="studies")
# add the layout to curdoc
curdoc().add_root(layout1)
# Setup


