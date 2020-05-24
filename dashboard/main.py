import math 
import glob
import shutil
import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (Button, ColumnDataSource, DataTable, Div, GraphRenderer,
                        NumberFormatter, Oval, Paragraph, RangeTool, Select, StaticLayoutProvider, 
                        StringFormatter, TableColumn, TextInput  )
from bokeh.palettes import Spectral11, Spectral8
from bokeh.plotting import figure
from bokeh.models.callbacks import CustomJS


from skeleton_handlers import (calcMiddlePoint, calcDistance,
                               cleanStrCoords, importStudy, importSkeleton, 
                               genCV_Skeleton, genBokeh_Skeleton, 
                               zoomCenterSmCoords, zoomCenterLgCoords)
from image_handlers import (bokeh_postProc, getVideoFrame)



from collections import Counter
from bokeh.sampledata.autompg2 import autompg2 as mpg
from bokeh.sampledata.stocks import AAPL
from bokeh.transform import cumsum

StudiesPath="../videos/"
VideosPath="../videos/"

VideoSrc=""
VideoOut=""
VideoTmp="dashboard/static/tmpVideo.mp4"
csvSrc="./videos/mov3.csv"


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


# Opening Menus
curdoc().title = "Dashboard Analisis Movimientos"
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
    VideoSrc=VideosPath+selectVideo.value+'.mp4'
    csvSrc=VideosPath+selectVideo.value+'.csv'
    workingFile.text=selectVideo.value+'.mp4'
    VideoOut=VideosPath+selectVideo.value+'_out.mp4'
    VideoTmp = './dashboard/static/'+selectVideo.value+'.mp4'
    shutil.copyfile(VideoSrc, VideoTmp)

buttonSave = Button(label="Guardar Estudio", button_type="success", disabled=True)
outputStatus = Paragraph(text="Seleccione estudio para continuar", width=80, css_classes=['text-success'])
selectStudy = Select(title='Cargar Estudio', value='', options=readFilesList('studies'))
selectVideo = Select(title='Abrir Video', value='', options=readFilesList('videos'))
#callbacks linking
buttonSave.on_click(update_me)
selectStudy.on_change('value', openStudioCbk)
selectVideo.on_change('value', openVideoCbk)

# create a layout for opening menus
layoutOpening = row(selectStudy, selectVideo, column(outputStatus, buttonSave, width=120), 
        sizing_mode="scale_width", name="studies")
# add the layout to curdoc
curdoc().add_root(layoutOpening)
# End Opening Menus


# Stats
kpiLongVal=statKpi('kpi_long', 'clock-o', 'Duración(frames)')
kpiMovementVal=statKpi('kpi_mov', 'child', 'Movim-x (m)')
kpiCaloriesVal=statKpi('kpi_cal', 'heart-o', 'Total Calorias')
curdoc().add_root(kpiLongVal.render)
curdoc().add_root(kpiMovementVal.render)
curdoc().add_root(kpiCaloriesVal.render)


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



# networkx
h=720
w=1280
mySkeleton=importSkeleton(csvSrc+".skl")
myStudy=importStudy(csvSrc)
currFrame=0

xmin, ymin, xmax, ymax = zoomCenterSmCoords(h,w)
#x_range=(xmin, xmax), y_range=(ymax, ymin)
skPlot = figure(x_range=(0, w), y_range=(h, 0), name="netGraph",
              tools='pan,wheel_zoom,box_zoom,reset', plot_width=580,plot_height=300, title=None)
              # toolbar_location=None )
netGraph = GraphRenderer()
myBkSkeleton, connxs = genBokeh_Skeleton(mySkeleton, myStudy.iloc[currFrame])

netGraph.node_renderer.data_source.add(myBkSkeleton.index, 'index')
netGraph.node_renderer.data_source.add(myBkSkeleton['color'], 'color')
netGraph.node_renderer.glyph = Oval(height=18, width=18, fill_color='color')
netGraph.edge_renderer.data_source.data = connxs
graph_layout = dict(zip(myBkSkeleton.index,  myBkSkeleton["coord2d"]))
netGraph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
skPlot.renderers.append(netGraph)

curdoc().add_root(skPlot)




# Current Photo

frameNumber=56
videoFile="./videos/mov3out.mp4"


frame=getVideoFrame(videoFile, frameNumber)
frameRGBA = bokeh_postProc(frame)
h, w, c = frameRGBA.shape

xmin, ymin, xmax, ymax = zoomCenterSmCoords(h, w)
MyPhoto = figure(x_range=(xmin, xmax), y_range=(ymin, ymax), name="currPhoto", 
                    tooltips=[("x coord", "$x"), ("y coord", "720-$y"), ("value", "@image")],
                    tools='pan,wheel_zoom,box_zoom,reset', plot_width=390,plot_height=360, title=None)
MyPhoto.image_rgba(image=[frameRGBA], x=0, y=0, dw=w, dh=h)

curdoc().add_root(MyPhoto)




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

# Bar chart 2

data1 = ("Mmm3", "Cosa2", "Nooo", "Para eso", "Claro")
values1 = (105, 202, 13, 68, 45)
platform1 = figure(plot_height=350, toolbar_location=None, outline_line_color=None, sizing_mode="scale_both", 
        name="graph2x", y_range=list(reversed(data1)), x_axis_location="above")
platform1.x_range.start = 0
platform1.ygrid.grid_line_color = None
platform1.axis.minor_tick_line_color = None
platform1.outline_line_color = None

platform1.hbar(left=0, right=values1, y=data1, height=0.8)

#curdoc().add_root(platform1)

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





workingFile=Div(text="", width=100, name="workingFile", id="workingFile")
workingFile.tags=['./dashboard/static/']
curdoc().add_root(workingFile)

callback1 = CustomJS(args=dict(txtCtrl=workingFile), code="""
var video = document.getElementById('videoplayer');
video.src = txtCtrl.tags[0]+txtCtrl.text;
video.play();
// models passed as args are automagically available
""")
buttonUpdVideo = Button(label="⟳", name="btn_updvideo", button_type="success", 
    width=50, height=31, css_classes=['text-right'])
buttonUpdVideo.js_on_click(callback1)

curdoc().add_root(buttonUpdVideo)








