import math 
import glob
import shutil
import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (BoxSelectTool, Button, Circle, ColorBar, ColumnDataSource, DataTable, Div, 
                    EdgesAndLinkedNodes, GraphRenderer, HoverTool, LinearColorMapper,
                    MultiLine, NodesAndLinkedEdges,
                    NumberFormatter, Oval, Panel, Paragraph, RangeTool, Select, StaticLayoutProvider, 
                    StringFormatter, TableColumn, Tabs, TapTool, TextInput)
from bokeh.models.callbacks import CustomJS
from bokeh.palettes import Spectral, Spectral11, Spectral8, Paired
from bokeh.plotting import figure
from bokeh.transform import transform
from bokeh.util.hex import axial_to_cartesian



from skeleton_handlers import (calcMiddlePoint, calcDistance,
                               cleanStrCoords, importStudy, importSkeleton, 
                               genCV_Skeleton, genBokeh_pelvis, genBokeh_Skeleton, 
                               zoomCenterSmCoords, zoomCenterLgCoords)
from image_handlers import (bokeh_postProc, getVideoFrame, process_freq_heatmap)
from signal_handlers import (calcJointSignals, deriveSignal, calcfftPower,
                           filterJointSignal)


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


# Program  Globals and Constants
curdoc().title = "Dashboard Analisis Movimientos"
currFrame=30

# Opening Menu Functions
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
# Update Stats Functions
def updateStats():
    outputStatus.text = selectVideo.value + ".csv Saved"
    kpiLongVal.setKpi(73)
    kpiCaloriesVal.setKpi(340)
    kpiMovementVal.setKpi(1.34)
    updateGraphSkeleton(None)

buttonSave = Button(label="Guardar Estudio", button_type="success", disabled=True)
outputStatus = Paragraph(text="Seleccione estudio para continuar", width=80, css_classes=['text-success'])
selectStudy = Select(title='Cargar Estudio', value='', options=readFilesList('studies'))
selectVideo = Select(title='Abrir Video', value='', options=readFilesList('videos'))
#callbacks linking
buttonSave.on_click(updateStats)
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






# Data get
VideoSrc="./videos/mov3out.mp4" # for current photo
#currFrame=30 ya definida al inicio
graphTileTextColor="#7e685a" #brown silver?
mySkeleton=importSkeleton(csvSrc+".skl")
myStudy=importStudy(csvSrc)
Fs=30
Ts=1/Fs


#extract signals
arr=myStudy['neck']
dfX1, dfY1, dX1, dY1, dMod, dArg = calcJointSignals(arr)
arr=myStudy['right_wrist']
dfX_rw, dfY_rw, dX_rw, dY_rw, dMod_rw, dArg_rw = calcJointSignals(arr)
arr=myStudy['right_elbow']
dfX_re, dfY_re, dX_re, dY_re, dMod_re, dArg_re = calcJointSignals(arr)
arr=myStudy['right_shoulder']
dfX_rs, dfY_rs, dX_rs, dY_rs, dMod_rs, dArg_rs = calcJointSignals(arr)
arr=myStudy['left_wrist']
dfX_lw, dfY_lw, dX_lw, dY_lw, dMod_lw, dArg_lw = calcJointSignals(arr)
arr=myStudy['left_elbow']
dfX_le, dfY_le, dX_le, dY_le, dMod_le, dArg_le = calcJointSignals(arr)
arr=myStudy['left_shoulder']
dfX_ls, dfY_ls, dX_ls, dY_ls, dMod_ls, dArg_ls = calcJointSignals(arr)
arr=myStudy['right_hip']
dfX_rh, dfY_rh, dX_rh, dY_rh, dMod_rh, dArg_rh = calcJointSignals(arr)
arr=myStudy['right_knee']
dfX_rk, dfY_rk, dX_rk, dY_rk, dMod_rk, dArg_rk = calcJointSignals(arr)
arr=myStudy['right_foot']
dfX_rf, dfY_rf, dX_rf, dY_rf, dMod_rf, dArg_rf = calcJointSignals(arr)
arr=myStudy['left_hip']
dfX_lh, dfY_lh, dX_lh, dY_lh, dMod_lh, dArg_lh = calcJointSignals(arr)
arr=myStudy['left_knee']
dfX_lk, dfY_lk, dX_lk, dY_lk, dMod_lk, dArg_lk = calcJointSignals(arr)
arr=myStudy['left_foot']
dfX_lf, dfY_lf, dX_lf, dY_lf, dMod_lf, dArg_lf = calcJointSignals(arr)
#calc relative speeds
dRel_rw=np.subtract(dMod_rw, dMod_re)
dRel_re=np.subtract(dMod_re, dMod_rs)
dRel_rs=np.subtract(dMod_rs, dMod)
dRel_lw=np.subtract(dMod_lw, dMod_le)
dRel_le=np.subtract(dMod_le, dMod_ls)
dRel_ls=np.subtract(dMod_ls, dMod)
dRel_rf=np.subtract(dMod_rf, dMod_rk)
dRel_rk=np.subtract(dMod_rk, dMod_rh)
dRel_rh=np.subtract(dMod_rh, dMod)
dRel_lf=np.subtract(dMod_lf, dMod_lk)
dRel_lk=np.subtract(dMod_lk, dMod_lh)
dRel_lh=np.subtract(dMod_lh, dMod)
#calc global joint signals
d_rArm=np.add(np.abs(dRel_re), np.abs(dRel_rw))
d_lArm=np.add(np.abs(dRel_le), np.abs(dRel_lw))
d_rLeg=np.add(np.abs(dRel_rk), np.abs(dRel_rf))
d_lLeg=np.add(np.abs(dRel_lk), np.abs(dRel_lf))

myStudy['body_fx']=dfX1
myStudy['body_fy']=dfY1
myStudy['body_dx']=dX1
myStudy['body_dy']=dY1
myStudy['body_dmod']=dMod
myStudy['body_darg']=dArg
myStudy['dRel_rw']=dRel_rw
myStudy['dRel_re']=dRel_re
myStudy['dRel_rs']=dRel_rs
myStudy['dRel_lw']=dRel_lw
myStudy['dRel_le']=dRel_le
myStudy['dRel_ls']=dRel_ls
myStudy['dRel_rk']=dRel_rk
myStudy['dRel_rf']=dRel_rf
myStudy['dRel_rh']=dRel_rh
myStudy['dRel_lk']=dRel_lk
myStudy['dRel_lf']=dRel_lf
myStudy['dRel_lh']=dRel_lh

#skeleton data arrangement and indicators inserts

mySkeleton["hexCoordQ"]=[None,0,-1,-2,-3,1,2,2,-1,-2,-3,0,0,0,None, None, None, None, None]
mySkeleton["hexCoordR"]=[None,0, 0, 0, 1,0,0,1, 1, 2, 3,1,2,3,None, None, None, None, None]
mySkeleton["name_es"]=['', 'Pecho', 'Hombro Der.', 'Codo Der.', 'Muñeca Der.',
       'Hombro Izq.', 'Codo Izq.', 'Muñeca Izq.', 'Cadera Der.', 'Rodilla Der.',
       'Pie Der.', 'Cadera Izq.', 'Rodilla Izq.', 'Pie Izq.', '',
       '', '', '', '']
mySkeleton["displacement"]=[None, np.sum(np.abs(dMod)), np.sum(np.abs(dRel_rs)), np.sum(np.abs(dRel_re)), 
        np.sum(np.abs(dRel_rw)),np.sum(np.abs(dRel_ls)), np.sum(np.abs(dRel_le)), np.sum(np.abs(dRel_lw)), 
        np.sum(np.abs(dRel_rh)),np.sum(np.abs(dRel_rk)), np.sum(np.abs(dRel_rf)), np.sum(np.abs(dRel_lh)), 
        np.sum(np.abs(dRel_lk)), np.sum(np.abs(dRel_rf)), None, None, None, None, None]
mySkeleton["abs_dsplc"]=[None, np.sum(dMod), np.sum(dMod_rs), np.sum(dMod_re), np.sum(dMod_rw),
        np.sum(dMod_ls), np.sum(dMod_le), np.sum(dMod_lw), np.sum(dMod_rh),
        np.sum(dMod_rk), np.sum(dMod_rf), np.sum(dMod_lh), 
        np.sum(dMod_lk), np.sum(dMod_rf), None, None, None, None, None]
mySkeleton["max_speed"]=[None, np.max(np.abs(dMod)), np.max(np.abs(dRel_rs)), np.max(np.abs(dRel_re)), 
        np.max(np.abs(dRel_rw)),np.sum(np.max(dRel_ls)), np.max(np.abs(dRel_le)), np.max(np.abs(dRel_lw)), 
        np.max(np.abs(dRel_rh)),np.sum(np.max(dRel_rk)), np.max(np.abs(dRel_rf)), np.max(np.abs(dRel_lh)), 
        np.sum(np.max(dRel_lk)), np.max(np.abs(dRel_rf)), None, None, None, None, None]
mySkeleton["avg_speed"]=np.dot(mySkeleton["displacement"],(1/len(dMod)))

#create skeleton for datasource
skelViewData=mySkeleton[mySkeleton["name_es"]!=""]
skelViewData=skelViewData.drop(['SINGLEPOSE', 'baseColor'], axis=1)

#frequency data calc
N=len(dMod)

powF_body=calcfftPower(dMod, N)
powF_rE=calcfftPower(dRel_re, N)
powF_rW=calcfftPower(dRel_rw, N)
powF_lE=calcfftPower(dRel_le, N)
powF_lW=calcfftPower(dRel_lw, N)
powF_rK=calcfftPower(dRel_rk, N)
powF_rF=calcfftPower(dRel_rf, N)
powF_lK=calcfftPower(dRel_lk, N)
powF_lF=calcfftPower(dRel_lf, N)

x_freq=np.linspace(0.0, 1.0/(2.0*Ts), N//2)
freqTable=pd.DataFrame(powF_body, columns=['powF_body'])
freqTable['x_freq']=x_freq
freqTable['powF_rE']=powF_rE
freqTable['powF_rW']=powF_rE
freqTable['powF_lE']=powF_lE
freqTable['powF_lW']=powF_lW
freqTable['powF_rK']=powF_rK
freqTable['powF_rF']=powF_rF
freqTable['powF_lK']=powF_lK
freqTable['powF_lF']=powF_lF


#create datasources
tableSource = ColumnDataSource(skelViewData)
timeSeriesSource = ColumnDataSource(myStudy)
freqDataSource = ColumnDataSource(freqTable)





# Timeseries Global

rollRg=(myStudy.index.size//2-myStudy.index.size//10,myStudy.index.size//2+myStudy.index.size//10)

mainGraph = figure(plot_height=160, tools="", toolbar_location=None, #name="line",
           x_range=rollRg, sizing_mode="scale_width")

mainGraph.line('index', 'body_fx', source=timeSeriesSource, line_width=2, alpha=0.7)
mainGraph.yaxis.axis_label = 'vel (px/frame)'
mainGraph.background_fill_color="#f5f5f5"
mainGraph.grid.grid_line_color="white"

select = figure(plot_height=50, plot_width=600, y_range=mainGraph.y_range,
                #x_axis_type="datetime", y_axis_type=None,
                tools="", toolbar_location=None, sizing_mode="scale_width")

range_rool = RangeTool(x_range=mainGraph.x_range)
range_rool.overlay.fill_color = "navy"
range_rool.overlay.fill_alpha = 0.2

select.line('index', 'body_fx', source=timeSeriesSource)
select.ygrid.grid_line_color = None
select.add_tools(range_rool)
select.toolbar.active_multi = range_rool
select.background_fill_color="#f5f5f5"
select.grid.grid_line_color="white"
select.x_range.range_padding = 0.1
select.yaxis.visible=False

layout = column(mainGraph, select, sizing_mode="scale_width", name="timeseries_global")
curdoc().add_root(layout)




# Current Photo

frame=getVideoFrame(VideoSrc, currFrame)
frameRGBA = bokeh_postProc(frame)
h, w, c = frameRGBA.shape

xmin, ymin, xmax, ymax = zoomCenterSmCoords(h, w)
MyPhoto = figure(x_range=(xmin, xmax), y_range=(-ymax, ymin), name="currPhoto", 
                    tooltips=[("x, y", "$x{0.0} $y{0.0}"), ("val", "@image")],
                    tools='pan,wheel_zoom,box_zoom,reset,save', plot_width=390,plot_height=360, title=None)
MyPhoto.yaxis.visible = False
MyPhoto.image_rgba(image=[frameRGBA], x=0, y=-ymax, dw=w, dh=h)

def updatePhotoFrame(evt):
    outputStatus.text = "Fired! "+VideoSrc 
    frame=getVideoFrame(VideoSrc, currFrame)
    frameRGBA = bokeh_postProc(frame)
    MyPhoto.image_rgba(image=[frameRGBA], x=0, y=-ymax, dw=w, dh=h)
MyPhoto.on_event('reset', updatePhotoFrame)
curdoc().add_root(MyPhoto)






# Timesweries Detail charts

timeGraphTop = figure(plot_width=570, plot_height=270)
timeGraphTop.line(source=timeSeriesSource, x='index', y='body_dmod', line_width=2, 
       color="gray", legend_label=skelViewData.name_es['neck']) 
timeGraphTop.line(source=timeSeriesSource, x='index', y='dRel_re', line_width=2, 
       color=Spectral[5][1], legend_label=skelViewData.name_es['right_elbow']) 
timeGraphTop.line(source=timeSeriesSource, x='index', y='dRel_rw', line_width=2, 
       color=Spectral[5][4], legend_label=skelViewData.name_es['right_wrist'])
timeGraphTop.line(source=timeSeriesSource, x='index', y='dRel_le', line_width=2, 
       color=Spectral[5][0], legend_label=skelViewData.name_es['left_elbow'])
timeGraphTop.line(source=timeSeriesSource, x='index', y='dRel_lw', line_width=2, 
       color=Spectral[5][3], legend_label=skelViewData.name_es['left_wrist'])

timeGraphTop.legend.location = "top_right"
timeGraphTop.legend.click_policy="hide"

timeGraphBott = figure(plot_width=570, plot_height=270)

timeGraphBott.line(source=timeSeriesSource, x='index', y='body_dmod', line_width=2, 
       color="gray", legend_label=skelViewData.name_es['neck']) 
timeGraphBott.line(source=timeSeriesSource, x='index', y='dRel_rk', line_width=2, 
                   color=Spectral[5][1], legend_label=skelViewData.name_es['right_knee']) 
timeGraphBott.line(source=timeSeriesSource, x='index', y='dRel_rf', line_width=2, 
                   color=Spectral[5][4], legend_label=skelViewData.name_es['right_foot'])
timeGraphBott.line(source=timeSeriesSource, x='index', y='dRel_lk', line_width=2, 
                   color=Spectral[5][0], legend_label=skelViewData.name_es['left_knee'])
timeGraphBott.line(source=timeSeriesSource, x='index', y='dRel_lf', line_width=2, 
                   color=Spectral[5][3], legend_label=skelViewData.name_es['left_foot'])
timeGraphBott.legend.location = "top_right"
timeGraphBott.legend.click_policy="hide"

tabTimeTop = Panel(child=timeGraphTop, title="Miembros Superiores")
tabMTimeBtm = Panel(child=timeGraphBott, title="Miembros Inferiores")
tabsTs = Tabs(tabs=[ tabTimeTop, tabMTimeBtm ], name="timeseries_detail")
curdoc().add_root(tabsTs)



#frequency charts
colPalette=Spectral[5]
freqGraphTop = figure(plot_width=570, plot_height=270)
freqGraphTop.line('x_freq', 'powF_body', source=freqDataSource, line_width=2, 
        color=colPalette[2], legend_label="FFT Pecho.")
freqGraphTop.line('x_freq', 'powF_rE', source=freqDataSource, line_width=2, 
        color=colPalette[1], legend_label="FFT Codo Der.")
freqGraphTop.line('x_freq', 'powF_rW', source=freqDataSource, line_width=2, 
        color=colPalette[4], legend_label="FFT Muñeca Der.")
freqGraphTop.line('x_freq', 'powF_lE', source=freqDataSource, line_width=2, 
        color=colPalette[0], legend_label="FFT Codo Izq.")
freqGraphTop.line('x_freq', 'powF_lW', source=freqDataSource, line_width=2, 
        color=colPalette[3], legend_label="FFT Muñeca Izq")
freqGraphTop.legend.location = "top_right"
freqGraphTop.legend.click_policy="hide"

freqGraphBott = figure(plot_width=570, plot_height=270)
freqGraphBott.line('x_freq', 'powF_body', source=freqDataSource, line_width=2, 
        color=colPalette[2], legend_label="FFT Pecho.")
freqGraphBott.line('x_freq', 'powF_rK', source=freqDataSource, line_width=2, 
        color=colPalette[1], legend_label="FFT Rodilla Der.")
freqGraphBott.line('x_freq', 'powF_rF', source=freqDataSource, line_width=2, 
        color=colPalette[4], legend_label="FFT Pie Der.")
freqGraphBott.line('x_freq', 'powF_lK', source=freqDataSource, line_width=2, 
        color=colPalette[0], legend_label="FFT Rodilla Izq.")
freqGraphBott.line('x_freq', 'powF_lF', source=freqDataSource, line_width=2, 
        color=colPalette[3], legend_label="FFT Pie Izq.")
freqGraphBott.legend.location = "top_right"
freqGraphBott.legend.click_policy="hide"

tabFreqTop = Panel(child=freqGraphTop, title="Miembros Superiores")
tabFreqBtm = Panel(child=freqGraphBott, title="Miembros Inferiores")
tabsFs = Tabs(tabs=[ tabFreqTop, tabFreqBtm ], name="frequency_detail")
curdoc().add_root(tabsFs)



# heatmaps Graphs

#for initilization abs color palette Greens[5]
colMapper = LinearColorMapper(palette=Spectral11, low=skelViewData.abs_dsplc.min(), 
                           high=skelViewData.abs_dsplc.max())

bodyHM = figure(plot_width=540,plot_height=290, toolbar_location=None)
bodyHM.xaxis.visible = False
bodyHM.yaxis.visible = False
bodyHM.grid.visible = False

bodyHM.hex_tile(q='hexCoordQ', r='hexCoordR', size=1, source=tableSource, 
           fill_color=transform('abs_dsplc', colMapper),
           line_color="black", line_width=1, alpha=0.9)

x, y = axial_to_cartesian(skelViewData.hexCoordQ, skelViewData.hexCoordR, 1, "pointytop")
bodyHM.text(x, y, text=[str(coord) for coord in skelViewData.name_es],
       text_baseline="middle", text_align="center", text_color="black", text_font_size="10px")

color_bar = ColorBar(color_mapper=colMapper, #ticker=LogTicker(), BasicTicker
                     label_standoff=5, major_label_text_color="gray", border_line_color=None, location=(0,0))
bodyHM.add_layout(color_bar, 'right')


data1 = ("Mmm3", "Cosa2", "Nooo", "Para eso", "Claro")
values1 = (105, 202, 13, 68, 45)
platform1 = figure(plot_width=570,plot_height=300, toolbar_location=None, outline_line_color=None, sizing_mode="scale_both", 
        name="graph2x", y_range=list(reversed(data1)), x_axis_location="above")
platform1.x_range.start = 0
platform1.ygrid.grid_line_color = None
platform1.axis.minor_tick_line_color = None
platform1.outline_line_color = None
platform1.hbar(left=0, right=values1, y=data1, height=0.8)



maxF=len(powF_body)//4
Fmax=(Fs/2)/4
peak_ceil=3.5


fqImage_raw=np.matrix([powF_rF[0:maxF], powF_rK[0:maxF], powF_rW[0:maxF], 
             powF_rE[0:maxF], powF_body[0:maxF], powF_lE[0:maxF], 
             powF_lW[0:maxF], powF_lK[0:maxF], powF_lF[0:maxF]])
#truncate max peak to 4 pixel/sample   a[a > 3] np.where(a<3,0,1)
fqImage_raw=np.where(fqImage_raw>peak_ceil, peak_ceil, fqImage_raw)  
fqImage_out=process_freq_heatmap(fqImage_raw, peak_ceil)
colMapperFq = LinearColorMapper(palette="Spectral11", low=0, high=peak_ceil)

freqHeatmap = figure(plot_width=540, plot_height=290, toolbar_location=None,
           tooltips=[("freq", "$x hz"), ("value", "@image")])
freqHeatmap.x_range.range_padding = freqHeatmap.y_range.range_padding = 0
#np.matrix([dRel_rf, dRel_rk, dRel_rw, dRel_re, powF_body, dRel_le, dRel_lw, dRel_lk, dRel_lf])
ticks = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5]
freqHeatmap.yaxis[0].ticker = ticks
freqHeatmap.yaxis[0].major_label_overrides = {0.5: 'R-Foot', 1.5: 'R-Knee', 2.5: 'R-Wrist', 3.5: 'R-Elbow', 4.5: 'neck',
                                   5.5: 'L-Elbow', 6.5:'L-Wrist', 7.5: 'L-Knee', 8.5: 'L-Foot'}
freqHeatmap.ygrid[0].ticker = ticks
freqHeatmap.image(image=[fqImage_out], x=0, y=0, dw=Fmax, dh=9, palette="Spectral11", level="image")
freqHeatmap.grid.grid_line_width = 0.5

color_bar = ColorBar(color_mapper=colMapperFq, #ticker=LogTicker(), BasicTicker
                     label_standoff=5, major_label_text_color="gray", border_line_color=None, location=(0,0))
freqHeatmap.add_layout(color_bar, 'right')




tabHeatMapBody = Panel(child=bodyHM, title="Heatmap Recorridos")
tabHeatMapFreq = Panel(child=freqHeatmap, title="Heatmap Frecuencias")
tabsHMaps = Tabs(tabs=[ tabHeatMapBody, tabHeatMapFreq ], name="heatMaps")
curdoc().add_root(tabsHMaps)





# Network Graph    - h, w, csvSrc, graphTileTextColor, currFrame, mySkeleton, myStudy

xmin, ymin, xmax, ymax = zoomCenterSmCoords(h,w)
skPlot = figure(x_range=(0, w), y_range=(h, 0), name="netGraph", outline_line_color=None,
              tools='pan,wheel_zoom,box_zoom,reset,tap,box_select,hover', plot_width=570,plot_height=300, 
              title="Network Graph (Articulaciones)", tooltips=[("coord", "$x{0.0} $y{0.0}"), ("art", "$index")])
skPlot.xaxis.visible = False
skPlot.title.text_color = graphTileTextColor
skPlot.title.text_font_size="15px"
skPlot.title.align = "center"

myBkSkeleton, connxs = genBokeh_Skeleton(mySkeleton, myStudy.iloc[currFrame])
netGraph = GraphRenderer()
netGraph.selection_policy = NodesAndLinkedEdges()
netGraph.inspection_policy = EdgesAndLinkedNodes()
netGraph.node_renderer.data_source.add(myBkSkeleton.index, 'index')
netGraph.node_renderer.data_source.add(myBkSkeleton['color'], 'color')
netGraph.node_renderer.glyph = Circle(size=10, line_color='color', fill_color='color', fill_alpha=0.4)
netGraph.node_renderer.selection_glyph = Circle(size=12, fill_color='color',  fill_alpha=0.7)
netGraph.node_renderer.hover_glyph = Circle(size=12, fill_color='color')     

netGraph.edge_renderer.data_source.data = connxs
netGraph.edge_renderer.glyph = MultiLine(line_color="#c8c8c8", line_width=2)
netGraph.edge_renderer.selection_glyph = MultiLine(line_color='#777777', line_width=2)
netGraph.edge_renderer.hover_glyph = MultiLine(line_color='#888888', line_width=2)

graph_layout = dict(zip(myBkSkeleton.index,  myBkSkeleton["coord2d"]))
netGraph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
skPlot.renderers.append(netGraph)

def updateGraphSkeleton(evt): # evaluate in startup... vars needs to be defined though
    global currFrame, netGraph
    currFrame+=10
    outputStatus.text = "Fire!!!! " + str(currFrame)
    skCoords=genBokeh_pelvis(myStudy.iloc[currFrame])
    graph_layout = dict(zip(myBkSkeleton.index, skCoords))
    netGraph.edge_renderer.data_source.data = connxs
    netGraph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
skPlot.on_event('reset', updateGraphSkeleton)
curdoc().add_root(skPlot)




# Table


columns = [
        TableColumn(field="name_es", title="Articulación"),
        TableColumn(field="displacement", title="Movimiento", 
                    formatter=NumberFormatter(format="0.00", text_align="right")),
        TableColumn(field="abs_dsplc", title="Mov(abs)", 
                    formatter=NumberFormatter(format="0.00", text_align="right")),
        TableColumn(field="max_speed", title="Vel Max", 
                    formatter=NumberFormatter(format="0.00", text_align="right")),
        TableColumn(field="avg_speed", title="Vel Prom", 
                    formatter=NumberFormatter(format="0.00", text_align="right")),
    ]
data_table = DataTable(source=tableSource, name="dataTable_kpis", columns=columns, width=560, height=290)

curdoc().add_root(data_table)





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





