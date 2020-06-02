import math 
import glob
import shutil
import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (BoxSelectTool, Button, Circle, ColorBar, ColumnDataSource, DataTable, Div, 
                    EdgesAndLinkedNodes, FuncTickFormatter, GraphRenderer, HoverTool, 
                    LinearColorMapper, MultiLine, NodesAndLinkedEdges,
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

# Filter wrong frame selections
def correct_n_frame(frame):
    global studyLen
    frame=int(frame)
    if frame> studyLen:
        return studyLen
    elif frame<0:
        return 0
    else:
        return frame

# Return FFT data for windowed selection
def getFreqWindowData(wstart, wend):
    global dMod, d_rArm, d_lArm, d_rLeg, d_lLeg
    wstart=correct_n_frame(wstart)
    wend=correct_n_frame(wend)
    Nw=wend-wstart

    x_wfreq=np.linspace(0.0, 1.0/(2.0*Ts), Nw//2)
    pFw_body=calcfftPower(dMod[wstart:wend], Nw)
    pFw_rArm=calcfftPower(d_rArm[wstart:wend], Nw)
    pFw_lArm=calcfftPower(d_lArm[wstart:wend], Nw)
    pFw_rLeg=calcfftPower(d_rLeg[wstart:wend], Nw)
    pFw_lLeg=calcfftPower(d_lLeg[wstart:wend], Nw)

    if Nw>30:
        Nw=len(pFw_body)//3
    else:
        Nw=len(pFw_body)

    freqWindTable=None
    freqWindTable=pd.DataFrame(x_wfreq[0:Nw], columns=['x_wfreq'])
    freqWindTable['pFw_body']=pFw_body[0:Nw]
    freqWindTable['pFw_rArm']=pFw_rArm[0:Nw]
    freqWindTable['pFw_lArm']=pFw_lArm[0:Nw]
    freqWindTable['pFw_rLeg']=pFw_rLeg[0:Nw]
    freqWindTable['pFw_lLeg']=pFw_lLeg[0:Nw]
    return freqWindTable

# classes for view objects (stats)

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


# Data model definition

def calcJointSpeeds(myStudy):
    global dMod, dMod_rw, dMod_re, dMod_rs, dMod_lw, dMod_le, dMod_ls
    global dMod_rh, dMod_rk, dMod_rf, dMod_lh, dMod_lk, dMod_lf
    global dRel_rw, dRel_re, dRel_rs, dRel_lw, dRel_le, dRel_ls
    global dRel_rf, dRel_rk, dRel_rh, dRel_lf, dRel_lk, dRel_lh
    global d_rArm, d_lArm, d_rLeg, d_lLeg
    #extract signals
    dfX1, dfY1, dX1, dY1, dMod, dArg = calcJointSignals(myStudy['neck'])
    dfX_rw, dfY_rw, dX_rw, dY_rw, dMod_rw, dArg_rw = calcJointSignals(myStudy['right_wrist'])
    dfX_re, dfY_re, dX_re, dY_re, dMod_re, dArg_re = calcJointSignals(myStudy['right_elbow'])
    dfX_rs, dfY_rs, dX_rs, dY_rs, dMod_rs, dArg_rs = calcJointSignals(myStudy['right_shoulder'])
    dfX_lw, dfY_lw, dX_lw, dY_lw, dMod_lw, dArg_lw = calcJointSignals(myStudy['left_wrist'])
    dfX_le, dfY_le, dX_le, dY_le, dMod_le, dArg_le = calcJointSignals(myStudy['left_elbow'])
    dfX_ls, dfY_ls, dX_ls, dY_ls, dMod_ls, dArg_ls = calcJointSignals(myStudy['left_shoulder'])
    dfX_rh, dfY_rh, dX_rh, dY_rh, dMod_rh, dArg_rh = calcJointSignals(myStudy['right_hip'])
    dfX_rk, dfY_rk, dX_rk, dY_rk, dMod_rk, dArg_rk = calcJointSignals(myStudy['right_knee'])
    dfX_rf, dfY_rf, dX_rf, dY_rf, dMod_rf, dArg_rf = calcJointSignals(myStudy['right_foot'])
    dfX_lh, dfY_lh, dX_lh, dY_lh, dMod_lh, dArg_lh = calcJointSignals(myStudy['left_hip'])
    dfX_lk, dfY_lk, dX_lk, dY_lk, dMod_lk, dArg_lk = calcJointSignals(myStudy['left_knee'])
    dfX_lf, dfY_lf, dX_lf, dY_lf, dMod_lf, dArg_lf = calcJointSignals(myStudy['left_foot'])
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
    #insert into dataframe
    myStudy['frameSecs']=np.dot(myStudy.index, Ts)
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
    myStudy['d_rArm']=d_rArm
    myStudy['d_lArm']=d_lArm
    myStudy['d_rLeg']=d_rLeg
    myStudy['d_lLeg']=d_lLeg
    return myStudy

def calcSkeletonData(mySkeleton):
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
    return mySkeleton

def calcFreqData(myStudy):
    N=myStudy.index.size
    global powF_body, powF_rE, powF_rW, powF_lE, powF_lW, powF_rK, powF_rF, powF_lK, powF_lF
    powF_body=calcfftPower(myStudy['body_dmod'].to_numpy(), N)
    powF_rE=calcfftPower(myStudy['dRel_re'].to_numpy(), N)
    powF_rW=calcfftPower(myStudy['dRel_rw'].to_numpy(), N)
    powF_lE=calcfftPower(myStudy['dRel_le'].to_numpy(), N)
    powF_lW=calcfftPower(myStudy['dRel_lw'].to_numpy(), N)
    powF_rK=calcfftPower(myStudy['dRel_rk'].to_numpy(), N)
    powF_rF=calcfftPower(myStudy['dRel_rf'].to_numpy(), N)
    powF_lK=calcfftPower(myStudy['dRel_lk'].to_numpy(), N)
    powF_lF=calcfftPower(myStudy['dRel_lf'].to_numpy(), N)

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
    return freqTable

def loadStudioFileNames(studioFileName):
    global csvSrc, VideoSrc, VideoOut
    csvSrc="./videos/"+studioFileName+".csv" 
    VideoSrc="./videos/"+studioFileName+".mp4" 
    VideoOut="./videos/"+studioFileName+"out.mp4"
def openStudioFiles(fileName):
    global csvSrc, outputStatus, currFrame, currStartFrame, currEndFrame
    global VideoSrc, VideoFileName, VideoOut, VideoTmp
    global mySkeleton, myStudy, studyLen, rawCoords, Fs, Ts
    VideoFileName=fileName
    loadStudioFileNames(VideoFileName)
    mySkeleton=importSkeleton(csvSrc+".skl")
    myStudy=importStudy(csvSrc)
    studyLen=myStudy.index.size
    rawCoords=importStudy(csvSrc)
    Fs=30
    Ts=1/Fs
    currFrame=30
    currStartFrame = studyLen//2 - studyLen//10
    currEndFrame = studyLen//2 + studyLen//10


# Initial data 

curdoc().title = "Analisis Movimientos"
StudiesPath="./videos/"
VideosPath="../videos/"
VideoTmp="dashboard/static/tmpVideo.mp4"

VideoFileName="mov1"                # default file
openStudioFiles(VideoFileName)      # loads basic studio info and datasets

myStudy=calcJointSpeeds(myStudy)    # inserts spped data in main dataset
mySkeleton=calcSkeletonData(mySkeleton)             # inserts global indicators in skeleton
skelViewData=mySkeleton[mySkeleton["name_es"]!=""]  # creates dataset for table -- clean
skelViewData=skelViewData.drop(['SINGLEPOSE', 'baseColor'], axis=1)

freqTable=calcFreqData(myStudy)     #frequency data calculations
freqWindTable = getFreqWindowData(currStartFrame, currEndFrame)

#create datasources --init
tableSource = ColumnDataSource(skelViewData)
timeSeriesSource = ColumnDataSource(myStudy)
freqDataSource = ColumnDataSource(freqTable)
fWindDataSource = ColumnDataSource(freqWindTable)


#js controls functions

# Opening Menu Functions
def openStudioCbk(attr, old, new):
    global outputStatus, myStudy, mySkeleton, skelViewData, freqTable, freqWindTable
    global currStartFrame, currEndFrame, mainGraph, bodyHMHexTile, bodyHMColbar
    global tableSource, timeSeriesSource, freqDataSource
    if attr=='value':
        openStudioFiles(selectStudy.value)
        myStudy=calcJointSpeeds(myStudy)    # inserts spped data in main dataset
        mySkeleton=calcSkeletonData(mySkeleton)             # inserts global indicators in skeleton
        skelViewData=mySkeleton[mySkeleton["name_es"]!=""]  # creates dataset for table -- clean
        skelViewData=skelViewData.drop(['SINGLEPOSE', 'baseColor'], axis=1)
        freqTable=calcFreqData(myStudy)     #frequency data calculations
        freqWindTable = getFreqWindowData(currStartFrame, currEndFrame)
        #bodyHMColbar.color_mapper=colMapper
        #bodyHMHexTile.fill_color=transform('abs_dsplc', colMapper)
        mainGraph.x_range.start=currStartFrame
        mainGraph.x_range.end=currEndFrame
        tableSource.data=skelViewData
        timeSeriesSource.data=myStudy
        freqDataSource.data=freqTable
        outputStatus.text = "Datasets linked"
        updateGraphSkeleton(None)
        updatePhotoFrame(None)
        updateFreqHeatMap()
        updateStats()
        loadJsVideo()
    else: 
        return None 
#loads video file in temp dir and updates html
workingFile=Div(text="", width=100, name="workingFile", id="workingFile")   #preliminar definition
workingFile.tags=['./dashboard/static/']                                    #preliminar definition
def loadJsVideo():
    global workingFile, VideoTmp, outputStatus
    workingFile.text=VideoFileName+'.mp4'
    VideoTmp = './dashboard/static/'+VideoFileName+'.mp4'
    shutil.copyfile(VideoSrc, VideoTmp)
    outputStatus.text = "Opened " + VideoFileName+".mp4"
netGraph = GraphRenderer()      # preliminar definition used in network graph
myBkSkeleton, connxs = genBokeh_Skeleton(mySkeleton, myStudy.iloc[currFrame])
def updateGraphSkeleton(evt):   # evaluate in startup... globals needs to be defined though
    global currFrame, netGraph, myBkSkeleton
    skCoords=genBokeh_pelvis(rawCoords.iloc[currFrame])
    graph_layout = dict(zip(myBkSkeleton.index, skCoords))
    netGraph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
    netGraph.edge_renderer.data_source.data = connxs
def updateCurrFrame(evt):
    global currFrame, currEndFrame, currStartFrame, fWindDataSource, freqWindTable
    currFrame=correct_n_frame(evt.x)
    updateGraphSkeleton(None)
    updatePhotoFrame(None)
def calcCalories(startFrame, endFrame):
    return 0.187 * skelViewData.displacement.sum()
def updateFreqHeatMap():
    global freqHeatmap
    fqImage_raw=np.matrix([powF_rF[0:maxF], powF_rK[0:maxF], powF_rW[0:maxF], 
            powF_rE[0:maxF], powF_body[0:maxF], powF_lE[0:maxF], 
            powF_lW[0:maxF], powF_lK[0:maxF], powF_lF[0:maxF]])
    fqImage_raw=np.where(fqImage_raw>peak_ceil, peak_ceil, fqImage_raw)  
    fqImage_out=process_freq_heatmap(fqImage_raw, peak_ceil)
    freqHeatmap.image(image=[fqImage_out], x=0, y=0, dw=Fmax, dh=9, palette="Spectral11", level="image")


##########  Graphic Controls #########

# Opening menus controls definition
buttonSave = Button(label="Guardar Estudio", button_type="success", disabled=True)
outputStatus = Paragraph(text="Seleccione estudio para continuar", width=80, css_classes=['text-success'])
selectStudy = Select(title='Cargar Estudio', value=VideoFileName, options=readFilesList('studies'))
selectStudy.on_change('value', openStudioCbk)
selectVideo = Select(title='Abrir Video', value='', options=readFilesList('videos'), disabled=True)
def openVideoCbk(attr,old,new):
    global VideoSrc, csvSrc, VideoOut, buttonSave, outputStatus
    buttonSave.disabled = False
    outputStatus.text = "Opening Video " + selectVideo.value+".mp4"
    VideoFileName=selectVideo.value+'.mp4'
    VideoSrc=VideosPath+selectVideo.value+'.mp4'
    csvSrc=VideosPath+selectVideo.value+'.csv'
    VideoOut=VideosPath+selectVideo.value+'out.mp4'
    loadJsVideo()
selectVideo.on_change('value', openVideoCbk)
layoutOpening = row(selectStudy, selectVideo, column(outputStatus, buttonSave, width=120), 
        sizing_mode="scale_width", name="studies")
# add the layout to curdoc
curdoc().add_root(layoutOpening)
# End Opening Menus

# Stats controls definition
kpiLongVal=statKpi('kpi_long', 'clock-o', 'Duración(frames)')
kpiMovementVal=statKpi('kpi_mov', 'child', 'Movimiento neto (px)')
kpiCaloriesVal=statKpi('kpi_cal', 'heart-o', 'Total Calorias')
curdoc().add_root(kpiLongVal.render)
curdoc().add_root(kpiMovementVal.render)
curdoc().add_root(kpiCaloriesVal.render)
def updateStats():
    #outputStatus.text = "stats updated"
    kpiLongVal.setKpi(studyLen)
    kpiCaloriesVal.setKpi(calcCalories(currStartFrame, currEndFrame))
    kpiMovementVal.setKpi(skelViewData.displacement.sum())
    #updateGraphSkeleton(None)
buttonSave.on_click(updateStats) # late linking
#hidden field for js video linking
curdoc().add_root(workingFile)
js_videoConnCallback = CustomJS(args=dict(txtCtrl=workingFile), code="""
var video = document.getElementById('videoplayer');
video.src = txtCtrl.tags[0]+txtCtrl.text;
video.play();
// models passed as args are automagically available
""")
buttonUpdVideo = Button(label="⟳", name="btn_updvideo", button_type="success", 
    width=50, height=31, css_classes=['text-right'])
buttonUpdVideo.js_on_click(js_videoConnCallback)
curdoc().add_root(buttonUpdVideo)


# Timeseries

rollRg=(currStartFrame, currEndFrame)
mainGraph = figure(plot_height=160, tools="", toolbar_location=None, #name="line",
           x_range=rollRg, sizing_mode="scale_width")

mainGraph.line(x='index', y='body_dmod', source=timeSeriesSource, line_width=3, 
        alpha=0.7, color="#e7717d", legend_label="mod")
mainGraph.line(x='index', y='body_dx', source=timeSeriesSource, line_width=2, 
        alpha=0.7, color="#80949c", legend_label="vel.x")
mainGraph.line(x='index', y='body_dy', source=timeSeriesSource, line_width=2, 
        alpha=0.7, color="#916771", legend_label="vel.y")
mainGraph.yaxis.axis_label = 'vel (px/frame)'
mainGraph.background_fill_color="#f5f5f5"
mainGraph.grid.grid_line_color="white"
mainGraph.on_event('doubletap', updateCurrFrame)

select = figure(plot_height=50, plot_width=600, y_range=mainGraph.y_range,
                #x_axis_type="datetime", y_axis_type=None,
                tools="", toolbar_location=None, sizing_mode="scale_width")

range_rool = RangeTool(x_range=mainGraph.x_range)
range_rool.overlay.fill_color = "#e7717d"
range_rool.overlay.fill_alpha = 0.2

select.line(x='index', y='body_dmod', color="#e7717d", source=timeSeriesSource)
select.ygrid.grid_line_color = None
select.add_tools(range_rool)
select.toolbar.active_multi = range_rool
select.background_fill_color="#f5f5f5"
select.grid.grid_line_color="white"
select.x_range.range_padding = 0.1
select.xaxis.formatter = FuncTickFormatter(code="return parseFloat(tick*%f).toFixed(2)+ 's'"% Ts)
select.yaxis.visible=False

def updatePanWindow(evt):
    global currFrame, currEndFrame, currStartFrame, fWindDataSource, freqWindTable
    currStartFrame=correct_n_frame(mainGraph.x_range.start)
    currEndFrame=correct_n_frame(mainGraph.x_range.end)
    currFrame=correct_n_frame(evt.x)
    updateGraphSkeleton(None)
    updatePhotoFrame(None)
    outputStatus.text = "SF:%d, EF:%d, CF:%d (%.2f)"%(currStartFrame,currEndFrame,currFrame,currFrame*Ts)

    freqWindTable = getFreqWindowData(currStartFrame, currEndFrame)
    fWindDataSource.data = freqWindTable

select.on_event('panend', updatePanWindow)


layout = column(mainGraph, select, sizing_mode="scale_width", name="timeseries_global")
curdoc().add_root(layout)




# Current Photo

frame=getVideoFrame(VideoOut, currFrame)
frameRGBA = bokeh_postProc(frame)
h, w, c = frameRGBA.shape

xmin, ymin, xmax, ymax = zoomCenterSmCoords(h, w)
MyPhoto = figure(x_range=(xmin, xmax), y_range=(-ymax, ymin), name="currPhoto", 
                    tooltips=[("x, y", "$x{0.0} $y{0.0}"), ("val", "@image")],
                    tools='pan,wheel_zoom,box_zoom,reset,save', plot_width=390,plot_height=360, title=None)
MyPhoto.yaxis.visible = False
MyPhoto.image_rgba(image=[frameRGBA], x=0, y=-ymax, dw=w, dh=h)

def updatePhotoFrame(evt):
    #outputStatus.text = "Fired! "+VideoSrc 
    frame=getVideoFrame(VideoOut, currFrame)
    frameRGBA = bokeh_postProc(frame)
    MyPhoto.image_rgba(image=[frameRGBA], x=0, y=-ymax, dw=w, dh=h)
MyPhoto.on_event('reset', updatePhotoFrame)
curdoc().add_root(MyPhoto)




# Timesweries Detail charts

timeGraphTop = figure(plot_width=570, plot_height=270, x_range=mainGraph.x_range)
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
timeGraphTop.on_event('doubletap', updateCurrFrame)

timeGraphBott = figure(plot_width=570, plot_height=270, x_range=mainGraph.x_range)
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
timeGraphBott.on_event('doubletap', updateCurrFrame)

tabTimeTop = Panel(child=timeGraphTop, title="Miembros Superiores")
tabMTimeBtm = Panel(child=timeGraphBott, title="Miembros Inferiores")
tabsTs = Tabs(tabs=[ tabTimeTop, tabMTimeBtm ], name="timeseries_detail")
curdoc().add_root(tabsTs)



#frequency charts


ymaxFbarT=freqWindTable[['pFw_rArm', 'pFw_lArm']].max(axis=1).max()
fBarsTop = figure(plot_width=570, plot_height=270, y_range=(0,ymaxFbarT/3))
fBarsTop.step(source=fWindDataSource, x='x_wfreq', y='pFw_rArm', line_width=2, alpha=0.5,
       mode="center", color="#e7717d", legend_label="Pot. Brazo Der.")
fBarsTop.step(source=fWindDataSource, x='x_wfreq', y='pFw_lArm', line_width=2, alpha=0.5,
       mode="center", color="#7e685a", legend_label="Pot. Brazo Izq.")
fBarsTop.legend.location = "top_right"
fBarsTop.legend.click_policy="hide"
fBarsTop.xaxis.axis_label = "Freq (Hz)"
fBarsTop.xaxis.axis_label_standoff = -3

ymaxFbarB=freqWindTable[['pFw_rLeg', 'pFw_lLeg']].max(axis=1).max()
fBarsBott = figure(plot_width=570, plot_height=270, y_range=(0,ymaxFbarB/3))
fBarsBott.step(source=fWindDataSource, x='x_wfreq', y='pFw_rLeg', line_width=2, alpha=0.5,
       mode="center", color="#e7717d", legend_label="Pot. Pierna Der.")
fBarsBott.step(source=fWindDataSource, x='x_wfreq', y='pFw_lLeg', line_width=2, alpha=0.5,
       mode="center", color="#7e685a", legend_label="Pot. Pierna Izq.")
fBarsBott.legend.location = "top_right"
fBarsBott.legend.click_policy="hide"
fBarsBott.xaxis.axis_label = "Freq (Hz)"
fBarsBott.xaxis.axis_label_standoff = -3

colPalette=Spectral[5]
freqGraphTop = figure(plot_width=570, plot_height=270)
freqGraphTop.line(x='x_freq', y='powF_body', source=freqDataSource, line_width=2, 
        color=colPalette[2], legend_label="FFT Pecho.")
freqGraphTop.line(x='x_freq', y='powF_rE', source=freqDataSource, line_width=2, 
        color=colPalette[1], legend_label="FFT Codo Der.")
freqGraphTop.line(x='x_freq', y='powF_rW', source=freqDataSource, line_width=2, 
        color=colPalette[4], legend_label="FFT Muñeca Der.")
freqGraphTop.line(x='x_freq', y='powF_lE', source=freqDataSource, line_width=2, 
        color=colPalette[0], legend_label="FFT Codo Izq.")
freqGraphTop.line(x='x_freq', y='powF_lW', source=freqDataSource, line_width=2, 
        color=colPalette[3], legend_label="FFT Muñeca Izq")
freqGraphTop.legend.location = "top_right"
freqGraphTop.legend.click_policy="hide"

freqGraphBott = figure(plot_width=570, plot_height=270)
freqGraphBott.line(x='x_freq', y='powF_body', source=freqDataSource, line_width=2, 
        color=colPalette[2], legend_label="FFT Pecho.")
freqGraphBott.line(x='x_freq', y='powF_rK', source=freqDataSource, line_width=2, 
        color=colPalette[1], legend_label="FFT Rodilla Der.")
freqGraphBott.line(x='x_freq', y='powF_rF', source=freqDataSource, line_width=2, 
        color=colPalette[4], legend_label="FFT Pie Der.")
freqGraphBott.line(x='x_freq', y='powF_lK', source=freqDataSource, line_width=2, 
        color=colPalette[0], legend_label="FFT Rodilla Izq.")
freqGraphBott.line(x='x_freq', y='powF_lF', source=freqDataSource, line_width=2, 
        color=colPalette[3], legend_label="FFT Pie Izq.")
freqGraphBott.legend.location = "top_right"
freqGraphBott.legend.click_policy="hide"

tabFBarsTop = Panel(child=fBarsTop, title="Bandas Freq Brazos (sel)")
tabFBarsBtm = Panel(child=fBarsBott, title="Bandas Freq Piernas (sel)")
tabFreqTop = Panel(child=freqGraphTop, title="FFT Global Brazos")
tabFreqBtm = Panel(child=freqGraphBott, title="FFT Global Piernas")
tabsFs = Tabs(tabs=[ tabFBarsTop, tabFBarsBtm, tabFreqTop, tabFreqBtm ], name="frequency_detail")
curdoc().add_root(tabsFs)



# heatmaps Graphs

#hex tile map
colMapper = LinearColorMapper(palette=Spectral11, low=skelViewData.abs_dsplc.min(), 
                           high=skelViewData.abs_dsplc.max())

bodyHM = figure(plot_width=540,plot_height=290, toolbar_location=None)
bodyHM.xaxis.visible = False
bodyHM.yaxis.visible = False
bodyHM.grid.visible = False

bodyHM.hex_tile(q='hexCoordQ', r='hexCoordR', size=1, source=tableSource, 
           fill_color=transform('abs_dsplc', colMapper), name="bodyHMHexTile",
           line_color="black", line_width=1, alpha=0.9)

x, y = axial_to_cartesian(skelViewData.hexCoordQ, skelViewData.hexCoordR, 1, "pointytop")
bodyHM.text(x=x, y=y, text=[str(coord) for coord in skelViewData.name_es],
       text_baseline="middle", text_align="center", text_color="black", text_font_size="10px")

color_bar = ColorBar(color_mapper=colMapper, name="bodyHMColorMap",#ticker=LogTicker(), BasicTicker
                     label_standoff=5, major_label_text_color="gray", border_line_color=None, location=(0,0))
bodyHM.add_layout(color_bar, 'right')
bodyHMColbar=bodyHM.select({"name":"bodyHMColorMap"})
bodyHMHexTile=bodyHM.select({"name":"bodyHMHexTile"})

#Freq image map
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

freqHeatmap = figure(plot_width=540, plot_height=270, toolbar_location=None,
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
graphTileTextColor="#7e685a" #brown silver?
skPlot = figure(x_range=(0, w), y_range=(h, 0), name="netGraph", outline_line_color=None,
              tools='pan,wheel_zoom,box_zoom,reset,tap,box_select,hover', plot_width=570,plot_height=300, 
              tooltips=[("coord", "$x{0.0} $y{0.0}"), ("art", "$index")])
skPlot.xaxis.visible = False
skPlot.title.text_color = graphTileTextColor
skPlot.title.text_font_size="15px"
skPlot.title.align = "center"

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
skPlot.on_event('reset', updateGraphSkeleton)
curdoc().add_root(skPlot)




# Table


tableCols = [
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
data_table = DataTable(source=tableSource, name="dataTable_kpis", columns=tableCols, width=560, height=290)

curdoc().add_root(data_table)


# final callbacks linking
updateStats()
loadJsVideo()
