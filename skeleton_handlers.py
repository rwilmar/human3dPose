import cv2
import numpy as np
import pandas as pd

# Calc the 2d-modulus (geometric distance) between 2 points
def calcDistance(p1, p2):
    x1,y1=p1
    x2,y2=p2
    return (int(abs(x2-x1))**2 + int(abs(y2-y1))**2)**0.5

# Calc joint position from joint heatmap
def calc_JointPosition(JointHeatmap):
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(JointHeatmap)
    return max_loc

# Calc the middle point between 2-D joints
def calcMiddlePoint(p1, p2):
    x1,y1=p1
    x2,y2=p2
    return (min(x1,x2)+int(abs(x2-x1)/2), min(y1,y2)+int(abs(y2-y1)/2))

# Converts coords in string to python tuples
def cleanStrCoords(coords):
    if isinstance(coords, str):
        res=coords[1:-1].split(",")
        coords=tuple([int(i) for i in res])
    return coords

#Export Skeleton dataframe to csv
def exportSkeleton(skeleton, csvPath):
    skeleton.to_csv(csvPath,sep=";",index=True,header=True)
    return True

# Creates pelvis in Skeleton coords (pandas series)
def genBokeh_pelvis(skelCoords):
    pelvisIx=skelCoords.size
    pelvis=(0,0)
    if (skelCoords.loc['left_hip']!=(0,0)) and (skelCoords.loc['right_hip']!=(0,0)):
        pelvis = calcMiddlePoint(skelCoords.loc['left_hip'],skelCoords.loc['right_hip'])
    if pelvisIx >18: #should have pelvis
        skelCoords['pelvis']=pelvis
    else:
        skelCoords=skelCoords.append(pd.Series([pelvis], index=[0]), ignore_index=True)
    return skelCoords[skelCoords!=(0,0)] #filter undetected joints

# Generates skeleton Graph data for Bokeh Networkx
def genBokeh_Skeleton(skeleton, skelCoords):
    #coordinates & colors
    mySkeleton = pd.DataFrame(index=skeleton.index)
    mySkeleton['coord2d']=skelCoords
    mySkeleton['color']=skeleton["hexColor"]
    mySkeleton['name']=skeleton.index
    mySkeleton.index=np.arange(mySkeleton.shape[0])
    # build pelvis
    pelvis=(0,0)  
    pelvisix=mySkeleton.shape[0]#should be 19 check genSkeletonPelvis
    if (skelCoords.loc['left_hip']!=(0,0)) and (skelCoords.loc['right_hip']!=(0,0)):
        pelvis = calcMiddlePoint(skelCoords.loc['left_hip'],skelCoords.loc['right_hip'])
    mySkeleton = mySkeleton.append({"name": "pelvis", "coord2d": pelvis,
                                   "color": mySkeleton.loc[0]['color']}, ignore_index=True) 
    mySkeleton = mySkeleton[mySkeleton.coord2d!=(0,0)] #filter undetected joints

    # graphbuilder- connections
    jSrc=[]
    jDst=[]
    jSrc.append(0)
    jDst.append(1)#head-neck
    jSrc.append(2)
    jDst.append(3)#right_shoulder - right_elbow
    jSrc.append(3)
    jDst.append(4)#right_elbow - right_wrist
    jSrc.append(5)
    jDst.append(6)#left_shoulder - left_elbow
    jSrc.append(6)
    jDst.append(7)#left_elbow - left_wrist
    jSrc.append(8)
    jDst.append(9)#right_hip - right_knee
    jSrc.append(9)
    jDst.append(10)#right_knee - right_foot
    jSrc.append(11)
    jDst.append(12)#left_hip - left_knee
    jSrc.append(12)
    jDst.append(13)#left_knee - left_foot
    jSrc.append(14)
    jDst.append(15)#right_eye - left_eye
    #jSrc.append(14)
    #jDst.append(1)#right_eye - head
    #jSrc.append(15)
    #jDst.append(1)#left_eye - head
    jSrc.append(5)
    jDst.append(2)#left_shoulder - right_shoulder
    jDst.append(2)
    jSrc.append(19)#right_shoulder - pelvis
    jDst.append(5)
    jSrc.append(19)#left_shoulder - pelvis
    jDst.append(8)
    jSrc.append(11)#right_hip - left_hip

    conns={'start':jSrc, 'end':jDst}
    return mySkeleton, conns

# Generates skeleton image frame with opencv
def genCV_Skeleton(skeleton, skelCoords, frame, darkmode=True):
    colorIx=skeleton['baseColor']
    bLineColor=colorIx['head']
    imout = np.zeros(frame.shape, dtype=np.int8)

    if not(skelCoords.loc['right_eye']==(0,0) or skelCoords.loc['left_eye']==(0,0)):
        cv2.line(imout, skelCoords.loc['right_eye'], skelCoords.loc['left_eye'], bLineColor,5)
        cv2.circle(imout, skelCoords.loc['right_eye'], 7, colorIx['right_eye'],5)
        cv2.circle(imout, skelCoords.loc['left_eye'], 7, colorIx['left_eye'],5)
        if (skelCoords.loc['head']!=(0,0)):
            cv2.line(imout, skelCoords.loc['right_eye'], skelCoords.loc['head'], bLineColor,5)
            cv2.line(imout, skelCoords.loc['left_eye'], skelCoords.loc['head'], bLineColor,5)

    if (skelCoords.loc['head']!=(0,0)):
        cv2.circle(imout, skelCoords.loc['head'], 7, colorIx['head'],5)

    pelvis=(0,0)
    if (skelCoords.loc['left_hip']!=(0,0)):
        cv2.circle(imout, skelCoords.loc['left_hip'], 7, colorIx['left_hip'],5)
        if (skelCoords.loc['right_hip']!=(0,0)):
            pelvis = calcMiddlePoint(skelCoords.loc['left_hip'],skelCoords.loc['right_hip'])
            cv2.line(imout, skelCoords.loc['left_hip'], skelCoords.loc['right_hip'], bLineColor,5)
    if (skelCoords.loc['right_hip']!=(0,0)):
        cv2.circle(imout, skelCoords.loc['right_hip'], 7, colorIx['right_hip'],5)

    #headRad=int(calcDistance(skelCoords.loc['left_shoulder'],skelCoords.loc['right_shoulder'])/4)+3
    if not(skelCoords.loc['left_shoulder']==(0,0) or skelCoords.loc['right_shoulder']==(0,0)):
        cv2.line(imout, skelCoords.loc['left_shoulder'], skelCoords.loc['right_shoulder'], bLineColor,5)
        neck = calcMiddlePoint(skelCoords.loc['left_shoulder'],skelCoords.loc['right_shoulder'])
        if (skelCoords.loc['head']!=(0,0)):
            cv2.line(imout, skelCoords.loc['head'], neck, bLineColor,5)
        if (pelvis!=(0,0)):
            cv2.line(imout, pelvis, skelCoords.loc['left_shoulder'], bLineColor,5)
            cv2.line(imout, pelvis, skelCoords.loc['right_shoulder'], bLineColor,5)

    if (skelCoords.loc['left_wrist']!=(0,0)):
        cv2.circle(imout, skelCoords.loc['left_wrist'], 7, colorIx['left_wrist'],5)
        if (skelCoords.loc['left_elbow']!=(0,0)):
            cv2.line(imout, skelCoords.loc['left_elbow'], skelCoords.loc['left_wrist'], bLineColor,5)
            cv2.circle(imout, skelCoords.loc['left_elbow'], 7, colorIx['left_elbow'],5)
    if (skelCoords.loc['left_shoulder']!=(0,0)):
        cv2.circle(imout, skelCoords.loc['left_shoulder'], 7, colorIx['left_shoulder'],5)
        if (skelCoords.loc['left_elbow']!=(0,0)):
            cv2.line(imout, skelCoords.loc['left_shoulder'], skelCoords.loc['left_elbow'], bLineColor,5)

    if (skelCoords.loc['right_wrist']!=(0,0)):
        cv2.circle(imout, skelCoords.loc['right_wrist'], 7, colorIx['right_wrist'],5)
        if (skelCoords.loc['right_elbow']!=(0,0)):
            cv2.line(imout, skelCoords.loc['right_elbow'], skelCoords.loc['right_wrist'], bLineColor,5)
            cv2.circle(imout, skelCoords.loc['right_elbow'], 7, colorIx['right_elbow'],5)
    if (skelCoords.loc['right_shoulder']!=(0,0)):
        cv2.circle(imout, skelCoords.loc['right_shoulder'], 7, colorIx['right_shoulder'],5)
        if (skelCoords.loc['right_elbow']!=(0,0)):
            cv2.line(imout, skelCoords.loc['right_shoulder'], skelCoords.loc['right_elbow'], bLineColor,5)

    if (skelCoords.loc['left_foot']!=(0,0)):
        cv2.circle(imout, skelCoords.loc['left_foot'], 7, colorIx['left_foot'],5)
        if (skelCoords.loc['left_knee']!=(0,0)):
            cv2.line(imout, skelCoords.loc['left_knee'], skelCoords.loc['left_foot'], bLineColor,5)
            cv2.circle(imout, skelCoords.loc['left_knee'], 7, colorIx['left_knee'],5)
    if not(skelCoords.loc['left_hip']==(0,0) or skelCoords.loc['left_knee']==(0,0)):
        cv2.line(imout, skelCoords.loc['left_hip'], skelCoords.loc['left_knee'], bLineColor,5)        

    if (skelCoords.loc['right_foot']!=(0,0)):
        cv2.circle(imout, skelCoords.loc['right_foot'], 7, colorIx['right_foot'],5)
        if (skelCoords.loc['right_knee']!=(0,0)):
            cv2.line(imout, skelCoords.loc['right_knee'], skelCoords.loc['right_foot'], bLineColor,5)
            cv2.circle(imout, skelCoords.loc['right_knee'], 7, colorIx['right_knee'],5)
    if not(skelCoords.loc['right_hip']==(0,0) or skelCoords.loc['right_knee']==(0,0)):
        cv2.line(imout, skelCoords.loc['right_hip'], skelCoords.loc['right_knee'], bLineColor,5)
    
    if darkmode==True:
        return imout.astype('uint8')
    else:
        return cv2.add(imout.astype('uint8'),frame)

def HexToBGR(HexColor):
    h=HexColor.lstrip('#')
    R,G,B= tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return (B,G,R,30)

# Imports skeleton from file
def importSkeleton(skeleton_csv):
    skeleton = pd.read_csv(skeleton_csv,sep=";",index_col=0)
    skeleton["baseColor"]=skeleton["baseColor"].apply(cleanStrCoords)
    return skeleton

# Imports csv study Dataframe of coords
def importStudy(study_csv):
    videoCoords=pd.read_csv(study_csv,sep=";",index_col=0) 
    for col in videoCoords.columns:
        videoCoords[col]=videoCoords[col].apply(cleanStrCoords)
    return videoCoords

# Creates a new skeleton with a Hex array[5] palette
def skeleton_creator(HexColorPalette = None):
    #index for multipose model
    ixn=['head','neck','right_shoulder','right_elbow','right_wrist','left_shoulder','left_elbow','left_wrist',
         'right_hip','right_knee','right_foot','left_hip','left_knee','left_foot',
         'right_eye','left_eye','right_ear', 'left_ear', 'indet']
    n=len(ixn)
    ixMultiPose=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    ixSinglePose=np.array([0,None,6,8,10,5,7,9,12,14,16,11,13,15,2,1,4,3,None])

    baseColor=[]
    hexColor=[]
    if HexColorPalette==None:
        for p in range(n):
            baseColor.append(( 0, 200, 200)) # default color if no palette
            hexColor.append("#00c8c8") # default hex color
    else:
        #should be a 5 row hex color palette
        colorArr=HexColorPalette
        baseColor=[HexToBGR(colorArr[2]), HexToBGR(colorArr[2]),
                   HexToBGR(colorArr[1]),HexToBGR(colorArr[1]),HexToBGR(colorArr[1]),
                   HexToBGR(colorArr[0]),HexToBGR(colorArr[0]),HexToBGR(colorArr[0]),
                   HexToBGR(colorArr[4]),HexToBGR(colorArr[4]),HexToBGR(colorArr[4]),
                   HexToBGR(colorArr[3]),HexToBGR(colorArr[3]),HexToBGR(colorArr[3]),
                   HexToBGR(colorArr[1]),HexToBGR(colorArr[0]),
                   HexToBGR(colorArr[1]), HexToBGR(colorArr[0]), (0,0,0)]
        hexColor=[colorArr[2], colorArr[2], colorArr[1], colorArr[1], colorArr[1],
                   colorArr[0], colorArr[0], colorArr[0], colorArr[4], colorArr[4], colorArr[4],
                   colorArr[3], colorArr[3], colorArr[3], colorArr[1], colorArr[0],
                   colorArr[1], colorArr[0], "#0f0f0f"]
    data = {'SINGLEPOSE':ixSinglePose, 'MULTIPOSE': ixMultiPose, 'baseColor':baseColor, 'hexColor':hexColor} 
    return pd.DataFrame(data, index=ixn)

# Return centered coords for square minsize frame (zoom-in center)
def zoomCenterSmCoords(h, w):
    if h>w:
        xmin, ymin, xmax, ymax = 0, int((h-w)/2), w, w+int((h-w)/2)
    else:
        xmin, ymin, xmax, ymax = int((w-h)/2), 0, h+int((w-h)/2), h
    return xmin, ymin, xmax, ymax
    
# Return centered coords for square maxsize frame (zoom-out center)
def zoomCenterLgCoords(h, w):
    if h>w:
        xmin, ymin, xmax, ymax = -int((h-w)/2), 0, w+int((h-w)/2), h
    else:
        xmin, ymin, xmax, ymax = 0, -int((w-h)/2), w, h+int((w-h)/2)
    return xmin, ymin, xmax, ymax
