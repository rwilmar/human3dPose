import bokeh
import cv2
import sys
from openvino.inference_engine import IENetwork, IECore

print('3d Pose Stimation Test Script')
print()
print('Sys version:', sys.version_info)
try:
    ie=IECore()
    print('OpenVINO: Successfully Loaded ')
except:
    raise TypeError("OpenVINO Failed Loading")
print('OpenCV version:',cv2.__version__)
print('Bokeh version:', bokeh.__version__)

