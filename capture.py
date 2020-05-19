import cv2
import time
import os


# Start cameras
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
capture1 = cv2.VideoCapture("rtsp://192.168.1.45:8554/stream.amp", cv2.CAP_FFMPEG)
capture2 = cv2.VideoCapture("rtsp://192.168.1.41:8080/video/h264", cv2.CAP_FFMPEG)
#video = cv2.VideoCapture(0);

# Check if camera opened successfully
if (capture1.isOpened() == False): 
  print("Unable to read camera 1")
if (capture2.isOpened() == False): 
  print("Unable to read camera 2")

w1 = int(capture1.get(3))
h1 = int(capture1.get(4))
w2 = int(capture2.get(3))
h2 = int(capture2.get(4))

# Start Videos
video1 = cv2.VideoWriter('outvideo1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (w1,h1))
video2 = cv2.VideoWriter('outvideo2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (w2,h2))


# Number of frames to capture
num_frames = 120;
print("Capturing {0} frames".format(num_frames))

# Start time
start = time.time()

# Grab a few frames
for i in range(0, num_frames):
   ret, frame1 = capture1.read()
   ret, frame2 = capture2.read()
   video1.write(frame1)
   video2.write(frame2)
# End time
end = time.time()

capture1.release()
capture2.release()

video1.release()
video2.release()

# Time elapsed
seconds = end - start
print("Time taken : {0} seconds".format(seconds))

# Calculate frames per second
fps = num_frames / seconds
print("Estimated frames per second : {0}".format(fps))