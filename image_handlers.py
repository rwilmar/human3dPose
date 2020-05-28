import cv2
import numpy as np

# takes frame picture from a video
def getVideoFrame(videoFile, frameNumber, enableErrImg=True):
    capture = cv2.VideoCapture(videoFile, cv2.CAP_FFMPEG)
    if (capture.isOpened() == False): 
        raise("Unable to open video file")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frameNumber)
    ret, frame = capture.read()
    capture.release()
    if not ret:
        if enableErrImg:
            frame=cv2.imread("./images/error.png")
        else:
            raise("Can't receive frame (video end?)")
    return frame

# Prepares frame to bokeh (RGBA + flipped)
def bokeh_postProc(image):
    frameRGBA = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return cv2.flip(frameRGBA, 0)

#Preprocess the input image.
def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize width and height
    - Transpose BGR to RGB
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image


def get_mask(processed_output):
    '''
    Given an input image size and processed output for a semantic mask,
    returns a masks able to be combined with the original image.
    '''
    # Create an empty array for other color channels of mask
    empty = np.zeros(processed_output.shape)
    # Stack to make a Green mask where text detected
    mask = np.dstack((empty, processed_output, empty))

    return mask

def draw_boxes(frame, result, ct, width, height):
    '''
    Draw bounding boxes onto the frame.
    with a minimun confidence thereshold ct
    '''
    for box in result: # Output shape is nx7
        conf = box[2]
        if conf >= ct:
            print(box)
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            print(xmin, ymin, xmax, ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 153, 255), 2)
    return frame

def create_output_image(model_type, image, output):
    '''
    Using the model type, input image, and processed output,
    creates an output image showing the result of inference.
    '''
    [h, w, c] = image.shape
    if model_type == "POSE":
        # Remove final part of output not used for heatmaps
        output = output[:-1]
        # Get only pose detections above 0.5 confidence, set to 255
        for c in range(len(output)):
            output[c] = np.where(output[c]>0.4, 250, 0)
        # Sum along the "class" axis
        output = np.sum(output, axis=0)
        # Get semantic mask
        pose_mask = get_mask(output)
        # Combine with original image
        image = image + pose_mask
        return image

    elif model_type == "SINGLEPOSE":
        #print (output.shape)
        # Get only pose detections above 0.5 confidence, set to 255
        for c in range(len(output)):
            output[c] = np.where(output[c]>0.4, (output[c]-0.4)*100, 0)
        # Sum along the "class" axis
        output = np.sum(output, axis=0)
        # Get semantic mask
        pose_mask = get_mask(output)
        # Combine with original image
        imageBk = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(imageBk, cv2.COLOR_GRAY2BGR)
        #image = image.astype(int) pose_mask.astype(int)
        return image

    elif model_type == "MULTIPOSE":
        # Remove final part of output not used for heatmaps
        output = output[:-1]

        # Get only pose detections above 0.5 confidence, set to 255
        for c in range(len(output)):
            output[c] = np.where(output[c]>0.4, (output[c]-0.4), 0)
        # Sum along the "class" axis
        output = np.sum(output, axis=0)
        # Get semantic mask
        pose_mask = get_mask(output)
        # Combine with original image
        imageBk = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(imageBk, cv2.COLOR_GRAY2BGR)
        #image = image.astype(int) pose_mask.astype(int)
        return image
    
    elif model_type == "TEXT":
        # Get only text detections above 0.5 confidence, set to 255
        output = np.where(output[1]>0.5, 255, 0)
        # Get semantic mask
        text_mask = get_mask(output)
        # Add the mask to the image
        image = image + text_mask
        return image

    elif model_type == "CAR_META":
        # Get the color and car type from their lists
        color = CAR_COLORS[output[0]]
        car_type = CAR_TYPES[output[1]]
        # Scale the output text by the image shape
        scaler = max(int(image.shape[0] / 1000), 1)
        # Write the text of color and type onto the image
        image = cv2.putText(image, 
            "Color: {}, Type: {}".format(color, car_type), 
            (50 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 
            2 * scaler, (255, 255, 255), 3 * scaler)
        return image

    elif model_type == "PERSON_DET":
        print("returned")
        # Get only text detections above 0.5 confidence, set to 255
        #output = np.where(output[1][2]>0.5, output, 0)
        nPers, perVector = output.shape
        print("modelo de personas: ", nPers)
        
        return draw_boxes(image, output, 0.4, w, h)
    else:
        print("Unknown model type, unable to create output image.")
        return image

def process_freq_heatmap(fqImage_raw, peak_ceil):
    fqImage_raw=np.dot(fqImage_raw, (254/peak_ceil))
    fqImage_out = cv2.resize(fqImage_raw.astype(np.uint8), (90,90)) # resize to trunk resol.
    fqImage_out =np.dot(fqImage_out.astype(np.double), (peak_ceil/254)) #recover values
    return fqImage_out


        