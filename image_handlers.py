import cv2
import numpy as np

'''
Preprocess the input image.
'''
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
