import cv2
import numpy as np


def handle_pose(output, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    '''
    # TODO 1: Extract only the second blob output (keypoint heatmaps)
    #[1, 38, 32, 57] and [1, 19, 32, 57] 
    poses = output['Mconv7_stage2_L2']
    print(poses.shape)
    [b,c,h,w]=poses.shape
    # TODO 2: Resize the heatmap back to the size of the input

    #out_heatmap = np.zeros([heatmaps.shape[1], input_shape[0], input_shape[1]])
    out_heatmap = np.zeros([c, input_shape[0], input_shape[1]])
    # Iterate through and re-size each heatmap
    for h in range(len(poses[0])):
        out_heatmap[h] = cv2.resize(poses[0][h], input_shape[0:2][::-1])

    return out_heatmap


def handle_text(output, input_shape):
    '''
    Handles the output of the Text Detection model.
    Returns ONLY the text/no text classification of each pixel,
        and not the linkage between pixels and their neighbors.
        
        The net outputs two blobs. Refer to PixelLink and demos for details.
    [1x2x192x320] - logits related to text/no-text classification for each pixel.        
        [1x16x192x320] - logits related to linkage between pixels and their neighbors.
    '''
    # Get only text detections above 0.5 confidence, set to 255
    [h, w, c]=input_shape
    print(input_shape, h, w, c)
    print(output.keys())    
    # TODO 1: Extract only the first blob output (text/no text classification)
    output_1=output['model/segm_logits/add']
    #[1x2x192x320] - logits related to text/no-text classification for each pixel.
    print(output_1.shape)
    
    # TODO 2: Resize this output back to the size of the input
    out_text = np.empty([output_1.shape[1], input_shape[0], input_shape[1]])
    for t in range(len(output_1[0])):
        out_text[t] = cv2.resize(output_1[0][t], input_shape[0:2][::-1])

    return out_text


def handle_car(output, input_shape):
    '''
    Handles the output of the Car Metadata model.
    Returns two integers: the argmax of each softmax output.
    The first is for color, and the second for type.
    '''
    # TODO 1: Get the argmax of the "color" output
    
    # TODO 2: Get the argmax of the "type" output
    
    #print(output.keys())
    out_color=output['color'].flatten()
    out_cartype=output['type'].flatten()
#    print(out_color.shape)
#    print(out_cartype.shape)
   
    color_class=np.argmax(out_color)
    type_class=np.argmax(out_cartype)

    return color_class, type_class

def handle_personDet(output, input_shape):
    '''
    Handles the output of the Person Detection model.
    Returns two integers: the argmax of each softmax output.
    The first is for color, and the second for type.
    '''
    print(output.keys())
    detPeople=output['detection_out'] #[1x1xNx7], where N is the number of detected pedestrians.   
    print(detPeople.shape)
#    print(detPeople[0][0].shape)
#    print(out_cartype.shape)
    return detPeople[0][0]

def handle_output(model_type):
    '''
    Returns the related function to handle an output,
        based on the model_type being used.
    '''
    if model_type == "POSE":
        return handle_pose
    elif model_type == "TEXT":
        return handle_text
    elif model_type == "CAR_META":
        return handle_car
    elif model_type == "PERSON_DET":
        return handle_personDet
    else:
        return None