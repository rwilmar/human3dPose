import cv2
import numpy as np

def handle_singlepose(output, input_shape):
    print(output.keys())
    poses = output['1109']
    [b,c,h,w]=poses.shape
    #print(poses.shape)
    #print(np.amax(poses[0][1]))
    out_heatmap = np.zeros([c, input_shape[0], input_shape[1]])
    for h in range(len(poses[0])):
        out_heatmap[h] = cv2.resize(poses[0][h], input_shape[0:2][::-1])
    return out_heatmap

def handle_pose(output, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    '''
    # TODO 1: Extract only the second blob output (keypoint heatmaps)
    #[1, 38, 32, 57] and [1, 19, 32, 57] 
    poses = output['Mconv7_stage2_L2']
    #print(poses.shape)
    [b,c,h,w]=poses.shape
    # TODO 2: Resize the heatmap back to the size of the input

    #out_heatmap = np.zeros([heatmaps.shape[1], input_shape[0], input_shape[1]])
    out_heatmap = np.zeros([c, input_shape[0], input_shape[1]])
    # Iterate through and re-size each heatmap
    for h in range(len(poses[0])):
        out_heatmap[h] = cv2.resize(poses[0][h], input_shape[0:2][::-1])

    return out_heatmap

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
    return detPeople[0][0]

def handle_output(model_type):
    '''
    Returns the related function to handle an output,
        based on the model_type being used.
    '''
    if model_type == "POSE":
        return handle_pose
    elif model_type == "PERSON_DET":
        return handle_personDet
    elif model_type == "SINGLEPOSE":
        return handle_singlepose
    elif model_type == "MULTIPOSE":
        return handle_pose
    else:
        return None
