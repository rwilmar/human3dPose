import argparse
import cv2
import numpy as np

from inference import Network
from model_handlers import handle_output
from image_handlers import preprocessing, create_output_image

#source /opt/intel/openvino/bin/setupvars.sh            ##remember##

_MODEL_BOXES="./models/person-detection-retail-0013.xml" # more available with different inputs !!!
_MODEL_MULTIPOSE="./models/human-pose-estimation-0001.xml"
_MODEL_SINGLEPOSE="./models/single-human-pose-estimation-0001.xml" 

def get_args():
    '''
    Gets the arguments from the command line. 
    means ho be always oppened from a command line interface
    '''

    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")
    # -- Create the descriptions for the commands

    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input image"
    t_desc = "The type of model: POSE, TEXT or CAR_META or PERSON_DET"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-i", help=i_desc, required=True)
    required.add_argument("-t", help=t_desc, default="PERSON_DET")
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    args = parser.parse_args()
    args.m=get_model(args.t)

    return args

def get_model(model_type):
    '''
    Returns the related model, based on the model_type required.
    '''
    if model_type == "POSE":
        return _MODEL_MULTIPOSE
    elif model_type == "PERSON_DET":
        return _MODEL_BOXES
    elif model_type == "SINGLEPOSE":
        return _MODEL_SINGLEPOSE
    else:
        return _MODEL_MULTIPOSE

def perform_inference(args):
    '''
    Performs inference #1 on an input image, given a model.
    only working on images temp.... video irl to implement
    '''
    # Create a Network for using the Inference Engine
    inference_network = Network()
    # Load the model in the network, and obtain its input shape
    n, c, h, w = inference_network.load_model(args.m, args.d, args.c)

    # Read the input image
    image = cv2.imread(args.i)

    preprocessed_image = preprocessing(image, h, w)
    
    # Perform synchronous inference on the image
    inference_network.sync_inference(preprocessed_image)

    # Obtain the output of the inference request
    output = inference_network.extract_output()

    processed_output = handle_output(args.t)(output,image.shape)

    # Create an output image based on network
    try:
        output_image = create_output_image(args.t, image, processed_output)
        print("Processing succeded :)")
    except:
        output_image = image
        print("Image not processed :(")
        
    #output_image = create_output_image(args.t, image, processed_output)

    # Save down the resulting image
    cv2.imwrite("./images/{}-output.png".format(args.t), output_image)



def main():
    args = get_args()

    print("test")

    perform_inference(args)


if __name__ == "__main__":
    main()