import cv2
import numpy as np
import datetime
import argparse
import imutils
from imutils.video import VideoStream
from dashed_line import drawline 

from utils import detector_utils as detector_utils

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()

Choice=int(input("Enter Choice 0 For Video Stream , 1 For File :"))

#For Machine
Line_Position1=int(input("Enter the position for the line of machine :"))

#For Safety
Line_Position2=int(input("Enter the position for the line of safety :"))

if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.60
    
    if(Choice is 0):
        # Get stream from webcam and set parameters)
        vs = VideoStream().start()

    else:        
        filename=input("Enter filename :")
        #Playing video file
        cap = cv2.VideoCapture(filename)
    
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")


    # max number of hands we want to detect/track
    num_hands_detect = 20

    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0
    
    im_height, im_width = (None, None)

    try:
        while True:
            # Read Frame and process
            cv2.namedWindow('Detection',cv2.WINDOW_NORMAL)
            if(Choice is 0):
                frame = vs.read()
        
               
            elif(Choice is 1):
                # Capture frame-by-frame
                ret, frame = cap.read()
                

            if im_height == None:
                im_height, im_width = frame.shape[:2]

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")
            
            
            #Draw line on the video frame for machine
            drawline(img=frame,pt1=(0,Line_Position1),pt2=(frame.shape[1], Line_Position1),color=(255, 0, 0),thickness=1,style='dotted',gap=20)
            
            #Draw line on the video frame for safety
            drawline(img=frame,pt1=(0,Line_Position2),pt2=(frame.shape[1], Line_Position2),color=(255, 0, 0),thickness=1,style='dotted',gap=20)
            #cv2.line(img=frame, pt1=(0, Line_Position2), pt2=(900, Line_Position2), color=(255, 0, 0), thickness=1, lineType=8, shift=0)

            # Run image through tensorflow graph
            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)

               
            # Draw bounding boxeses and text
            bounding_mid=detector_utils.draw_box_on_image(
                num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame)
            
            if(bounding_mid):
                cv2.line(img=frame, pt1=bounding_mid, pt2=(bounding_mid[0],Line_Position2), color=(255, 0, 0), thickness=1, lineType=8, shift=0)
                distance_from_line=bounding_mid[1]-Line_Position2
            
            
            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time

            if args['display']:
                # Display FPS on frame
                detector_utils.draw_text_on_image1("FPS : " + str("{0:.2f}".format(fps)), frame)
                if(bounding_mid):
                    detector_utils.draw_text_on_image2("D : " + str("{0:.2f}".format(distance_from_line)), frame)
                    if(distance_from_line<=0):
                        detector_utils.draw_text_on_image3("ALERT", frame)
                else:
                    detector_utils.draw_text_on_image2("D : No Hand", frame)
               
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    if(Choice is 1):
                        # When everything done, release the video capture object
                        cap.release()
                    cv2.destroyAllWindows()
                    if(Choice is 0):    
                        vs.stop()
                    break

        print("Average FPS: ", str("{0:.2f}".format(fps)))

    except KeyboardInterrupt:
        print("Average FPS: ", str("{0:.2f}".format(fps)))
