import cv2,time
video = cv2.VideoCapture(0)
first_frame = None
while True:
    check,frame = video.read()
    #convert the frames into gray else is will not supported by
    # findCoutours etc.
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    if first_frame is None:
        first_frame = gray
        continue


    # Calculate the difference between the first frame and next
    # other frames
    delta_frame = cv2.absdiff(first_frame,gray)

    # Provide the threshold value such that less than 30 value will be
    # black and greater than 30 pixels will be white.
    thresh_delta = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta,None,iterations=0)


    # the counter area to add the borders.
    (_,cnts,_) = cv2.findContours(thresh_delta.copy(),
                    cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    # To remove shadows , basically it will keep that part which
    # is white 
    for coutour in cnts:
        if cv2.contourArea(coutour) < 999:
            continue

        # Create a rectangular box
        (x,y,w,h) = cv2.boundingRect(coutour)
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,5,10),3)

    # It will show normal video with box around the object
    cv2.imshow("Frame",frame)

    #Tr will show gray video
    cv2.imshow("Capturing",gray)

    #Gaussian blur image
    cv2.imshow("Delta",delta_frame)

    # Avobe 30 value object it will show while less blackq
    cv2.imshow("Thresh",thresh_delta)
    key = cv2.waitKey(1)
    # when user press 'q' then all windows will closed.
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()



 
