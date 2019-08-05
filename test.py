import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print(ret, frame.shape)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Display the resulting frame
    # if ret:
    plt.imshow(frame)
    plt.ion()
    plt.show()
    plt.pause(0.0001)
    # cv2.imshow('frame',frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()