import cv2

# Returns the number of frames
def FrameCount(filename):
    video = cv2.VideoCapture(filename)

    if not video.isOpened():
        print("Invalid File")
        return None

    return int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Get Specific Frame From Video
def GetFrame(filename, framenum):
    video = cv2.VideoCapture(filename)
    
    if not video.isOpened():
        print("Invalid File")
        return None

    video.set(cv2.CAP_PROP_POS_FRAMES, framenum)

    valid, frame = video.read()

    if valid:
        return frame
    else:
        return None
    

# test Frame count function
print(FrameCount("testvideo.mp4"))
frame = GetFrame("testvideo.mp4", 40)

# test get specific frame from video
cv2.imshow('Frame', frame)
cv2.waitKey(0)


