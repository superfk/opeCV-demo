from pypylon import pylon
import cv2 as cv


# camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
# camera.Open()

# # demonstrate some feature access
# new_width = camera.Width.GetValue() - camera.Width.GetInc()
# if new_width >= camera.Width.GetMin():
#     camera.Width.SetValue(new_width)

# # numberOfImagesToGrab = 100
# # camera.StartGrabbingMax(numberOfImagesToGrab)
# camera.StartGrabbing()

# while camera.IsGrabbing():
#     grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

#     if grabResult.GrabSucceeded():
#         # Access the image data.
#         print("SizeX: ", grabResult.Width)
#         print("SizeY: ", grabResult.Height)
#         img = grabResult.Array
        
#         img = cv.resize(img, (0, 0),None,0.2,0.2)
#         cv.imshow('basler', img)
#         k = cv.waitKey(50)
#         if k==27:    # Esc key to stop
#             break
#         elif k==-1:  # normally -1 returned,so don't print it
#             continue
#         else:
#             pass
#         print("Gray value of first pixel: ", img[0, 0])

#     grabResult.Release()
# camera.Close()

class Cam():
    def __init__(self):
        self.camera = None
        self.converter = None

    def open(self):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        self.converter = pylon.ImageFormatConverter()
        # converting to opencv bgr format
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    
    def start(self):
        self.camera.StartGrabbing()
    
    def snap(self, count=1):
        img = None
        self.camera.StartGrabbingMax(count)
        while self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                grabResult.Release()
                break
            grabResult.Release()
        self.camera.StopGrabbing()
        return img
    
    def close(self):
        self.camera.Close()

def test():
    cam = Cam()
    cam.open()
    while True:
        img = cam.snap()
        img = cv.resize(img, (0, 0),None,0.3,0.3)
        cv.imshow('basler', img)
        k = cv.waitKey(50)
        if k==27:    # Esc key to stop
            break
        elif k==-1:  # normally -1 returned,so don't print it
            continue
        else:
            pass
    cam.close()

if __name__=='__main__':
    test()
