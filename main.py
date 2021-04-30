import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from PIL import Image
from baslerCam import Cam

def basic_numpy():
    new_list = [1, 2, 3]
    print(type(new_list))
    np_arr = np.array(new_list)
    print(np_arr)
    print(type(np_arr))
    print(np.max(new_list))

    zeros_1D = np.zeros(10)
    print(zeros_1D)
    zeros_2D = np.zeros((10,5))
    print(zeros_2D)
    zeros_3D = np.zeros((3,10,5))
    print(zeros_3D)

    mat = np.random.randint(0,2,100).reshape((10,10))
    print(mat)

    sub = mat[:,2]
    print(sub)

    sub = mat[0,:]
    print(sub)

    sub = mat[0:2,1:3]
    print(sub)

def read_write_image():
    pic = Image.open('data/00-puppy.jpg')
    pic_arr = np.asarray(pic)
    print(pic_arr.shape)
    # plt.imshow(pic_arr)
    # plt.show()

    # sub_img = pic_arr[400:1000, 500:1500, 1]
    # plt.imshow(sub_img)
    # plt.show()

    img = cv.imread('data/00-puppy.jpg')

    pic_arr = cv.resize(pic_arr, (300, 300))
    cv.imshow('Puppy_rbg', pic_arr)
    cv.waitKey(0)

    img = cv.resize(img, (0, 0),None,0.2,0.2)
    cv.imshow('Puppy_bgr', img)
    cv.waitKey(0)

    cv.destroyAllWindows()
    # if image is read from PIL, the channel is RGB, so the image needs to convert channel order to BGR
    new_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(new_img.shape)
    cv.imshow('Puppy_bgr_new', new_img)
    cv.waitKey(0)
    cv.imwrite('my_new_picture.jpg',new_img)


def calibration_realtime():
    col = 9
    row = 6
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((row*col,3), np.float32)
    objp[:,:2] = np.mgrid[0:col,0:row].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    # images = glob.glob('*.jpg')
    cam = Cam()
    cam.open()
    counts = 0
    while True:
        img = cam.snap()
        img_proc = cv.resize(img, (0, 0),None,0.4,0.4)
        
        gray = cv.cvtColor(img_proc, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (col,row), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            # objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            # imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img_proc, (col,row), corners2, ret)
            # gray = cv.resize(gray, (0, 0),None,0.2,0.2)
            # cv.imshow('img', gray)
            
        cv.imshow('img', img_proc)
        k = cv.waitKey(50)
        if k==27:    # Esc key to stop
            break
        elif k==13:  # normally -1 returned,so don't print it
            cv.imwrite(f'data\calibImg\img_{counts}.jpg',img)
            counts += 1
            continue
        else:
            pass

    cv.destroyAllWindows()
    
    cam.close()

def calibration():
    col = 9
    row = 6
    interval = 11 # mm
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((row*col,3), np.float32)
    objp[:,:2] = np.mgrid[0:col,0:row].T.reshape(-1,2) * interval
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('data/calibImg/*.jpg')
    for fname in images:
        print(f"fname: {fname}")
        img = cv.imread(fname)        
        img = cv.resize(img, (0, 0),None,0.4,0.4)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # cv.imshow('gray', gray)
        # cv.waitKey(500)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (col,row), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (col,row), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
    cv.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(f"ret: {ret}")
    print(f"mtx: {mtx}")
    print(f"dist: {dist}")
    print(f"rvecs: {rvecs}")
    print(f"tvecs: {tvecs}")
    return mtx, dist

def undistort(mtx, dist):
    cam = Cam()
    cam.open()
    img = cam.snap()
    img = cv.resize(img, (0, 0),None,0.4,0.4)
    # img = cv.imread('left12.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('calibresult.png', dst)
    cam.close()

def remap(mtx, dist):
    print(f"mtx: {mtx}")
    cam = Cam()
    cam.open()
    img = cam.snap()
    img = cv.resize(img, (0, 0),None,0.4,0.4)
    # img = cv.imread('left12.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    print(f"newcameramtx: {newcameramtx}")
    # undistort
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('calibresult.png', dst)
    cam.close()

def resolution(mtx, dist):
    cam = Cam()
    cam.open()
    img = cam.snap()
    cam.close()
    img = cv.resize(img, (0, 0),None,0.4,0.4)
    # img = cv.imread('left12.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    print(f"newcameramtx: {newcameramtx}")
    # undistort
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    col = 9
    row = 6
    interval = 11 # mm
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((row*col,3), np.float32)
    objp[:,:2] = np.mgrid[0:col,0:row].T.reshape(-1,2) * interval
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (col,row), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(dst, (col,row), corners2, ret)
        
    cv.imshow('img', dst)
    cv.waitKey()
    cv.destroyAllWindows()
    count = 0
    lastObj = None
    lastImg = None
    mmPerPxs = []
    for objPoint, imgPoint in zip(objpoints[0], imgpoints[0]):
        if count == 0:
            lastObj = objPoint
            lastImg = imgPoint
        else:
            distObj = objPoint -lastObj
            distImg = imgPoint[0] -lastImg[0]
            if distObj[0] > 0:
                mmPerPx = distObj[0] / np.sqrt(distImg[0]*distImg[0] + distImg[1]*distImg[1]) * 0.4
                mmPerPxs.append(mmPerPx)
                print(f"distObj: {distObj}")
                print(f"distImg: {distImg}")
                print(f"mm per px: {mmPerPx}")
            lastObj = objPoint
            lastImg = imgPoint
        count += 1
    avgRes = np.average(mmPerPxs)
    stdRes = np.std(mmPerPxs)
    print("")
    print(f"mm per px list: {mmPerPxs}")
    print("")
    print(f"resolution average: {avgRes}")
    print("")
    print(f"resolution std: {stdRes}")
    return avgRes

if __name__=='__main__':
    mtx, dist = calibration()
    resolution(mtx, dist)
