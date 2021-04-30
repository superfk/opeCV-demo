import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json, os, datetime, math
from numpy.linalg import inv


def read_label(path = 'template.json'):
    with open(path, 'r') as jsonfile:
        data = json.load(jsonfile)
    return data['shapes']

def plot_polygon(img, labels, color=255):
    for label in labels:
        dst = label['points']
        img = cv.polylines(img,[np.int32(dst)],True,color,3, cv.LINE_AA)
    return img

def read_image(path):
    imag = cv.imread(path,cv.IMREAD_COLOR)
    # imag = cv.cvtColor(imag, cv.COLOR_BGR2GRAY)
    return imag

def show_img(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img,),plt.show()

def get_target_rect(img, H, labels, color=10):
    for label in labels:
        pts = np.float32(label['points']).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,H)
        img = cv.polylines(img,[np.int32(dst)],True,color,3, cv.LINE_AA)
    return img

def cut_and_save(img, H, labels, folder):
    for label in labels:
        pts = np.float32(label['points']).reshape(-1,1,2)
        pts = cv.perspectiveTransform(pts,H)

        ## (1) Crop the bounding rect
        rect = cv.boundingRect(pts)
        x,y,w,h = rect
        croped = img[y:y+h, x:x+w].copy()

        ## (2) make mask
        pts = pts - pts.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        # cv.drawContours(mask, [pts], -1, 5, -1, cv.LINE_AA)

        ## (3) do bit-op
        dst = cv.bitwise_and(croped, croped, mask=mask)
        targetFolder = os.path.join(folder, label['label'])
        fileName = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        targetPath = os.path.join(targetFolder, f"{label['label']}_{fileName}.jpg")
        if not os.path.exists(targetFolder):
            os.makedirs(targetFolder)
        cv.imwrite(targetPath, croped)

def getFeatures(img, tempImg):
    MIN_MATCH_COUNT = 10
    img1 = tempImg
    img2 = img

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.5*n.distance:
            matchesMask[i]=[1,0]
            good.append(m)
    
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        try:
            H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            theta = - math.atan2(H[0,1], H[0,0]) * 180 / math.pi
            print(f"H: {H}")
            # print(f"theta: {theta}")
            matchesMask = mask.ravel().tolist()
            h,w,c = img1.shape
            # cut_and_save(img2, H, labels, r'D:\\03_Coding\\Vision\\invoiceMatch\\crop')
            # img2 = get_target_rect(img2, H, labels, 10 )
            # show_img(img2)
            pts = np.float32([ [0,0],[0,h-1, ],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,H)
            img3 = cv.polylines(img2,[np.int32(dst)],True,(0,0,255),3, cv.LINE_AA)
            # M = cv.getPerspectiveTransform(dst, pts)
            H_inv = inv(H)
            theta = - math.atan2(H_inv[0,1], H_inv[0,0]) * 180 / math.pi
            print(f"M_inv: {H_inv}")
            # print(f"theta: {theta}")
            warped = cv.warpPerspective(img3, H_inv, (w, h))
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
            img4 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
            # show_img(img4)
            cv.imshow('img4', img4)
        except:
            return img, tempImg
        return img3, warped
    else:
        return img, tempImg

def realtime():
    tempImage = read_image(r'data\template.jpg')
    cap = cv.VideoCapture(0)
    while(True):
        # 從攝影機擷取一張影像
        ret, frame = cap.read()
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        feature, warped = getFeatures(frame,tempImage )

        # 顯示圖片
        cv.imshow('frame', feature)
        cv.imshow('warped', warped)

        # 若按下 q 鍵則離開迴圈
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放攝影機
    cap.release()

    # 關閉所有 OpenCV 視窗
    cv.destroyAllWindows()


if __name__=='__main__':
    realtime()