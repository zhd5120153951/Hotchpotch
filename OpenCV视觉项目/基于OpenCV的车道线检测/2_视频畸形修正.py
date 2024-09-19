import cv2
import glob
import numpy as np


# Step 1 读入图片、预处理图片、检测交点、标定相机的一系列操作
def getCameraCalibrationCoefficients(chessboardname, nx, ny):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(chessboardname)
    if len(images) > 0:
        print("images num for calibration : ", len(images))
    else:
        print("No image for calibration.")
        return

    ret_count = 0
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = (img.shape[1], img.shape[0])
        # Finde the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            ret_count += 1
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    print('Do calibration successfully')
    return ret, mtx, dist, rvecs, tvecs


# Step 2 传入计算得到的畸变参数，即可将畸变的图像进行畸变修正处理
def undistortImage(distortImage, mtx, dist):
    return cv2.undistort(distortImage, mtx, dist, None, mtx)


if __name__ == "__main__":
    nx = 9
    ny = 6

    # Step 1获取畸变参数
    rets, mtx, dist, rvecs, tvecs = getCameraCalibrationCoefficients('camera_cal/calibration*.jpg', nx, ny)

    # 视频输入
    video_input = 'challenge.mp4'
    cap = cv2.VideoCapture(video_input)
    count = 1
    while True:
        ret, image = cap.read()
        if ret:
            undistort_image = undistortImage(image, mtx, dist)
            cv2.imwrite('original_image/' + str(count) + '.jpg', undistort_image)
            count += 1
        else:
            break
    cap.release()

