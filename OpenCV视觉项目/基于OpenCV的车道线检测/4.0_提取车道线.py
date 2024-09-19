import cv2
import glob
import numpy as np


# 读入图片、预处理图片、检测交点、标定相机的一系列操作
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

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)
    print('Do calibration successfully')
    return ret, mtx, dist, rvecs, tvecs


# 传入计算得到的畸变参数，即可将畸变的图像进行畸变修正处理
def undistortImage(distortImage, mtx, dist):
    return cv2.undistort(distortImage, mtx, dist, None, mtx)


# 透视变换 : Warp image based on src_points and dst_points
# The type of src_points & dst_points should be like
# np.float32([ [0,0], [100,200], [200, 300], [300,400]])
def warpImage(image, src_points, dst_points):
    image_size = (image.shape[1], image.shape[0])
    # rows = img.shape[0] 720
    # cols = img.shape[1] 1280
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped_image = cv2.warpPerspective(
        image, M, image_size, flags=cv2.INTER_LINEAR)

    return warped_image, M, Minv


# Sobel算子提取车道线
def absSobelThreshold(img, orient='x', thresh_min=30, thresh_max=150):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) &
                  (scaled_sobel <= thresh_max)] = 255

    # Return the result
    return binary_output


if __name__ == "__main__":
    nx = 9
    ny = 6

    # ##获取畸变参数
    rets, mtx, dist, rvecs, tvecs = getCameraCalibrationCoefficients(
        'camera_cal/calibration*.jpg', nx, ny)

    # 读取图片
    test_distort_image = cv2.imread('test_img/vlcsnap-2024-09-19-10.jpg')

    # ##畸变修正
    test_undistort_image = undistortImage(test_distort_image, mtx, dist)

    # ##透视变换
    # “不断调整src和dst的值，确保在直线道路上，能够调试出满意的透视变换图像”
    # 左图梯形区域的四个端点
    src = np.float32([[580, 460], [700, 460], [1096, 720], [200, 720]])
    # 右图矩形区域的四个端点
    dst = np.float32([[300, 0], [950, 0], [950, 720], [300, 720]])

    # 变换，得到变换后的结果图，类似俯视图
    test_warp_image, M, Minv = warpImage(test_undistort_image, src, dst)

    # ##提取车道线
    sobel_img = absSobelThreshold(
        test_warp_image, orient='x', thresh_min=30, thresh_max=150)

    # 显示
    # cv2.imshow('img_0', test_warp_image)
    cv2.imshow('img_1', sobel_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
