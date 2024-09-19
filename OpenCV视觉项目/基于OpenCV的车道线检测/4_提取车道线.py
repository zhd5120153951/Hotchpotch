import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


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

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)
    print('Do calibration successfully')
    return ret, mtx, dist, rvecs, tvecs


# Step 2 传入计算得到的畸变参数，即可将畸变的图像进行畸变修正处理
def undistortImage(distortImage, mtx, dist):
    return cv2.undistort(distortImage, mtx, dist, None, mtx)


# Step 3 透视变换 : Warp image based on src_points and dst_points
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


# Step 4 : Create a thresholded binary image
# 亮度划分函数
def hlsLSelect(img, thresh=(220, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    l_channel = l_channel*(255/np.max(l_channel))
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 255
    return binary_output


# 亮度划分函数
def hlsSSelect(img, thresh=(125, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_channel = s_channel*(255/np.max(s_channel))
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 255
    return binary_output


def dirThreshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction_sobelxy = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(direction_sobelxy)
    binary_output[(direction_sobelxy >= thresh[0]) &
                  (direction_sobelxy <= thresh[1])] = 1
    # 6) Return the binary image
    return binary_output


def magThreshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    # 3) Calculate the magnitude
    # abs_sobelx = np.absolute(sobelx)
    # abs_sobely = np.absolute(sobely)
    abs_sobelxy = np.sqrt(sobelx * sobelx + sobely * sobely)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobelxy = np.uint8(abs_sobelxy/np.max(abs_sobelxy) * 255)
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobelxy)
    binary_output[(scaled_sobelxy >= mag_thresh[0]) &
                  (scaled_sobelxy <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def absSobelThreshold(img, orient='x', thresh_min=0, thresh_max=255):
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
    # binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    # binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return scaled_sobel


# Lab蓝黄通道划分函数
def labBSelect(img, thresh=(215, 255)):
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab_b = lab[:, :, 2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 100:
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 255
    # 3) Return a binary image of threshold result
    return binary_output


if __name__ == "__main__":
    nx = 9
    ny = 6

    # Step 1 获取畸变参数
    rets, mtx, dist, rvecs, tvecs = getCameraCalibrationCoefficients(
        'camera_cal/calibration*.jpg', nx, ny)

    # 读取图片
    test_distort_image = cv2.imread('test_img/test3.jpg')

    # Step 2 畸变修正
    test_undistort_image = undistortImage(test_distort_image, mtx, dist)

    # Step 3 透视变换
    # “不断调整src和dst的值，确保在直线道路上，能够调试出满意的透视变换图像”
    # 左图梯形区域的四个端点
    src = np.float32([[580, 440], [700, 440], [1100, 720], [200, 720]])
    # 右图矩形区域的四个端点
    dst = np.float32([[300, 0], [950, 0], [950, 720], [300, 720]])

    # 变换，得到变换后的结果图，类似俯视图
    test_warp_image, M, Minv = warpImage(test_undistort_image, src, dst)

    # Step 4 提取车道线
    # 提取1：sobel算子提取
    # min_thresh = 30
    # max_thresh = 100
    # Result_sobelAbs = absSobelThreshold(test_warp_image, 'x', min_thresh, max_thresh)

    # 提取2：hls方法提取，保留黄色和白色的车道线
    # min_thresh = 150
    # max_thresh = 255
    # Result_hlsS = hlsSSelect(test_warp_image, (min_thresh, max_thresh))

    # 提取3：hls方法提取，只保留白色的车道线
    # min_thresh = 215
    # max_thresh = 255
    # Result_hlsL = hlsLSelect(test_warp_image, (min_thresh, max_thresh))

    # 提取4：lab方法提取，只保留黄色的车道线
    # min_thresh = 195
    # max_thresh = 255
    # Result_labB = labBSelect(test_warp_image, (min_thresh, max_thresh))

    # 提取5：方法1-4结合
    # sx_binary = absSobelThreshold(test_warp_image, 'x', 30, 150)
    # hlsL_binary = hlsLSelect(test_warp_image)
    # labB_binary = labBSelect(test_warp_image)
    # combined_binary = np.zeros_like(sx_binary)
    # combined_binary[(hlsL_binary == 255) | (labB_binary == 255)] = 255

    #  提取6：只要求hsl与lab即可，结果与5一致，省略Sobel算子
    hlsL_binary = hlsLSelect(test_warp_image)
    labB_binary = labBSelect(test_warp_image)
    combined_line_img = np.zeros_like(hlsL_binary)
    combined_line_img[(hlsL_binary == 255) | (labB_binary == 255)] = 255

    # 显示
    cv2.imshow('img_0', combined_line_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
