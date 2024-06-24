import numpy as np
import cv2
import glob

images = glob.glob('./calibration_img/*.png')

points_3d = []  # 실제 point
points_2d = []  # 2D point


def calibration() :
    obj_points = np.zeros((6 * 9, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) # x, y 좌표

    for pngfile in images :
        img = cv2.imread(pngfile)
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # chessboard의 코너를 찾음
        ret, corners = cv2.findChessboardCorners(grayscale, (9, 6), None)

        # 코너가 검출되었다면 각 리스트에 append
        if ret:
            points_2d.append(corners)
            points_3d.append(obj_points)
        else:
            continue

    # 왜곡계수 계산
    ret, matrix, dist, _, _ = cv2.calibrateCamera(points_3d, points_2d,
                                                      grayscale.shape[::-1], None, None)

    return matrix, dist


def undistort(img, matrix, dist) :
    return cv2.undistort(img, matrix, dist, None, matrix)


if __name__ == '__main__' :
    matrix, dist = calibration()  # 보정 행렬, 거리 정보 반환
    video = './original_videos/day_yellow.mp4'

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, 500)
    ret, frame = cap.read()  # 첫 프레임 읽기

    # 변환 행렬과 거리 정보로 왜곡 보정
    undistorted_frame = cv2.undistort(frame, matrix, dist, None, matrix)
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Undistorted Frame', undistorted_frame)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
