import cv2
from calibration import calibration


def update_lines(val=0): # 트랙바 위치에 따라 선을 그리고 화면을 업데이트하는 함수
    frame_copy = frame.copy()
    # 두 트랙바에서 값 가져오기
    line1_pos = cv2.getTrackbarPos('Upper Line', 'Frame')
    line2_pos = cv2.getTrackbarPos('Lower Line', 'Frame')
    # 두 수평선 그리기
    cv2.line(frame_copy, (0, line1_pos), (frame.shape[1], line1_pos), (0, 0, 255), 2)
    cv2.line(frame_copy, (0, line2_pos), (frame.shape[1], line2_pos), (0, 0, 255), 2)

    cv2.imshow('Frame', frame_copy)


def crop_image(frame, upper_line = 680, lower_line = 952):
    return frame[upper_line:lower_line, :]


if __name__ == '__main__' :
    matrix, dist = calibration()
    video = './test_videos/night_curve.mp4'

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, 500)
    ret, frame = cap.read()  # 첫 프레임 읽기

    frame = cv2.undistort(frame, matrix, dist, None, matrix)

    # 윈도우 생성
    cv2.namedWindow('Frame')
    # 트랙바 생성
    cv2.createTrackbar('Upper Line', 'Frame', 0, frame.shape[0] - 1, update_lines)
    cv2.createTrackbar('Lower Line', 'Frame', 0, frame.shape[0] - 1, update_lines)

    update_lines()

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

    cropped_frame = crop_image(frame)
    cv2.imshow('Cropped Frame', cropped_frame)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
