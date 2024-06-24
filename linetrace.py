import numpy as np
import cv2


class Line:
    def __init__(self):
        self.detected = False  # 차선 탐지 여부
        self.window_margin = 32  # 슬라이딩 윈도우 margin 설정
        self.prevx = []  # x값 history
        self.current_fit = [np.array([False])]  # 계수 history
        self.startx = None  # 시작 x좌표
        self.endx = None  # 종료 x좌표
        self.allx = None  # 전체 x좌표 배열
        self.ally = None  # 전체 y좌표 배열


def smoothing(lines, pre_lines=3):
    #  차선 평균 출력
    lines = np.squeeze(lines)
    avg_line = np.zeros(720)

    for ii, line in enumerate(reversed(lines)):
        if ii == pre_lines:
            break
        avg_line += line
    avg_line = avg_line / pre_lines

    return avg_line


def update_line_fit(fity, nonzerox, nonzeroy, line, window_img, color):
    # 2차원 함수로 근사
    fit = np.polyfit(nonzeroy, nonzerox, 2)
    line.current_fit = fit

    # 2차함수 계산
    fitx = fit[0] * fity ** 2 + fit[1] * fity + fit[2]

    # 이전 x 값 리스트에 저장
    line.prevx.append(fitx)

    if len(line.prevx) > 10:
        avg_line = smoothing(line.prevx, 10)
        avg_fit = np.polyfit(fity, avg_line, 2)
        fit_plotx = avg_fit[0] * fity ** 2 + avg_fit[1] * fity + avg_fit[2]
        line.current_fit = avg_fit
        line.allx, line.ally = fit_plotx, fity
    else:
        line.current_fit = fit
        line.allx, line.ally = fitx, fity

    # 시작과 끝 x좌표 업데이트
    line.startx = line.allx[-1]
    line.endx = line.allx[0]

    # 인식된 차선 그리기
    window_img[nonzeroy, nonzerox] = color

    return window_img


def line_search(binary_warped, left_line, right_line):
    # 아래 절반 이미지에 대한 히스토그램
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    output = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # 이미지의 가운데 기준으로 좌우 x좌표 설정
    midpoint = np.int_(histogram.shape[0] // 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # 현재 x좌표 초기화
    current_leftX = left_base
    current_rightX = right_base

    num_windows = 16  # 슬라이딩 윈도우의 개수 설정
    window_height = np.int_(binary_warped.shape[0] // num_windows)  # 슬라이딩 윈도우 높이

    # 색상 및 그래디언트가 인식된 부분 저장
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    minpix = 50  # 윈도우를 재정렬할 최소 임계값

    # 좌우 차선 인덱스를 저장할 리스트
    left_lane_inds = []
    right_lane_inds = []

    margin = left_line.window_margin  # 윈도우 이동 범위

    for window in range(num_windows):  # 윈도우 수 만큼 반복
        # 윈도우 x, y 범위 및 좌표 설정
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = current_leftX - margin
        win_xleft_high = current_leftX + margin
        win_xright_low = current_rightX - margin
        win_xright_high = current_rightX + margin

        # 윈도우 그리기
        cv2.rectangle(output, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(output, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # 윈도우 내부의 차선 인식 부분 설정
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # 각 윈도우 append
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 픽셀 최소 임계값보다 높다면 윈도우 재정렬
        if len(good_left_inds) > minpix:
            current_leftX = np.int_(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            current_rightX = np.int_(np.mean(nonzerox[good_right_inds]))

    # 각 차선 윈도우를 합침
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 검출된 차선 추출
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    # y값 linspace 생성
    fity = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    # 차선을 업데이트하고 인식된 차선 그리기
    output = update_line_fit(fity, leftx, lefty, left_line, output, [255, 0, 0])
    output = update_line_fit(fity, rightx, righty, right_line, output, [0, 0, 255])

    left_line.detected, right_line.detected = True, True

    return output

def prev_window_refer(binary_warped, left_line, right_line):
    # 출력할 output
    output = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # 색상 및 그래디언트가 인식된 부분 저장
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # 슬라이딩 윈도우 margin 설정
    margin = left_line.window_margin

    # 현재 곡선함수 불러옴
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit

    # 윈도우 x, y 범위 및 좌표 설정
    leftx_min = left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2] - margin
    leftx_max = left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2] + margin
    rightx_min = right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2] - margin
    rightx_max = right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2] + margin

    # 검출된 차선 인덱스 저장
    left_inds = ((nonzerox >= leftx_min) & (nonzerox <= leftx_max)).nonzero()[0]
    right_inds = ((nonzerox >= rightx_min) & (nonzerox <= rightx_max)).nonzero()[0]

    # 검출된 픽셀 좌표 저장
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    # y값 linspace 생성
    fity = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    # 차선을 업데이트하고 인식된 차선 그리기
    output = update_line_fit(fity, leftx, lefty, left_line, output, [255, 0, 0])
    output = update_line_fit(fity, rightx, righty, right_line, output, [0, 0, 255])

    # 양쪽 차선의 거리 계산
    standard = np.std(right_line.allx - left_line.allx)

    if (standard > 80):  # 양쪽 차선 거리가 너무 길다면
        left_line.detected = False

    return output


def find_LR_lines(binary_warped, left_line, right_line):

    if not left_line.detected:  # 차선이 인식되지 않은 상태이면
        return line_search(binary_warped, left_line, right_line)

    else:  # 인식된 상태이면
        return prev_window_refer(binary_warped, left_line, right_line)


def draw_lane(img, left_line, right_line, lane_color=(0, 128, 0), road_color=(223, 188, 80)):
    window_img = np.zeros_like(img)  # 출력할 이미지

    margin = left_line.window_margin
    left_fitx, right_fitx = left_line.allx, right_line.allx
    fity = left_line.ally

    # 탐색된 포인트를 fillPoly 함수 형식으로 x, y 변환
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin/5, fity]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin/5, fity])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin/5, fity]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin/5, fity])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    #  차선 두께만큼 색상 채움
    cv2.fillPoly(window_img, np.int_([left_line_pts]), lane_color)
    cv2.fillPoly(window_img, np.int_([right_line_pts]), lane_color)

    #  fillPoly 함수 형식으로 x, y 변환
    pts_left = np.array([np.transpose(np.vstack([left_fitx+margin/5, fity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx-margin/5, fity])))])
    pts = np.hstack((pts_left, pts_right))

    #  양쪽 차선 사이를 다각형으로 색상 채움.
    cv2.fillPoly(window_img, np.int_([pts]), road_color)

    #  반투명하게 가중합
    result = cv2.addWeighted(img, 1, window_img, 0.5, 0)

    return result, window_img