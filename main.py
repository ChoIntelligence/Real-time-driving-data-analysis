import numpy as np
import cv2
from calibration import calibration, undistort
from crop_image import crop_image
from daynight import day_or_night
from gradients import gradient_ensemble
from colours import lab_essemble
from ensemble import ensemble_result
from linetrace import Line, find_LR_lines, draw_lane
from perspective_transform import apply_perspective_transform
from yolo_vehicles import detect_and_draw_boxes, initialize_yolo

video = './original_videos/day_blue.mp4'  # 비디오 파일 경로

matrix, dist = calibration()  # 카메라 보정 매트릭스와 왜곡 계수
left_line, right_line = Line(), Line()  # 좌우 차선 라인 객체 초기화
net, classes, output_layers = initialize_yolo()  # YOLO 초기화

# 비디오 출력 설정
cap = cv2.VideoCapture(video)
ret, frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (frame.shape[1], frame.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./output_videos/output.mp4', fourcc, fps, frame_size)


if __name__ == '__main__':
    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        # 왜곡 보정
        undist_img = undistort(frame, matrix, dist)
        # 객체 탐지 및 바운딩 박스 그리기
        detected_img = detect_and_draw_boxes(undist_img, net, classes, output_layers)

        detected_img_crop = crop_image(detected_img)
        result_img = undist_img.copy()

        undist_img = crop_image(undist_img)
        day = day_or_night(undist_img)  # 낮/밤 판단

        rows, cols = undist_img.shape[:2]

        # 그래디언트, LAB 색상 마스킹
        ensemble_gradient = gradient_ensemble(undist_img)  # 그래디언트 기반 마스크 생성
        ensemble_lab = lab_essemble(undist_img, day)  # LAB 색상 기반 마스크 생성
        ensembled_result = ensemble_result(ensemble_gradient, ensemble_lab)  # 마스크 결합

        c_rows, c_cols = ensembled_result.shape[:2]

        # 차선 탐지 및 그리기
        warp_img, Minv = apply_perspective_transform(ensembled_result)  # 원근 변환 적용
        searching_img = find_LR_lines(warp_img, left_line, right_line)  # 차선 탐지
        w_comb_result, w_color_result = draw_lane(searching_img, left_line, right_line)  # 차선 그리기

        # 도로 위에 탐지된 차선 이미지 적용
        w_color_result = cv2.resize(w_color_result, (cols, rows))
        color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))

        # 차선 이미지 반투명하게 적용
        lane_color = np.zeros_like(undist_img)
        lane_color[:] = color_result
        result = cv2.addWeighted(detected_img_crop, 1, lane_color, 0.5, 0)

        # 객체 탐지 영상에 차선 탐지 이미지를 함께 출력
        searching_img = cv2.resize(searching_img, (360, 360))
        detected_img[100:460, -460:-100] = searching_img

        # crop 범위를 차선 이미지로 교체
        detected_img[680:952, :] = result.astype(np.uint8)

        # 최종 이미지 출력
        cv2.imshow('result', detected_img)
        out.write(detected_img)  # 출력 비디오 파일에 프레임 쓰기

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
