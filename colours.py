import cv2
import numpy as np
from gradients import load_and_process_images


def apply_threshold(channel, thresh_min, thresh_max):
    # 채널의 값을 설정한 범위로 이진화
    binary_output = np.zeros_like(channel)
    binary_output[(channel >= thresh_min) & (channel <= thresh_max)] = 255
    return binary_output


def lab_combine(img, l_range, a_range, b_range):
    # 이미지를 Lab 색상 공간으로 변환
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    # L, a, b 채널 각각에 대해 이진화 적용
    l_binary = apply_threshold(lab[:, :, 0], l_range[0], l_range[1])
    a_binary = apply_threshold(lab[:, :, 1], a_range[0], a_range[1])
    b_binary = apply_threshold(lab[:, :, 2], b_range[0], b_range[1])

    # 이진화된 채널들을 결합
    combined_binary = cv2.bitwise_and(l_binary, a_binary)
    combined_binary = cv2.bitwise_and(combined_binary, b_binary)

    return combined_binary


def lab_essemble(img, daytime):
    # 주간과 야간에 따라 다른 범위로 이진화된 이미지를 결합
    if daytime:
        white = lab_combine(img, (190, 255), (115, 130), (128, 140))
        yellow = lab_combine(img, (150, 255), (100, 150), (150, 200))
        blue = lab_combine(img, (149, 255), (90, 119), (100, 130))
        ensembled = cv2.bitwise_or(white, yellow)
        ensembled = cv2.bitwise_or(ensembled, blue)
    else:
        ensembled = lab_combine(img, (80, 160), (115, 128), (100, 150))

    return ensembled


def update_images(val=None):
    # 트랙바에서 설정한 값을 가져와서 이미지 업데이트
    l_min = cv2.getTrackbarPos('L Min', 'Control Panel')
    l_max = cv2.getTrackbarPos('L Max', 'Control Panel')
    a_min = cv2.getTrackbarPos('A Min', 'Control Panel')
    a_max = cv2.getTrackbarPos('A Max', 'Control Panel')
    b_min = cv2.getTrackbarPos('B Min', 'Control Panel')
    b_max = cv2.getTrackbarPos('B Max', 'Control Panel')

    # 이진화 결과 출력
    for idx, img in enumerate(images):
        processed = lab_combine(img, (l_min, l_max), (a_min, a_max), (b_min, b_max))
        cv2.imshow(f'Processed Image {idx + 1}', processed)


if __name__ == '__main__':
    paths = ['./test_img/yellow_straight.png', './test_img/night_curve.png', './test_img/blue_curve.png']
    images = load_and_process_images(paths)

    # 임계값 조정을 위한 제어판 설정
    cv2.namedWindow('Control Panel')
    cv2.createTrackbar('L Min', 'Control Panel', 0, 255, update_images)
    cv2.createTrackbar('L Max', 'Control Panel', 255, 255, update_images)
    cv2.createTrackbar('A Min', 'Control Panel', 0, 255, update_images)
    cv2.createTrackbar('A Max', 'Control Panel', 255, 255, update_images)
    cv2.createTrackbar('B Min', 'Control Panel', 0, 255, update_images)
    cv2.createTrackbar('B Max', 'Control Panel', 255, 255, update_images)

    # 원본 이미지 표시
    for idx, img in enumerate(images):
        cv2.imshow(f'Original Image {idx + 1}', img)

    # 처리된 이미지 업데이트
    update_images()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
