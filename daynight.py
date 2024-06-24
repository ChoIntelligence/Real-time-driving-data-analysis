import cv2
import numpy as np

from gradients import load_and_process_images


def calculate_brightness(image):
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 그레이스케일 이미지의 평균 밝기 계산
    return np.mean(gray)

def update_threshold(val):
    # 트랙바에서 설정한 밝기 임계값
    threshold = cv2.getTrackbarPos('Brightness Threshold', 'Control Panel')
    for idx, img in enumerate(images):
        # 각 이미지의 밝기 계산
        brightness = calculate_brightness(img)
        # 밝기 임계값에 따라 낮 또는 밤으로 분류
        if brightness > threshold:
            classification = 'Day'
        else:
            classification = 'Night'
        # 분류 결과를 창 제목에 표시
        cv2.imshow(f'Image {idx+1} - {classification}', img)


def day_or_night(img):
    # 이미지의 밝기가 60 이상인지 여부를 반환 (60 이상이면 낮, 아니면 밤)
    return calculate_brightness(img) >= 60


if __name__ == '__main__':
    paths = ['./test_img/yellow_straight.png', './test_img/night_curve.png', './test_img/blue_curve.png']
    images = load_and_process_images(paths)

    cv2.namedWindow('Control Panel')
    cv2.createTrackbar('Brightness Threshold', 'Control Panel', 80, 255, update_threshold)

    update_threshold(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
