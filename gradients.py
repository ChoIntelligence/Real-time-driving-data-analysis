import numpy as np
import cv2
from calibration import calibration
from crop_image import crop_image


def xy_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # 그래디언트 방향 설정
    dx, dy = (1, 0) if orient == 'x' else (0, 1)
    sobel = cv2.Sobel(img, cv2.CV_64F, dx, dy)
    abs_sobel = np.absolute(sobel)  # 미분 결과의 절댓값

    # 임계값 적용
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Sobel x와 y 적용
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    # 스케일링 및 이진화
    scale_factor = np.max(grad_mag) / 255
    grad_mag = (grad_mag / scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(grad_mag)
    binary_output[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1])] = 255
    return binary_output


def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 그래디언트 방향 계산
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # 이진화
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255
    return binary_output


def gradient_ensemble(img, th_x=20, th_y=20, th_mag=90, th_dir=1.3):
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 각 조건에 따른 마스크 적용
    sobelx = xy_thresh(gray, 'x', th_x, 255)
    sobely = xy_thresh(gray, 'y', th_y, 255)
    mag_img = mag_thresh(gray, 3, (0, th_mag))
    dir_img = dir_thresh(gray, 15, (0.7, th_dir))

    # 마스크 조합
    gradient_ensemble = np.zeros_like(dir_img, dtype=np.uint8)
    gradient_ensemble[((sobelx > 1) & (mag_img > 1) & (dir_img > 1)) | ((sobelx > 1) & (sobely > 1))] = 255
    return gradient_ensemble


def update_ensemble(pos=None):
    global images
    th_x = cv2.getTrackbarPos('th_x', 'Control Panel')
    th_y = cv2.getTrackbarPos('th_y', 'Control Panel')
    th_mag = cv2.getTrackbarPos('th_mag', 'Control Panel')
    th_dir = cv2.getTrackbarPos('th_dir', 'Control Panel')

    for idx, img in enumerate(images):
        processed = gradient_ensemble(img, th_x, th_y, th_mag, np.radians(th_dir))
        cv2.imshow(f'Processed Image {idx+1}', processed)


def load_and_process_images(paths):
    images = []
    matrix, dist = calibration()
    for path in paths:
        img = cv2.imread(path)
        img = cv2.undistort(img, matrix, dist, None, matrix)
        img = crop_image(img)
        images.append(img)
    return images


if __name__ == '__main__':
    paths = ['./test_img/yellow_straight.png', './test_img/night_curve.png', './test_img/blue_curve.png']
    images = load_and_process_images(paths)

    # threshold 조정하는 트랙바
    cv2.namedWindow('Control Panel')
    cv2.createTrackbar('th_x', 'Control Panel', 20, 255, update_ensemble)
    cv2.createTrackbar('th_y', 'Control Panel', 20, 255, update_ensemble)
    cv2.createTrackbar('th_mag', 'Control Panel', 90, 255, update_ensemble)
    cv2.createTrackbar('th_dir', 'Control Panel', 75, 180, update_ensemble)  # 각도

    # 원본 이미지 출력
    for idx, img in enumerate(images):
        cv2.imshow(f'Original Image {idx + 1}', img)

    update_ensemble()  # 업데이트 반영

    cv2.waitKey(0)
    cv2.destroyAllWindows()