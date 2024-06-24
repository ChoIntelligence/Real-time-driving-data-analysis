import cv2
import numpy as np
from colours import lab_essemble
from daynight import day_or_night
from gradients import gradient_ensemble, load_and_process_images


def ensemble_grad_color(img, daytime):
    ensembled_img = np.zeros_like(img)

    # Lab 마스크 값이 1보다 큰 위치에 빨간색 설정
    mask = lab_essemble(img, daytime)
    ensembled_img[mask > 1] = [255, 0, 0]

    # 그래디언트 마스크 값이 1보다 큰 위치에 파란색 설정
    mask = gradient_ensemble(img)
    ensembled_img[mask > 1] = [0, 0, 255]

    return ensembled_img


def ensemble_result(grad, lab):
    result = np.zeros_like(lab).astype(np.uint8)

    # 그래디언트 값이 1보다 큰 위치에 255 설정
    result[(grad > 1)] = 100
    # lab 값이 1보다 큰 위치에 100 설정
    result[(lab > 1)] = 255

    return result


if __name__ == '__main__':
    paths = ['./test_img/yellow_straight.png', './test_img/night_curve.png', './test_img/blue_curve.png']
    images = load_and_process_images(paths)

    for idx, img in enumerate(images):
        processed_img = ensemble_grad_color(img, day_or_night(img))

        cv2.imshow(f'Original Image {idx + 1}', img)
        cv2.imshow(f'Processed Image {idx + 1}', processed_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

