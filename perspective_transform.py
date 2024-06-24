import cv2
import numpy as np

from calibration import calibration
from crop_image import crop_image


def apply_perspective_transform(img, top_x=100, bottom_x=600):
    rows, cols = img.shape[:2]
    # src
    src_pts = np.float32([
        [cols // 2 - top_x, 0],  # 좌상단
        [cols // 2 + top_x, 0],  # 우상단
        [cols // 2 - bottom_x, rows - 1],  # 좌하단
        [cols // 2 + bottom_x, rows - 1]  # 우하단
    ])
    # dst
    dst_pts = np.float32([
        [0, 0],  # 좌상단
        [cols - 1, 0],  # 우상단
        [0, rows - 1],  # 좌하단
        [cols - 1, rows - 1]  # 우하단
    ])

    # 원근 변환 행렬 계산 및 적용
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    transformed_img = cv2.warpPerspective(img, matrix, (cols, rows))
    transformed_img = cv2.resize(transformed_img, (720, 720))

    return transformed_img, Minv


def update_perspective(val=0):
    top_x = cv2.getTrackbarPos('Top X', 'Original Image')
    bottom_x = cv2.getTrackbarPos('Bottom X', 'Original Image')

    rows, cols = img.shape[:2]
    src_pts = np.float32([
        [cols // 2 - top_x, 0],
        [cols // 2 + top_x, 0],
        [cols // 2 + bottom_x, rows - 1],
        [cols // 2 - bottom_x, rows - 1]
    ])
    dst_pts = np.float32([
        [0, 0],
        [cols - 1, 0],
        [cols - 1, rows - 1],
        [0, rows - 1]
    ])

    # 원근 변환 적용
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(img, matrix, (cols, rows))

    # 원본 이미지에 빨간색 점과 선을 업데이트
    img_with_points = img.copy()
    # 빨간색 선 그리기
    src_pts_int = src_pts.astype(int)
    cv2.polylines(img_with_points, [src_pts_int], isClosed=True, color=(0, 0, 255), thickness=2)

    # 모서리에 빨간색 원 그리기
    for x, y in src_pts_int:
        cv2.circle(img_with_points, (x, y), 5, (0, 0, 255), -1)

    cv2.imshow('Original Image', img_with_points)
    cv2.imshow('Perspective Transform', result)


if __name__ == "__main__":
    matrix, dist = calibration()
    img = cv2.imread('./test_img/perspective.png')
    img = cv2.undistort(img, matrix, dist, None, matrix)
    img = crop_image(img)

    cv2.namedWindow('Original Image')
    cv2.createTrackbar('Top X', 'Original Image', 50, img.shape[1] // 2, update_perspective)
    cv2.createTrackbar('Bottom X', 'Original Image', 200, img.shape[1] // 2, update_perspective)

    update_perspective()
    cv2.waitKey(0)

    cv2.destroyAllWindows()
