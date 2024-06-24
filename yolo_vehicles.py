import cv2
import numpy as np
# https://github.com/eric612/Vehicle-Detection/tree/master
# 다크넷 trained model 사용


def initialize_yolo():  # YOLO 모델과 클래스 이름 초기화
    #  훈련된 모델 불러오기
    net = cv2.dnn.readNet("./models/yolov3-tiny.weights", "./models/yolov3-tiny.cfg")

    #  클래스 이름 불러오기
    with open("./models/voc.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers


def detect_and_draw_boxes(img, net, classes, output_layers):
    height, width, channels = img.shape

    # 모델에 입력할 블롭 객체 준비
    blob = cv2.dnn.blobFromImage(img, 1./255, (416, 416), (0, 0, 0), True, crop=False)
    # 모델에 블롭 객체 전달
    net.setInput(blob)

    # 이미지 추론
    outs = net.forward(output_layers)

    # 출력 파싱
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # 원본 이미지 크기로 다시 스케일링
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # 바운딩박스, 객체 신뢰도, 객체 이름 저장
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 비최대 억제 적용
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 각 클래스에 대한 색상 및 레이블 설정
    colors_labels = {
        'bicycle': (0, 255, 0, 'Bicycle'),
        'car': (255, 0, 0, 'Vehicle'),
        'motorbike': (0, 0, 255, 'Motorbike'),
        'person': (0, 0, 255, 'Pedestrian'),
        'cones': (255, 0, 255, 'Cones')
    }

    # 이미지에 바운딩 박스 그리기
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label in colors_labels:  # 지정된 클래스만 바운딩 박스 그리기
                color, display_label = colors_labels[label][:3], colors_labels[label][3]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # 바운딩 박스 그리기
                cv2.putText(img, display_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 텍스트 출력

    return img
