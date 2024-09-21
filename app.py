import streamlit as st
import numpy as np
import cv2
import openvino as ov
from pathlib import Path

# OpenVINO 모델 초기화
base_model_dir = Path("model")
detection_model_name = "vehicle-detection-0200"
precision = "FP32"
base_model_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1"

# 모델 다운로드 함수
def download_model(model_url, model_name):
    response = requests.get(model_url)
    with open(base_model_dir / f"{model_name}.xml", 'wb') as f:
        f.write(response.content)
    response_bin = requests.get(model_url.replace('.xml', '.bin'))
    with open(base_model_dir / f"{model_name}.bin", 'wb') as f_bin:
        f_bin.write(response_bin.content)

# OpenVINO 모델 로드
core = ov.Core()
detection_model = core.read_model(model=str(base_model_dir / f"{detection_model_name}.xml"))
compiled_model = core.compile_model(detection_model, "CPU")

# Streamlit 앱 시작
st.title('Vehicle Detection with OpenVINO')

# 파일 업로드 기능
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 파일을 OpenCV 형식으로 변환
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # 이미지가 로드되었는지 확인
    if image is not None:
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # OpenVINO 모델을 사용하여 감지 실행
        resized_image = cv2.resize(image, (256, 256))
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
        output = compiled_model([input_image])[compiled_model.output(0)]
        output = np.squeeze(output)

        # 자동차 감지 및 윤곽선 그리기
        detected_vehicles = 0
        for detection in output:
            conf = detection[2]
            if conf > 0.5:  # 신뢰도 임계값
                xmin, ymin, xmax, ymax = int(detection[3] * image.shape[1]), int(detection[4] * image.shape[0]), int(detection[5] * image.shape[1]), int(detection[6] * image.shape[0])
                detected_vehicles += 1
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 결과 표시
        st.image(image, caption=f'Detected {detected_vehicles} vehicles', use_column_width=True)
    else:
        st.error("Error loading the image.")