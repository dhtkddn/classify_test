from keras.models import load_model
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# 📌 **앱 제목 및 소개 추가**
st.title("🍊천혜향 레드향 오렌지지 귤 분류 ")

# 📌 **사이드바 추가**

st.sidebar.markdown("### 📌 사용 방법")
st.sidebar.write("**이미지를 업로드하거나 카메라를 사용하세요.**")


# 📌 **모델 정보 표시**
st.sidebar.markdown("### 🧠 모델 정보")
st.sidebar.write("모델: `keras_model.h5`")
st.sidebar.write("입력 크기: `224x224`")


# Load the model
model = load_model('keras_model.h5', compile=False)

# Load the labels
class_names = open('labels.txt', 'r', encoding='utf-8').readlines()

# 선택 옵션: 카메라 입력 또는 파일 업로드
input_method = st.radio("📷 이미지 입력 방식 선택", ["카메라 사용", "파일 업로드"])

if input_method == "카메라 사용":
    img_file_buffer = st.camera_input("정중앙에 사물을 위치하고 사진찍기 버튼을 누르세요")
else:
    img_file_buffer = st.file_uploader("📂 이미지 파일 업로드", type=["png", "jpg", "jpeg"])

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

if img_file_buffer is not None:
    # 원본 이미지 불러오기
    image = Image.open(img_file_buffer).convert('RGB')

    # 🔹 **업로드한 원본 이미지를 화면에 표시**
    st.image(image, caption="업로드한 이미지", use_container_width=True)

    # 모델에 들어갈 수 있는 224 x 224 사이즈로 변환
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # 이미지를 넘파이 행렬로 변환
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # 빈 ARRAY에 전처리를 완료한 이미지를 복사
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)

    # labels.txt 파일에서 가져온 값을 index로 호출
    class_name = class_names[index].strip()

    # 예측 결과에서 신뢰도를 꺼내 옵니다  
    confidence_score = prediction[0][index]

    # 📌 **결과 표시**
    st.subheader("📊 예측 결과")
    st.write(f"### 🍊 Class: `{class_name}`")
    st.write(f"### ✅ Confidence score: `{confidence_score:.4f}`")

    # 📌 **사이드바에 예측 결과 추가**
    st.sidebar.markdown("### 🔍 예측 결과")
    st.sidebar.write(f"**Class:** {class_name}")
    st.sidebar.write(f"**Confidence:** {confidence_score:.4f}")
