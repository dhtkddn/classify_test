from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# ✅ Load the model (파일 이름 대소문자 주의)
model = load_model("keras_model.h5", compile=False)

# ✅ Load the labels (개행 문자 제거)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

# ✅ 실제 사용할 이미지 경로 입력 (수동 입력 or 사용자 업로드)
image_path = "test_image.jpg"  # <-- 여기에 실제 사용할 이미지 파일 경로 입력
image = Image.open(image_path).convert("RGB")

# ✅ resizing the image to 224x224
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# ✅ turn the image into a numpy array
image_array = np.asarray(image)

# ✅ Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# ✅ Create the array of the right shape
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = normalized_image_array

# ✅ Predict the model
prediction = model.predict(data)
index = np.argmax(prediction)  
class_name = class_names[index]  # 개행 문자 제거된 label 사용
confidence_score = prediction[0][index]

# ✅ Print prediction and confidence score
print("Class:", class_name)
print("Confidence Score:", confidence_score)
