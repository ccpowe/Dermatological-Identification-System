import os
import numpy as np
import cv2
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from openai import OpenAI

app = Flask(__name__)
CORS(app)  # 解决跨域问题

print("已检测环境、可正常使用")

# Load the feature extractor (VGG16 without the top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
print("VGG16 feature extractor loaded")

# Load the Keras classification model (6claass.h5)
# Use a path relative to the script location, not the current working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "6claass.h5")
print(f"Looking for model at: {model_path}")
assert os.path.exists(model_path), f"Model path does not exist: {model_path}"
classification_model = load_model(model_path)
print("Classification model loaded successfully")

# Define class names for the 6 classes
class_names = [
'Acne and Rosacea Photos',
 'Normal',
 'vitiligo',
 'Tinea Ringworm Candidiasis and other Fungal Infections',
 'Melanoma Skin Cancer Nevi and Moles',
 'Eczema Photos'
]

def preprocess_image(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to RGB color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 180x180 (as used in the notebook)
    img = cv2.resize(img, (180, 180))
    
    # Convert to array and preprocess for VGG16
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def extract_features(img_array):
    # Extract features using VGG16
    features = base_model.predict(img_array)
    
    # Reshape to the expected shape (None, 12800) as seen in the notebook
    num_samples = features.shape[0]
    features_flat = features.reshape(num_samples, -1)  # This should be (1, 12800)
    
    return features_flat

def get_prediction(image_bytes):
    try:
        # Preprocess the image
        preprocessed_img = preprocess_image(image_bytes)
        
        # Extract features using VGG16
        features = extract_features(preprocessed_img)
        
        # Make prediction with the classification model
        predictions = classification_model.predict(features)
        
        # Format results
        results = []
        for i, prob in enumerate(predictions[0]):
            results.append((class_names[i], float(prob)))
        
        # Sort by probability (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Format output
        template = "class:{:<60} probability:{:.3f}"
        text = [template.format(k, v) for k, v in results]
        
        # Using result_model1 key to match frontend expectations
        return_info = {"result_model1": text}
    except Exception as e:
        return_info = {"result_model1": [str(e), str(type(e))]}
    return return_info


@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    print(info)
    return jsonify(info)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("shouye.html")


@app.route("/预测页面", methods=["GET", "POST"])
def AI():
    return render_template("yuce.html")
@app.route("/QA", methods=["GET", "POST"])
def QA():
    return render_template("QA.html")
@app.route('/病种展示')
def bing():
    img_directory = "static/img"
    image_paths = []
    
    # Create directory if it doesn't exist
    if not os.path.exists(img_directory):
        os.makedirs(img_directory)
        print(f"Created directory: {img_directory}")
    
    try:
        for filename in os.listdir(img_directory):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                image_path = os.path.join("img", filename)
                image_paths.append(image_path)
    except Exception as e:
        print(f"Error accessing {img_directory}: {str(e)}")

    return render_template("pfb.html", image_paths=image_paths)
@app.route('/人员展示')
def people():
    img_directory = "static/picture"
    image_paths = []
    
    # Create directory if it doesn't exist
    if not os.path.exists(img_directory):
        os.makedirs(img_directory)
        print(f"Created directory: {img_directory}")
    
    try:
        for filename in os.listdir(img_directory):
            if filename.endswith(".mp4") or filename.endswith(".png"):
                image_path = os.path.join("img", filename)
                image_paths.append(image_path)
    except Exception as e:
        print(f"Error accessing {img_directory}: {str(e)}")

    return render_template("people.html", image_paths=image_paths)
@app.route('/chat')
def chat():
    return render_template('chat.html')


@app.route('/ask', methods=['POST'])
def ask():
    question_text = request.json['question']
    print("Received question:", question_text)

    if not question_text[0].strip():
        return jsonify({'answer': '我是你的健康小助手☺\n\t\t您可以想我咨询你想要了解的健康知识或问题。'})
    
    # Initialize DeepSeek client
    client = OpenAI(api_key="sk-968bf0c", base_url="https://api.deepseek.com")
    
    # Create system prompt for health assistant
    system_prompt = ("你是一个专业的皮肤病健康助手，可以回答用户关于皮肤健康、皮肤病识别、防治的问题。"
                     "记住你只是提供健康和生活建议，不能替代专业医疗诊断。")
    
    try:
        # Call DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text[0]},
            ],
            stream=False
        )
        
        # Extract the answer
        answer = response.choices[0].message.content
        return jsonify({'answer': answer})
        
    except Exception as e:
        print(f"Error calling DeepSeek API: {str(e)}")
        return jsonify({'answer': '抱歉，我现在无法回答您的问题。请稍后再试。'})
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5085)

# 127.0.0.1:5000 进入本地地址进行查看
