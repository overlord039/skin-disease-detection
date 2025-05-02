import torch
import torchvision.models as models
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- Added CORS
from PIL import Image
from torchvision import transforms
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # <-- Enable CORS for all routes

# Decode classes for skin diseases
decode_classes = {
    0: "Acne and Rosacea",
    1: "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    2: "Atopic Dermatitis",
    3: "Bullous Disease",
    4: "Cellulitis Impetigo and other Bacterial Infections",
    5: "Eczema",
    6: "Exanthems and Drug Eruptions",
    7: "Hair Loss Photos Alopecia and other Hair Diseases",
    8: "Herpes HPV and other STDs Photos",
    9: "Light Diseases and Disorders of Pigmentation",
    10: "Lupus and other Connective Tissue diseases",
    11: "Melanoma Skin Cancer Nevi and Moles",
    12: "Nail Fungus and other Nail Disease",
    13: "Poison Ivy Photos and other Contact Dermatitis",
    14: "Psoriasis pictures Lichen Planus and related diseases",
    15: "Scabies Lyme Disease and other Infestations and Bites",
    16: "Seborrheic Keratoses and other Benign Tumors",
    17: "Systemic Disease",
    18: "Tinea Ringworm Candidiasis and other Fungal Infections",
    19: "Urticaria Hives",
    20: "Vascular Tumors",
    21: "Vasculitis",
    22: "Warts Molluscum and other Viral Infections"
}

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
n_classes = len(decode_classes)
model.fc = nn.Linear(model.fc.in_features, n_classes)
model.to(device)

# Load the trained model weights
model.load_state_dict(torch.load(r"C:\Users\91863\Documents\skin-disease-detector\project\best_model.pth", map_location=device))
model.eval()

# Image preprocessing
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Make prediction
def predict_image(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        _, predicted = torch.max(outputs, 1)
        return decode_classes[predicted.item()]

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    image_tensor = preprocess_image(file_path)
    predicted_disease = predict_image(image_tensor)

    return jsonify({
        'prediction': predicted_disease,
        'confidence': 1.0  # Placeholder confidence value if not using softmax
    }), 200

if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
