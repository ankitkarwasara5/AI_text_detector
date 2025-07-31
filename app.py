import torch
import torch.nn.functional as F
# Add render_template to the imports
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer

from model import BERTClassifier

# --- Configuration --- (No changes here)
MODEL_PATH = 'best_model_state_v2.bin'
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_TOKEN_LEN = 512 

# --- Initialization --- (No changes here)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = BERTClassifier(n_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval() 
app = Flask(__name__)

# --- Prediction Logic --- (No changes here)
def predict_text(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_TOKEN_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        prediction = 'AI' if predicted_class.item() == 1 else 'Human'
        confidence_score = confidence.item()
    
    return prediction, confidence_score

# --- API Endpoints ---
# THIS IS THE ONLY PART THAT CHANGES
@app.route('/')
def index():
    # Serve the index.html file
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text_to_analyze = data['text']
        
        if not text_to_analyze:
            return jsonify({"error": "No text provided"}), 400
            
        prediction, confidence = predict_text(text_to_analyze)
        
        return jsonify({
            "prediction": prediction,
            "confidence": f"{confidence:.4f}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Run the App --- (No changes here)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)