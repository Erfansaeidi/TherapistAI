from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained('./models/trained_model')
tokenizer = AutoTokenizer.from_pretrained('./models/trained_model')

app = Flask(__name__)

@app.route('/message', methods=['POST'])
def message():
    user_message = request.json.get('message')
    
    if not user_message:
        return jsonify({'error': 'Message is required'}), 400
    
    # Tokenize the user message and generate a response from the model
    inputs = tokenizer(user_message, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=50)
    response_message = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({'response': response_message})

if __name__ == '__main__':
    app.run(debug=True)
