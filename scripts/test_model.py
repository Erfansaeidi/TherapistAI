from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the saved model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./saved_model")
tokenizer = AutoTokenizer.from_pretrained("./saved_model")

# Example message to the model
user_input = "Hello, how are you?"

inputs = tokenizer(user_input, return_tensors="pt")
outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)

# Decode the generated response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
