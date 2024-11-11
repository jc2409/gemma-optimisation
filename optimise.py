import contextlib
import os
import re
import torch
import time
import copy
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch.nn.utils.prune as prune

# Constants
VARIANT = "2b-it"  # Model Variant
weights_dir = f"google/gemma-{VARIANT}"  # Hugging Face model identifier

# Hugging Face Login
# Replace 'YOUR_HF_TOKEN' with your actual Hugging Face token
login("YOUR_HF_TOKEN", add_to_git_credential=True)

# Configure default tensor type
@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

# Load the Model Configuration and Weights
tokenizer = AutoTokenizer.from_pretrained(weights_dir)

num_gpus = torch.cuda.device_count()

if num_gpus > 1:
    print(f"Using {num_gpus} GPUs")
    model = AutoModelForCausalLM.from_pretrained(weights_dir, torch_dtype=torch.bfloat16, device_map="auto")
else:
    print("Using single CPU or GPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    model = model.to(device)

model = model.eval()

USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"

# Pruning the Model
def prune_model(model, amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

# Apply pruning to the model
pruned_model = prune_model(model, amount=0.3)

# Run inference
def chat_with_gemma(prompt, selected_model):
    conversation = USER_CHAT_TEMPLATE.format(prompt=prompt) + "<start_of_turn>model\n"
    input_ids = tokenizer.encode(conversation, return_tensors='pt').to(model.device)
    start_time = time.time()
    output = selected_model.generate(input_ids, max_length=100)  # Adjust max_length as needed
    end_time = time.time()
    latency = end_time - start_time
    results = tokenizer.decode(output[0], skip_special_tokens=True)
    model_responses = re.findall(r'model\n(.*?)(?=(?:\nuser|$))', results, re.DOTALL)[0]
    return model_responses, latency


# Function to get adjusted model size
def get_model_size(model, bit=32): 
    torch.save(model.state_dict(), "temp.p") 
    size_in_bytes = os.path.getsize("temp.p")
    size_in_mb = size_in_bytes / 1e6  
    adjusted_size = size_in_mb * (bit / 32.0) 
    os.remove("temp.p")
    return adjusted_size

# Main execution
if __name__ == "__main__":
    # Example usage
    prompt = "Hello! How are you?"

    print(f"User: {prompt}\n")
    
    
    # Evaluate Standard Model
    model_response, latency = chat_with_gemma(prompt, model)
    print(f"Gemma (Standard): {model_response}")
    print(f"Response Latency: {latency:.2f} seconds\n")
    original_size = get_model_size(model, 32)
    print(f"Original model size: {original_size:.2f} MB")

    # Evaluate Pruned Model
    model_response, latency = chat_with_gemma(prompt, pruned_model)
    print(f"Gemma (Pruned): {model_response}")
    print(f"Response Latency: {latency:.2f} seconds\n")
    pruned_size = get_model_size(pruned_model, 32)
    print(f"Pruned model size: {pruned_size:.2f} MB")

    model = AutoModelForCausalLM.from_pretrained(weights_dir, torch_dtype=torch.bfloat16)
    model = model.to("cpu")
    model = model.eval()
    quantized_model = torch.quantization.prepare(model, inplace=True)
    quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)
    quantized_size = get_model_size(quantized_model,8)
    print(f"Quantized model size: {quantized_size:.2f} MB")
    # Evaluate Quantized Model
    model_response, latency = chat_with_gemma(prompt, quantized_model)
    print(f"Gemma (Quantized): {model_response}")
    print(f"Response Latency: {latency:.2f} seconds\n")

    # Print model sizes
    original_size = get_model_size(model, 32)
    pruned_size = get_model_size(pruned_model, 32)
    quantized_size = get_model_size(quantized_model, 8)


    print('original model size: {:.3f}MB'.format(original_size))
    print('pruned model size: {:.3f}MB'.format(pruned_size))
    print('quantized model size: {:.3f}MB'.format(quantized_size))