import os
import shutil
import torch
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def get_nested_attr(obj, attr_str):
    attrs = attr_str.split('.')
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

model = GPT2LMHeadModel.from_pretrained("gpt2")
print("Model loaded.")

layers = model.transformer.h  # LMHeadModel has transformer attribute

layer_idx = random.randint(0, len(layers) - 1)
layer = layers[layer_idx]

print(f"Selected layer index: {layer_idx}")

params = [(name, param) for name, param in layer.named_parameters() if param.ndim == 2]

if not params:
    print("No 2D matrix parameters found in this layer.")
else:
    param_name, param_tensor = random.choice(params)

    print(f"Modifying parameter: {param_name} with shape {param_tensor.shape}")

    with torch.no_grad():
        param_tensor.copy_(torch.randn_like(param_tensor))

print(f"Modified parameter tensor:\n{get_nested_attr(layer, param_name)}")

output_dir = "output"
if os.path.exists(output_dir):
    # makes sure loading zone is empty
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
else:
    os.makedirs(output_dir)

# save the entire model weights
model.save_pretrained(output_dir)
print(f"Model weights saved to {output_dir}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model.eval()

# use cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("\nYou can now type prompts to generate text. Type 'exit' to quit.")

while True:
    prompt = input("\nEnter prompt: ")
    if prompt.lower() in {"exit", "quit"}:
        print("Exiting.")
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    print(f"Response: {generated_text}")