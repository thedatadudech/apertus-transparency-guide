
#!/usr/bin/env python3
"""
🇨🇭 Swiss German AI Model Comparison Script
Test verschiedene Modelle auf ihre Fähigkeit, KI auf Schweizerdeutsch zu erklären
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
import json
from datetime import datetime

def test_model_generation(model_name, model_id, prompt, max_new_tokens=150):
    """Test text generation for a specific model"""
    print(f"\n{'='*60}")
    print(f"🧠 Testing: {model_name}")
    print(f"📦 Model ID: {model_id}")
    print(f"❓ Prompt: {prompt}")
    print('='*60)

    result = {
        "model_name": model_name,
        "model_id": model_id,
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "error": None,
        "response": None,
        "token_count": None,
        "generation_time": None
    }

    try:
        start_time = time.time()

        # Load tokenizer
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Format prompt based on model type
        if "Apertus" in model_id:
            formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### System:
You are a helpful Swiss AI assistant. You understand and speak Swiss German (Schweizerdeutsch) fluently. Please respond in authentic Swiss German when asked.

### Instruction:
{prompt}

### Response:
"""
        elif "Llama" in model_id:
            # Llama-3 format (access granted)
            formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant who can speak Swiss German fluently. When asked to explain something in Swiss German (Schweizerdeutsch), please respond authentically in that dialect.

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        elif "Mistral" in model_id:
            # Mistral format
            formatted_prompt = f"""[INST] You are a helpful assistant who speaks Swiss German. Please respond to the following request in authentic Swiss German (Schweizerdeutsch):

{prompt} [/INST]"""
        elif "bloom" in model_id.lower():
            # BLOOM - simple format with context
            formatted_prompt = f"""Human: Please respond in Swiss German (Schweizerdeutsch):

{prompt}

AI:"""
        elif "german" in model_id.lower():
            # Better prompting for German models
            formatted_prompt = f"""Als hilfreicher Assistent beantworte bitte die folgende Frage ausführlich:

Frage: {prompt}

Antwort:"""
        else:
            # For English models, clarify the task
            if any(swiss_word in prompt.lower() for swiss_word in ['schweiz', 'chönd', 'isch', 'mer']):
                formatted_prompt = f"""Please respond to this Swiss German question by explaining the topic in Swiss German language:

Question: {prompt}

Answer:"""
            else:
                formatted_prompt = prompt

        print(f"📝 Formatted prompt: {formatted_prompt[:100]}...")

        # Tokenize
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True  # Add padding
        )
        input_length = inputs["input_ids"].shape[1]
        result["input_tokens"] = input_length

        print(f"🔢 Input tokens: {input_length}")

        # Load model
        print("🚀 Loading model...")

        # Try different loading strategies based on available hardware
        if torch.cuda.is_available():
            print("🎮 Using CUDA")
            # Use appropriate dtype for each model
            if "Apertus" in model_id:
                torch_dtype = torch.bfloat16
                print("🔧 Using bfloat16 for Apertus compatibility")
            elif any(large_model in model_id for large_model in ["Llama", "Mistral", "bloom"]):
                torch_dtype = torch.bfloat16  # Large modern models prefer bfloat16
                print("🔧 Using bfloat16 for large model compatibility")
            else:
                torch_dtype = torch.float16
                print("🔧 Using float16 for smaller models")
                
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            # Move inputs to same device as model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            print("💻 Using CPU")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )

        # Generate response
        print("⚡ Generating response...")
        generation_start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_length=input_length + max_new_tokens,
                temperature=0.8,  # Bit more creative
                do_sample=True,
                top_p=0.9,  # Nucleus sampling
                top_k=50,   # Limit choices
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.15,  # Stronger repetition penalty
                no_repeat_ngram_size=4   # Longer n-gram blocking
                # Removed early_stopping - not supported by Apertus
            )

        generation_time = time.time() - generation_start

        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = full_response[len(formatted_prompt):].strip()

        result["success"] = True
        result["response"] = response_only
        result["generation_time"] = generation_time
        result["total_tokens"] = outputs[0].shape[0]

        print(f"✅ Generation successful!")
        print(f"⏱️  Time: {generation_time:.2f}s")
        print(f"🔤 Generated tokens: {outputs[0].shape[0] - input_length}")
        print(f"\n📖 ANTWORT:")
        print("-" * 40)
        print(response_only)
        print("-" * 40)

    except Exception as e:
        result["error"] = str(e)
        print(f"❌ Error: {e}")

    return result

def test_tokenization_only(model_name, model_id, prompt):
    """Test only tokenization for large models"""
    print(f"\n{'='*60}")
    print(f"🔍 Tokenization Test: {model_name}")
    print('='*60)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Show different prompt formats
        if "Apertus" in model_id:
            formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### System:
You are a helpful AI assistant that can speak Swiss German.

### Instruction:
{prompt}

### Response:
"""
        else:
            formatted_prompt = prompt

        # Tokenize
        tokens = tokenizer.tokenize(formatted_prompt)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        print(f"📝 Formatted prompt: {formatted_prompt}")
        print(f"🔢 Token count: {len(tokens)}")
        print(f"🎯 Tokens per character: {len(tokens)/len(formatted_prompt):.3f}")
        print(f"🏷️  First 10 tokens: {tokens[:10]}")
        print(f"🔑 First 10 token IDs: {token_ids[:10]}")

        # Check for problematic tokens
        problematic = [t for t in tokens if "Ã" in t or (len(t) == 1 and t.isalpha())]
        if problematic:
            print(f"⚠️  Problematic tokens: {problematic[:5]}")
        else:
            print("✅ No obvious tokenization problems")

        return True

    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return False

def main():
    print("🇨🇭 SWISS GERMAN AI MODEL COMPARISON")
    print("=" * 50)
    print(f"🕐 Started at: {datetime.now()}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"🎮 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎯 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Check HuggingFace login for gated models
    print("\n🔐 Checking HuggingFace Authentication...")
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Logged in as: {user_info['name']}")
    except Exception as e:
        print("⚠️  Not logged in to HuggingFace")
        print("   Gated models (like Apertus) will be skipped")
        print("   Run: huggingface-cli login")

    # Test prompts
    prompts = [
        "Bitte erkläre mir KI auf Schweizerdeutsch",
        "Chönd Sie mer d Künstlichi Intelligänz erchläre?",
        "Was isch KI und wie funktioniert das?"
    ]

    # Models to test (ordered by size - smallest first)
    models = [
        ("🇩🇪 German GPT-2", "dbmdz/german-gpt2"),
        ("🤖 DistilGPT-2 English", "distilgpt2"),
        ("🇩🇪 German BERT (encoder only)", "bert-base-german-cased"),
        ("🦙 Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"),  # Access granted
        ("🌸 Mistral-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.1"),  # Earlier public version
        ("🌺 BLOOM-7B1", "bigscience/bloom-7b1"),
        ("🤖 DialoGPT-Large", "microsoft/DialoGPT-large"),
        ("🇨🇭 Apertus 8B", "swiss-ai/Apertus-8B-Instruct-2509"),
    ]

    all_results = []

    # Test each prompt with each model
    for prompt in prompts:
        print(f"\n🎯 TESTING PROMPT: '{prompt}'")
        print("=" * 80)

        for model_name, model_id in models:
            try:
                if "bert" in model_id.lower():
                    print(f"\n⚠️  Skipping {model_name} (encoder-only model)")
                    continue

                # Check if model needs special handling for size
                large_models = ["Apertus", "Llama", "Mistral", "bloom", "DialoGPT-large"]
                is_large_model = any(large_model in model_id for large_model in large_models)
                
                if is_large_model:
                    # Check GPU memory for large models
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
                    if gpu_memory > 35:  # 35GB+ should handle 7B-8B models
                        print(f"\n🚀 GPU has {gpu_memory:.1f}GB - attempting {model_name} generation!")
                        # Reduce tokens for large models to prevent OOM
                        max_tokens = 80 if "Apertus" in model_id else 100
                        result = test_model_generation(model_name, model_id, prompt, max_new_tokens=max_tokens)
                        all_results.append(result)
                    else:
                        print(f"\n📏 Large model detected: {model_name}")
                        print(f"🔍 GPU only has {gpu_memory:.1f}GB - tokenization only")
                        test_tokenization_only(model_name, model_id, prompt)
                else:
                    # Try full generation for smaller models
                    result = test_model_generation(model_name, model_id, prompt)
                    all_results.append(result)

            except KeyboardInterrupt:
                print("\n⏹️  Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error with {model_name}: {e}")
                continue

    # Save results
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"swiss_german_test_results_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {filename}")

    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 50)
    successful = [r for r in all_results if r["success"]]
    failed = [r for r in all_results if not r["success"]]

    print(f"✅ Successful generations: {len(successful)}")
    print(f"❌ Failed generations: {len(failed)}")

    if successful:
        print(f"\n🏆 BEST RESPONSES:")
        for result in successful:
            print(f"\n🤖 {result['model_name']}:")
            response = result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
            print(f"   '{response}'")

    print(f"\n🏁 Test completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
