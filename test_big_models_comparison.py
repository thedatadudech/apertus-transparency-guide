#!/usr/bin/env python3
"""
🏆 Big Models Swiss German Comparison
Vergleicht die großen Open Source Modelle mit Apertus
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def test_swiss_german_comparison():
    print("🏆 BIG MODELS SWISS GERMAN COMPARISON")
    print("=" * 50)
    
    # Check setup
    if not torch.cuda.is_available():
        print("❌ CUDA required for big models")
        return
        
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 Memory: {gpu_memory:.1f} GB")
    
    if gpu_memory < 35:
        print("⚠️  Warning: Need 35GB+ for all models")
    
    # Big models to compare - using public versions
    models = [
        ("🇨🇭 Apertus-8B", "swiss-ai/Apertus-8B-Instruct-2509"),
        ("🦙 Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct"),  # Access granted
        ("🌸 Mistral-7B", "mistralai/Mistral-7B-Instruct-v0.1"),  # Public version
        ("🌺 BLOOM-7B", "bigscience/bloom-7b1"),
    ]
    
    # Test question in Swiss German
    question = "Grüezi! Chönd Sie mer bitte erchläre was KI isch?"
    
    print(f"\n🎯 Question: {question}")
    print("=" * 50)
    
    results = []
    
    for model_name, model_id in models:
        print(f"\n{'='*60}")
        print(f"🧠 Testing: {model_name}")
        print(f"📦 Model: {model_id}")
        print('='*60)
        
        try:
            # Format prompt for each model
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Model-specific prompting
            if "Apertus" in model_id:
                prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### System:
Du bisch en hilfreiche Schwyzer KI-Assistent. Du verstahsch und redsch flüssig Schweizerdütsch.

### Instruction:
{question}

### Response:
"""
            elif "Llama" in model_id:
                # Llama-3 format (access granted)
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant fluent in Swiss German. Please respond in authentic Schweizerdeutsch.

<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            elif "Mistral" in model_id:
                prompt = f"[INST] Du bisch en hilfreiche Assistent wo Schweizerdütsch redt. Bitte antworte uf Schweizerdütsch:\n\n{question} [/INST]"
            else:  # BLOOM
                prompt = f"Human: Please respond in Swiss German:\n\n{question}\n\nAssistant:"
            
            print(f"📝 Prompt format: {prompt[:60]}...")
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            print(f"🔢 Input tokens: {inputs['input_ids'].shape[1]}")
            
            # Load model
            print("🚀 Loading model...")
            start_load = time.time()
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            load_time = time.time() - start_load
            print(f"✅ Loaded in {load_time:.1f}s")
            
            # Move inputs to model device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            print(f"🎯 Model device: {device}")
            
            # Generate
            print("⚡ Generating...")
            start_gen = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=120,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=1.1
                )
            
            gen_time = time.time() - start_gen
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response[len(prompt):].strip()
            
            # Analyze Swiss German quality
            swiss_indicators = ['isch', 'cha', 'mer', 'chönd', 'gäh', 'hend', 'sind', 'vo', 'uf', 'mit']
            swiss_count = sum(1 for word in swiss_indicators if word in answer.lower())
            
            german_words = ['ist', 'kann', 'mir', 'können', 'geben', 'haben', 'sind', 'von', 'auf', 'mit']
            german_count = sum(1 for word in german_words if word in answer.lower())
            
            results.append({
                'model': model_name,
                'response': answer,
                'swiss_score': swiss_count,
                'german_score': german_count,
                'load_time': load_time,
                'gen_time': gen_time,
                'length': len(answer)
            })
            
            print(f"✅ Generated in {gen_time:.2f}s")
            print(f"📊 Swiss indicators: {swiss_count}, German words: {german_count}")
            print(f"📖 RESPONSE ({len(answer)} chars):")
            print("-" * 50)
            print(answer)
            print("-" * 50)
            
            # Clear memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                'model': model_name,
                'response': f"ERROR: {e}",
                'swiss_score': 0,
                'german_score': 0,
                'load_time': 0,
                'gen_time': 0,
                'length': 0
            })
    
    # Final comparison
    print(f"\n🏆 FINAL COMPARISON")
    print("=" * 60)
    
    # Sort by Swiss German authenticity
    successful = [r for r in results if not r['response'].startswith('ERROR')]
    if successful:
        ranked = sorted(successful, key=lambda x: x['swiss_score'], reverse=True)
        
        print("🥇 RANKING (by Swiss German authenticity):")
        for i, result in enumerate(ranked):
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
            authenticity = "🇨🇭 Authentic" if result['swiss_score'] > result['german_score'] else "🇩🇪 Standard German" if result['german_score'] > result['swiss_score'] else "🤔 Mixed"
            
            print(f"{rank_emoji} {result['model']}: {result['swiss_score']} Swiss indicators, {authenticity}")
            print(f"    Response: {result['response'][:100]}...")
            print()
    
    print("🏁 Comparison complete!")

if __name__ == "__main__":
    test_swiss_german_comparison()