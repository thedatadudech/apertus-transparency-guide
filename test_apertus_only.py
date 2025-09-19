#!/usr/bin/env python3
"""
🇨🇭 Apertus Swiss German Test
Fokussiert nur auf Apertus Model Testing
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def test_apertus_swiss_german():
    print("🇨🇭 APERTUS SWISS GERMAN TEST")
    print("=" * 40)
    
    model_id = "swiss-ai/Apertus-8B-Instruct-2509"
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available - Apertus needs GPU")
        return
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 Memory: {gpu_memory:.1f} GB")
    
    if gpu_memory < 20:
        print("⚠️  Warning: Low GPU memory for Apertus-8B")
    
    # Swiss German test questions
    questions = [
        "Grüezi! Chönd Sie mer bitte erchläre was KI isch?",
        "Wie funktioniert Künstlichi Intelligänz?", 
        "Was sind d Vorteile und Nochteile vo KI?",
        "Chönd Sie mer es Bispiil vo KI im Alldag gäh?"
    ]
    
    try:
        print("\n📥 Loading Apertus tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("🚀 Loading Apertus model...")
        # Use bfloat16 to match the model's internal expectations
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # Changed from float16
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        print(f"✅ Model loaded on: {next(model.parameters()).device}")
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*60}")
            print(f"📝 Question {i}: {question}")
            print('='*60)
            
            # Format with Swiss German system prompt
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### System:
Du bisch en hilfreiche Schwyzer KI-Assistent. Du verstahsch und redsch flüssig Schweizerdütsch. Bitte antworte uf Schweizerdütsch wänn du drüm bete wirst.

### Instruction:
{question}

### Response:
"""
            
            print(f"🔢 Tokenizing...")
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            print(f"⚡ Generating... (Input: {inputs['input_ids'].shape[1]} tokens)")
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=1.1
                    # Removed early_stopping - not supported by this model
                )
            
            generation_time = time.time() - start_time
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            
            print(f"✅ Generated in {generation_time:.2f}s")
            print(f"📖 ANTWORT:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
            # Analyze response quality
            swiss_indicators = sum(1 for word in ['isch', 'mer', 'chönd', 'gäh', 'wänd', 'hend', 'sind', 'bin'] 
                                 if word in response.lower())
            german_words = sum(1 for word in ['ist', 'mir', 'können', 'geben', 'wollen', 'haben', 'sind', 'bin'] 
                              if word in response.lower())
            
            print(f"🔍 Analysis:")
            print(f"   Swiss German indicators: {swiss_indicators}")
            print(f"   Standard German words: {german_words}")
            print(f"   Response length: {len(response)} chars, {len(response.split())} words")
            
            if swiss_indicators > german_words:
                print(f"   ✅ Appears to be Swiss German!")
            elif german_words > swiss_indicators:
                print(f"   ⚠️  Appears to be Standard German")
            else:
                print(f"   🤔 Mixed or unclear")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_apertus_swiss_german()