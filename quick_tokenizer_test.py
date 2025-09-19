#!/usr/bin/env python3
"""
🔍 Quick Swiss German Tokenizer Comparison
Schneller Test ohne Model-Loading - nur Tokenization
"""

from transformers import AutoTokenizer
import time

def compare_tokenizers():
    print("🇨🇭 SWISS GERMAN TOKENIZER COMPARISON")
    print("=" * 50)
    
    # Test texts
    texts = {
        "Swiss German 1": "Grüezi! Chönd Sie mer bitte d Schwyzer KI erchläre?",
        "Swiss German 2": "Was isch KI und wie funktioniert das?", 
        "Standard German": "Hallo! Können Sie mir bitte die Schweizer KI erklären?",
        "Swiss Dialect": "Mir händ hüt es schöns Wätter, gäll?",
        "Technical German": "Die Künstliche Intelligenz verwendet neuronale Netzwerke."
    }
    
    # Models to compare
    models = [
        ("🇨🇭 Apertus Swiss AI", "swiss-ai/Apertus-8B-Instruct-2509"),
        ("🇩🇪 German BERT", "bert-base-german-cased"),
        ("🇩🇪 German GPT-2", "dbmdz/german-gpt2"),
        ("🌍 Multilingual BERT", "bert-base-multilingual-cased"),
        ("🤖 Standard GPT-2", "gpt2")
    ]
    
    print("📝 Test Texts:")
    for name, text in texts.items():
        print(f"  {name}: {text}")
    print()
    
    # Compare each model
    results = {}
    
    for model_name, model_id in models:
        print(f"🧠 Testing: {model_name}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            load_time = time.time() - start_time
            
            model_results = {}
            
            for text_name, text in texts.items():
                # Tokenize
                tokens = tokenizer.tokenize(text)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                
                # Analyze problems
                problems = []
                if any("Ã" in t for t in tokens):
                    problems.append("UTF-8 encoding issues")
                single_chars = [t for t in tokens if len(t) == 1 and t.isalpha()]
                if single_chars:
                    problems.append(f"{len(single_chars)} single character tokens")
                
                # Calculate efficiency
                efficiency = len(tokens) / len(text)
                
                model_results[text_name] = {
                    "tokens": tokens,
                    "token_count": len(tokens),
                    "efficiency": efficiency,
                    "problems": problems,
                    "problematic_tokens": [t for t in tokens if "Ã" in t or (len(t) == 1 and t.isalpha())]
                }
                
                print(f"  {text_name:15s}: {len(tokens):2d} tokens, {efficiency:.3f} tok/char")
                if problems:
                    print(f"    ⚠️  Issues: {', '.join(problems)}")
                if model_results[text_name]["problematic_tokens"]:
                    prob_tokens = model_results[text_name]["problematic_tokens"][:3]
                    print(f"    🔍 Examples: {prob_tokens}")
            
            results[model_name] = model_results
            print(f"  ⏱️  Load time: {load_time:.2f}s")
            print()
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            print()
    
    # Summary comparison
    print("📊 EFFICIENCY SUMMARY (Swiss German 1)")
    print("=" * 50)
    
    swiss_results = []
    for model_name, model_data in results.items():
        if "Swiss German 1" in model_data:
            data = model_data["Swiss German 1"]
            swiss_results.append({
                "model": model_name,
                "tokens": data["token_count"],
                "efficiency": data["efficiency"],
                "problems": len(data["problematic_tokens"])
            })
    
    # Sort by efficiency (lower = better)
    swiss_results.sort(key=lambda x: x["efficiency"])
    
    print("Ranking (lower tokens/char = better):")
    for i, result in enumerate(swiss_results):
        rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
        print(f"{rank_emoji} {result['model']:20s}: {result['efficiency']:.3f} tok/char, "
              f"{result['tokens']} tokens, {result['problems']} problems")
    
    # Show detailed tokenization for best and worst
    if len(swiss_results) >= 2:
        best = swiss_results[0]
        worst = swiss_results[-1]
        
        print(f"\n🔍 DETAILED COMPARISON")
        print("=" * 50)
        
        text = texts["Swiss German 1"]
        print(f"Text: {text}")
        print()
        
        for model_type, model_name in [(best, "BEST"), (worst, "WORST")]:
            print(f"{model_name}: {model_type['model']}")
            tokens = results[model_type['model']]["Swiss German 1"]["tokens"]
            print("Tokens:")
            for i, token in enumerate(tokens):
                marker = " ⚠️" if ("Ã" in token or (len(token) == 1 and token.isalpha())) else ""
                print(f"  {i+1:2d}: |{token}|{marker}")
            print()

if __name__ == "__main__":
    compare_tokenizers()