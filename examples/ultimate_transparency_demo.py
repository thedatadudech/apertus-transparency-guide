"""
🇨🇭 Ultimate Apertus Transparency Demo
Shows EVERYTHING happening in the model - layer by layer, step by step
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from apertus_core import ApertusCore
from transparency_analyzer import ApertusTransparencyAnalyzer
import warnings
warnings.filterwarnings('ignore')

class UltimateTransparencyDemo:
    """Complete transparency analysis of Apertus model"""
    
    def __init__(self):
        print("🇨🇭 APERTUS ULTIMATE TRANSPARENCY DEMO")
        print("=" * 60)
        print("Loading Apertus model with full transparency enabled...")
        
        self.apertus = ApertusCore(enable_transparency=True)
        self.analyzer = ApertusTransparencyAnalyzer(self.apertus)
        
        print("✅ Model loaded! Ready for complete transparency analysis.\n")
    
    def complete_analysis(self, text: str = "Apertus ist ein transparentes KI-Modell aus der Schweiz."):
        """Run complete transparency analysis on input text"""
        
        print(f"🔍 ANALYZING: '{text}'")
        print("=" * 80)
        
        # 1. Architecture Overview
        print("\n🏗️  STEP 1: MODEL ARCHITECTURE")
        self._show_architecture()
        
        # 2. Token Breakdown
        print("\n🔤 STEP 2: TOKENIZATION")
        self._show_tokenization(text)
        
        # 3. Layer-by-Layer Processing
        print("\n🧠 STEP 3: LAYER-BY-LAYER PROCESSING")
        hidden_states = self._analyze_all_layers(text)
        
        # 4. Attention Analysis
        print("\n👁️  STEP 4: ATTENTION PATTERNS")
        self._analyze_attention_all_layers(text)
        
        # 5. Token Prediction Process
        print("\n🎲 STEP 5: TOKEN PREDICTION PROCESS")
        self._analyze_prediction_process(text)
        
        # 6. Summary
        print("\n📊 STEP 6: TRANSPARENCY SUMMARY")
        self._show_summary(text, hidden_states)
    
    def _show_architecture(self):
        """Show model architecture details"""
        config = self.apertus.model.config
        total_params = sum(p.numel() for p in self.apertus.model.parameters())
        
        print(f"🏗️  Model: {self.apertus.model_name}")
        print(f"📊 Architecture Details:")
        print(f"   • Layers: {config.num_hidden_layers}")
        print(f"   • Attention Heads: {config.num_attention_heads}")
        print(f"   • Hidden Size: {config.hidden_size}")
        print(f"   • Vocab Size: {config.vocab_size:,}")
        print(f"   • Parameters: {total_params:,}")
        print(f"   • Context Length: {config.max_position_embeddings:,}")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"   • GPU Memory: {memory_used:.1f} GB")
    
    def _show_tokenization(self, text):
        """Show detailed tokenization process"""
        tokens = self.apertus.tokenizer.tokenize(text)
        token_ids = self.apertus.tokenizer.encode(text)
        
        print(f"📝 Original Text: '{text}'")
        print(f"🔢 Token Count: {len(tokens)}")
        print(f"🔤 Tokens: {tokens}")
        print(f"🔢 Token IDs: {token_ids}")
        
        # Show token-by-token breakdown
        print("\n📋 Token Breakdown:")
        for i, (token, token_id) in enumerate(zip(tokens, token_ids[1:])):  # Skip BOS if present
            print(f"   {i+1:2d}. '{token}' → ID: {token_id}")
    
    def _analyze_all_layers(self, text):
        """Analyze processing through all layers"""
        inputs = self.apertus.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.apertus.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        num_layers = len(hidden_states)
        
        print(f"🧠 Processing through {num_layers} layers...")
        
        layer_stats = []
        
        # Analyze each layer
        for layer_idx in range(0, num_layers, max(1, num_layers//8)):  # Sample every ~8th layer
            layer_state = hidden_states[layer_idx][0]  # Remove batch dimension
            
            # Calculate statistics
            mean_activation = layer_state.mean().item()
            std_activation = layer_state.std().item()
            l2_norm = torch.norm(layer_state, dim=-1).mean().item()
            max_activation = layer_state.max().item()
            min_activation = layer_state.min().item()
            
            layer_stats.append({
                'layer': layer_idx,
                'mean': mean_activation,
                'std': std_activation,
                'l2_norm': l2_norm,
                'max': max_activation,
                'min': min_activation
            })
            
            print(f"   Layer {layer_idx:2d}: L2={l2_norm:.3f}, Mean={mean_activation:+.3f}, "
                  f"Std={std_activation:.3f}, Range=[{min_activation:+.2f}, {max_activation:+.2f}]")
        
        return hidden_states
    
    def _analyze_attention_all_layers(self, text):
        """Analyze attention patterns across all layers"""
        inputs = self.apertus.tokenizer(text, return_tensors="pt")
        tokens = self.apertus.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        with torch.no_grad():
            outputs = self.apertus.model(**inputs, output_attentions=True)
        
        attentions = outputs.attentions
        print(f"👁️  Analyzing attention across {len(attentions)} layers...")
        
        # Sample key layers for attention analysis
        key_layers = [0, len(attentions)//4, len(attentions)//2, 3*len(attentions)//4, len(attentions)-1]
        
        for layer_idx in key_layers:
            if layer_idx >= len(attentions):
                continue
                
            attention_weights = attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]
            
            # Average attention across heads
            avg_attention = attention_weights.mean(dim=0).cpu().numpy()
            
            # Find most attended tokens
            total_attention_received = avg_attention.sum(axis=0)
            total_attention_given = avg_attention.sum(axis=1)
            
            print(f"\n   Layer {layer_idx} Attention Summary:")
            print(f"   • Matrix Shape: {avg_attention.shape}")
            print(f"   • Attention Heads: {attention_weights.shape[0]}")
            
            # Top tokens that receive attention
            top_receivers = np.argsort(total_attention_received)[-3:][::-1]
            print(f"   • Most Attended Tokens:")
            for i, token_idx in enumerate(top_receivers):
                if token_idx < len(tokens):
                    attention_score = total_attention_received[token_idx]
                    print(f"     {i+1}. '{tokens[token_idx]}' (score: {attention_score:.3f})")
            
            # Attention distribution stats
            attention_entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-12), axis=1).mean()
            print(f"   • Avg Attention Entropy: {attention_entropy:.3f}")
    
    def _analyze_prediction_process(self, text):
        """Analyze the token prediction process in detail"""
        print(f"🎲 Predicting next tokens for: '{text}'")
        
        # Get model predictions for next token
        inputs = self.apertus.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.apertus.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token predictions
        
        # Convert to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, 10)
        
        print(f"🎯 Top 10 Next Token Predictions:")
        for i in range(10):
            token_id = top_indices[i].item()
            token = self.apertus.tokenizer.decode([token_id])
            prob = top_probs[i].item()
            logit = logits[token_id].item()
            
            # Confidence indicator
            if prob > 0.2:
                confidence = "🔥 High"
            elif prob > 0.05:
                confidence = "✅ Medium"
            elif prob > 0.01:
                confidence = "⚠️  Low"
            else:
                confidence = "❓ Very Low"
            
            print(f"   {i+1:2d}. '{token}' → {prob:.1%} (logit: {logit:+.2f}) {confidence}")
        
        # Probability distribution stats
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12)).item()
        max_prob = probabilities.max().item()
        top_10_prob_sum = top_probs.sum().item()
        
        print(f"\n📊 Prediction Statistics:")
        print(f"   • Entropy: {entropy:.2f} (randomness measure)")
        print(f"   • Max Probability: {max_prob:.1%}")
        print(f"   • Top-10 Probability Sum: {top_10_prob_sum:.1%}")
        print(f"   • Confidence: {'High' if max_prob > 0.5 else 'Medium' if max_prob > 0.2 else 'Low'}")
    
    def _show_summary(self, text, hidden_states):
        """Show complete transparency summary"""
        num_tokens = len(self.apertus.tokenizer.tokenize(text))
        num_layers = len(hidden_states)
        
        print(f"📋 COMPLETE TRANSPARENCY ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"🔤 Input: '{text}'")
        print(f"📊 Processed through:")
        print(f"   • {num_tokens} tokens")
        print(f"   • {num_layers} transformer layers")
        print(f"   • {self.apertus.model.config.num_attention_heads} attention heads per layer")
        print(f"   • {self.apertus.model.config.hidden_size} hidden dimensions")
        
        total_operations = num_tokens * num_layers * self.apertus.model.config.num_attention_heads
        print(f"   • ~{total_operations:,} attention operations")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"   • {memory_used:.1f} GB GPU memory used")
        
        print(f"\n✨ This is what makes Apertus transparent:")
        print(f"   🔍 Every layer activation is accessible")
        print(f"   👁️  Every attention weight is visible")
        print(f"   🎲 Every prediction probability is shown")
        print(f"   🧠 Every hidden state can be analyzed")
        print(f"   📊 Complete mathematical operations are exposed")
        
        print(f"\n🇨🇭 Swiss AI Transparency: No black boxes, complete visibility! ✨")

def main():
    """Run the ultimate transparency demo"""
    try:
        demo = UltimateTransparencyDemo()
        
        # Default examples
        examples = [
            "Apertus ist ein transparentes KI-Modell aus der Schweiz.",
            "Machine learning requires transparency for trust.",
            "La Suisse développe des modèles d'IA transparents.",
            "Artificial intelligence should be explainable.",
        ]
        
        print("🎯 Choose an example or enter your own text:")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
        print("5. Enter custom text")
        
        try:
            choice = input("\nChoice (1-5): ").strip()
            
            if choice == "5":
                text = input("Enter your text: ").strip()
                if not text:
                    text = examples[0]  # Default fallback
            elif choice in ["1", "2", "3", "4"]:
                text = examples[int(choice) - 1]
            else:
                print("Invalid choice, using default...")
                text = examples[0]
            
        except (KeyboardInterrupt, EOFError):
            text = examples[0]
            print(f"\nUsing default: {text}")
        
        # Run complete analysis
        demo.complete_analysis(text)
        
        print("\n🎉 Complete transparency analysis finished!")
        print("This demonstrates the full transparency capabilities of Apertus Swiss AI.")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        print("Make sure the model is properly loaded and accessible.")

if __name__ == "__main__":
    main()