"""
🇨🇭 Advanced Apertus Transparency Toolkit
Native weights inspection, attention visualization, layer tracking, and tokenizer comparisons
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import time
from apertus_core import ApertusCore
from transparency_analyzer import ApertusTransparencyAnalyzer
import warnings
warnings.filterwarnings('ignore')

import sys
from io import StringIO
from datetime import datetime

class AdvancedTransparencyToolkit:
    """Advanced transparency analysis with complete logging of all outputs"""
    
    def __init__(self):
        self.apertus = ApertusCore(enable_transparency=True)
        self.analyzer = ApertusTransparencyAnalyzer(self.apertus)
        
        # Setup logging capture
        self.log_buffer = StringIO()
        self.original_stdout = sys.stdout
        
        # Log the initialization
        self.log_and_print("🇨🇭 ADVANCED APERTUS TRANSPARENCY TOOLKIT")
        self.log_and_print("=" * 70)
        self.log_and_print("✅ Advanced toolkit ready!\n")
    
    def log_and_print(self, message):
        """Print to console AND capture to log"""
        print(message)
        self.log_buffer.write(message + "\n")
        
    def start_logging(self):
        """Start capturing all print output"""
        sys.stdout = self
        
    def stop_logging(self):
        """Stop capturing and restore normal output"""
        sys.stdout = self.original_stdout
        
    def write(self, text):
        """Capture output for logging"""
        self.original_stdout.write(text)
        self.log_buffer.write(text)
        
    def flush(self):
        """Flush both outputs"""
        self.original_stdout.flush()
    
    def native_weights_inspection(self, layer_pattern: str = "layers.15.self_attn"):
        """Native inspection of model weights with detailed analysis"""
        print(f"⚖️  NATIVE WEIGHTS INSPECTION: {layer_pattern}")
        print("=" * 70)
        
        matching_layers = []
        for name, module in self.apertus.model.named_modules():
            if layer_pattern in name and hasattr(module, 'weight'):
                matching_layers.append((name, module))
        
        if not matching_layers:
            print(f"❌ No layers found matching pattern: {layer_pattern}")
            return
        
        for name, module in matching_layers[:3]:  # Show first 3 matching layers
            print(f"\n🔍 Layer: {name}")
            print("-" * 50)
            
            # Convert bfloat16 to float32 for numpy compatibility
            weights = module.weight.data.cpu()
            if weights.dtype == torch.bfloat16:
                weights = weights.float()
            weights = weights.numpy()
            
            # Basic statistics
            print(f"📊 Weight Statistics:")
            print(f"   Shape: {weights.shape}")
            print(f"   Parameters: {weights.size:,}")
            print(f"   Memory: {weights.nbytes / 1024**2:.1f} MB")
            print(f"   Data type: {weights.dtype}")
            
            # Distribution analysis
            print(f"\n📈 Distribution Analysis:")
            print(f"   Mean: {np.mean(weights):+.6f}")
            print(f"   Std:  {np.std(weights):.6f}")
            print(f"   Min:  {np.min(weights):+.6f}")
            print(f"   Max:  {np.max(weights):+.6f}")
            print(f"   Range: {np.max(weights) - np.min(weights):.6f}")
            
            # Sparsity analysis
            thresholds = [1e-4, 1e-3, 1e-2, 1e-1]
            print(f"\n🕸️  Sparsity Analysis:")
            for threshold in thresholds:
                sparse_ratio = np.mean(np.abs(weights) < threshold)
                print(f"   |w| < {threshold:.0e}: {sparse_ratio:.1%}")
            
            # Weight magnitude distribution
            weight_magnitudes = np.abs(weights.flatten())
            percentiles = [50, 90, 95, 99, 99.9]
            print(f"\n📊 Magnitude Percentiles:")
            for p in percentiles:
                value = np.percentile(weight_magnitudes, p)
                print(f"   {p:4.1f}%: {value:.6f}")
            
            # Gradient statistics (if available)
            if hasattr(module.weight, 'grad') and module.weight.grad is not None:
                grad = module.weight.grad.data.cpu()
                if grad.dtype == torch.bfloat16:
                    grad = grad.float()
                grad = grad.numpy()
                print(f"\n🎯 Gradient Statistics:")
                print(f"   Mean: {np.mean(grad):+.6e}")
                print(f"   Std:  {np.std(grad):.6e}")
                print(f"   Max:  {np.max(np.abs(grad)):.6e}")
            
            # Layer-specific analysis
            if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                print(f"\n🔍 Attention Projection Analysis:")
                # Analyze attention projection patterns
                if len(weights.shape) == 2:
                    # Calculate column norms (output dimension norms)
                    col_norms = np.linalg.norm(weights, axis=0)
                    row_norms = np.linalg.norm(weights, axis=1)
                    
                    print(f"   Output dim norms - Mean: {np.mean(col_norms):.4f}, Std: {np.std(col_norms):.4f}")
                    print(f"   Input dim norms - Mean: {np.mean(row_norms):.4f}, Std: {np.std(row_norms):.4f}")
                    
                    # Check for any unusual patterns
                    zero_cols = np.sum(col_norms < 1e-6)
                    zero_rows = np.sum(row_norms < 1e-6)
                    if zero_cols > 0 or zero_rows > 0:
                        print(f"   ⚠️  Zero columns: {zero_cols}, Zero rows: {zero_rows}")
        
        print(f"\n✨ Native weights inspection completed!")
    
    def real_time_attention_visualization(self, text: str, num_steps: int = 3):
        """Real-time attention pattern visualization during generation"""
        print(f"👁️  REAL-TIME ATTENTION VISUALIZATION")
        print("=" * 70)
        print(f"Text: '{text}'")
        
        # Initial encoding
        inputs = self.apertus.tokenizer(text, return_tensors="pt")
        input_ids = inputs['input_ids']
        
        # Move to model device
        device = next(self.apertus.model.parameters()).device
        input_ids = input_ids.to(device)
        
        attention_history = []
        
        for step in range(num_steps):
            print(f"\n--- GENERATION STEP {step + 1} ---")
            
            # Get current text
            current_text = self.apertus.tokenizer.decode(input_ids[0])
            print(f"Current: '{current_text}'")
            
            # Forward pass with attention
            with torch.no_grad():
                outputs = self.apertus.model(input_ids, output_attentions=True)
                logits = outputs.logits[0, -1, :]
                attentions = outputs.attentions
            
            # Analyze attention in last layer
            last_layer_attention = attentions[-1][0]  # [num_heads, seq_len, seq_len]
            # Convert bfloat16 to float32 for numpy compatibility
            attention_cpu = last_layer_attention.mean(dim=0).cpu()
            if attention_cpu.dtype == torch.bfloat16:
                attention_cpu = attention_cpu.float()
            avg_attention = attention_cpu.numpy()
            
            # Get tokens
            tokens = self.apertus.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            print(f"Tokens: {tokens}")
            print(f"Attention matrix shape: {avg_attention.shape}")
            
            # Show attention patterns for last token
            if len(tokens) > 1:
                last_token_attention = avg_attention[-1, :-1]  # What last token attends to
                top_attended = np.argsort(last_token_attention)[-3:][::-1]
                
                print(f"Last token '{tokens[-1]}' attends most to:")
                for i, token_idx in enumerate(top_attended):
                    if token_idx < len(tokens) - 1:
                        attention_score = last_token_attention[token_idx]
                        print(f"   {i+1}. '{tokens[token_idx]}' ({attention_score:.3f})")
            
            # Store attention history
            attention_history.append({
                'step': step + 1,
                'tokens': tokens.copy(),
                'attention': avg_attention.copy(),
                'text': current_text
            })
            
            # Generate next token
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, 1)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
            
            next_token = self.apertus.tokenizer.decode([next_token_id.item()])
            print(f"Next token: '{next_token}'")
        
        print(f"\n✅ Real-time attention visualization completed!")
        return attention_history
    
    def layer_evolution_real_time_tracking(self, text: str):
        """Real-time tracking of layer evolution during forward pass"""
        print(f"🧠 REAL-TIME LAYER EVOLUTION TRACKING")
        print("=" * 70)
        print(f"Text: '{text}'")
        
        inputs = self.apertus.tokenizer(text, return_tensors="pt")
        tokens = self.apertus.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Move to model device
        device = next(self.apertus.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"Tokens: {tokens}")
        print(f"Tracking through {self.apertus.model.config.num_hidden_layers} layers...\n")
        
        # Forward pass with hidden states
        with torch.no_grad():
            start_time = time.time()
            outputs = self.apertus.model(**inputs, output_hidden_states=True)
            forward_time = time.time() - start_time
        
        hidden_states = outputs.hidden_states
        
        # Track evolution through layers
        layer_evolution = []
        
        print(f"⏱️  Forward pass took {forward_time:.3f}s")
        print(f"\n🔄 Layer-by-Layer Evolution:")
        
        # Sample layers for detailed analysis
        sample_layers = list(range(0, len(hidden_states), max(1, len(hidden_states)//10)))
        
        for i, layer_idx in enumerate(sample_layers):
            layer_state = hidden_states[layer_idx][0]  # Remove batch dimension
            
            # Per-token analysis
            token_stats = []
            for token_pos in range(layer_state.shape[0]):
                token_vector = layer_state[token_pos]
                
                stats = {
                    'token': tokens[token_pos] if token_pos < len(tokens) else '<pad>',
                    'l2_norm': torch.norm(token_vector).item(),
                    'mean': token_vector.mean().item(),
                    'std': token_vector.std().item(),
                    'max': token_vector.max().item(),
                    'min': token_vector.min().item(),
                    'sparsity': (torch.abs(token_vector) < 0.01).float().mean().item()
                }
                token_stats.append(stats)
            
            # Layer-level statistics
            layer_stats = {
                'layer': layer_idx,
                'avg_l2_norm': np.mean([s['l2_norm'] for s in token_stats]),
                'max_l2_norm': np.max([s['l2_norm'] for s in token_stats]),
                'avg_activation': np.mean([s['mean'] for s in token_stats]),
                'activation_spread': np.std([s['mean'] for s in token_stats]),
                'avg_sparsity': np.mean([s['sparsity'] for s in token_stats])
            }
            
            layer_evolution.append(layer_stats)
            
            print(f"Layer {layer_idx:2d}: L2={layer_stats['avg_l2_norm']:.3f}, "
                  f"Mean={layer_stats['avg_activation']:+.4f}, "
                  f"Spread={layer_stats['activation_spread']:.4f}, "
                  f"Sparsity={layer_stats['avg_sparsity']:.1%}")
            
            # Show most active tokens in this layer
            top_tokens = sorted(token_stats, key=lambda x: x['l2_norm'], reverse=True)[:3]
            active_tokens = [f"'{t['token']}'({t['l2_norm']:.2f})" for t in top_tokens]
            print(f"         Most active: {', '.join(active_tokens)}")
        
        # Evolution analysis
        print(f"\n📊 Evolution Analysis:")
        
        # Check for increasing/decreasing patterns
        l2_norms = [stats['avg_l2_norm'] for stats in layer_evolution]
        if len(l2_norms) > 1:
            trend = "increasing" if l2_norms[-1] > l2_norms[0] else "decreasing"
            change = ((l2_norms[-1] - l2_norms[0]) / l2_norms[0]) * 100
            print(f"L2 norm trend: {trend} ({change:+.1f}%)")
        
        # Check layer specialization
        sparsity_levels = [stats['avg_sparsity'] for stats in layer_evolution]
        if len(sparsity_levels) > 1:
            sparsity_trend = "increasing" if sparsity_levels[-1] > sparsity_levels[0] else "decreasing"
            print(f"Sparsity trend: {sparsity_trend} (early: {sparsity_levels[0]:.1%}, late: {sparsity_levels[-1]:.1%})")
        
        print(f"\n✅ Real-time layer tracking completed!")
        return layer_evolution
    
    def decision_process_analysis(self, prompt: str, max_steps: int = 5, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50):
        """Deep analysis of the decision-making process"""
        print(f"🎲 DECISION PROCESS ANALYSIS")
        print("=" * 70)
        print(f"Prompt: '{prompt}'")
        print(f"🎛️ Sampling Parameters:")
        print(f"   Temperature: {temperature} (creativity control)")
        print(f"   Top-P: {top_p} (nucleus sampling)")  
        print(f"   Top-K: {top_k} (candidate pool size)")
        
        input_ids = self.apertus.tokenizer.encode(prompt, return_tensors="pt")
        
        # Move to model device
        device = next(self.apertus.model.parameters()).device
        input_ids = input_ids.to(device)
        decision_history = []
        
        for step in range(max_steps):
            print(f"\n--- DECISION STEP {step + 1} ---")
            current_text = self.apertus.tokenizer.decode(input_ids[0])
            print(f"Current text: '{current_text}'")
            
            # Forward pass
            with torch.no_grad():
                outputs = self.apertus.model(input_ids, output_attentions=True, output_hidden_states=True)
                logits = outputs.logits[0, -1, :]
                attentions = outputs.attentions
                hidden_states = outputs.hidden_states
            
            # Apply temperature scaling for decision analysis
            scaled_logits = logits / temperature
            probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            # Show the effect of temperature
            original_probs = torch.nn.functional.softmax(logits, dim=-1)
            
            print(f"🌡️ Temperature Effect Analysis:")
            orig_top5 = torch.topk(original_probs, 5)[0]
            temp_top5 = torch.topk(probabilities, 5)[0]
            print(f"   Without temp: Top-5 = {[f'{p.item():.1%}' for p in orig_top5]}")
            print(f"   With temp={temperature}: Top-5 = {[f'{p.item():.1%}' for p in temp_top5]}")
            
            # Entropy analysis (uncertainty measure)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12)).item()
            
            # Confidence analysis
            max_prob = probabilities.max().item()
            top_probs, top_indices = torch.topk(probabilities, 10)
            
            print(f"🎯 Decision Metrics:")
            print(f"   Entropy: {entropy:.3f} (uncertainty)")
            print(f"   Max confidence: {max_prob:.1%}")
            print(f"   Top-3 probability mass: {top_probs[:3].sum().item():.1%}")
            
            # Analyze what influenced this decision
            last_hidden = hidden_states[-1][0, -1, :]  # Last token's final hidden state
            hidden_magnitude = torch.norm(last_hidden).item()
            
            print(f"   Hidden state magnitude: {hidden_magnitude:.2f}")
            
            # Check attention focus
            last_attention = attentions[-1][0, :, -1, :-1].mean(dim=0)  # Average over heads
            top_attention_indices = torch.topk(last_attention, 3)[1]
            tokens = self.apertus.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            print(f"   Attention focused on:")
            for i, idx in enumerate(top_attention_indices):
                if idx < len(tokens):
                    attention_score = last_attention[idx].item()
                    print(f"     {i+1}. '{tokens[idx]}' ({attention_score:.3f})")
            
            # Show top candidates with reasoning
            print(f"\n🏆 Top candidates:")
            for i in range(5):
                token_id = top_indices[i].item()
                token = self.apertus.tokenizer.decode([token_id])
                prob = top_probs[i].item()
                logit = logits[token_id].item()
                
                # Confidence assessment
                if prob > 0.3:
                    confidence_level = "🔥 Very High"
                elif prob > 0.1:
                    confidence_level = "✅ High"
                elif prob > 0.05:
                    confidence_level = "⚠️  Medium"
                else:
                    confidence_level = "❓ Low"
                
                print(f"   {i+1}. '{token}' → {prob:.1%} (logit: {logit:+.2f}) {confidence_level}")
            
            # Decision quality assessment
            if entropy < 2.0:
                decision_quality = "🎯 Confident"
            elif entropy < 4.0:
                decision_quality = "⚖️  Balanced"
            else:
                decision_quality = "🤔 Uncertain"
            
            print(f"\n📊 Decision quality: {decision_quality}")
            
            # Apply top-k filtering if specified
            sampling_probs = probabilities.clone()
            if top_k > 0 and top_k < len(probabilities):
                top_k_values, top_k_indices = torch.topk(probabilities, top_k)
                # Zero out probabilities not in top-k
                sampling_probs = torch.zeros_like(probabilities)
                sampling_probs[top_k_indices] = top_k_values
                sampling_probs = sampling_probs / sampling_probs.sum()
                print(f"🔄 Top-K filtering: Reduced {len(probabilities)} → {top_k} candidates")
            
            # Apply top-p (nucleus) filtering if specified  
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(sampling_probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                nucleus_mask = cumulative_probs <= top_p
                nucleus_mask[0] = True  # Keep at least one token
                
                nucleus_probs = torch.zeros_like(sampling_probs)
                nucleus_probs[sorted_indices[nucleus_mask]] = sorted_probs[nucleus_mask]
                sampling_probs = nucleus_probs / nucleus_probs.sum()
                
                nucleus_size = nucleus_mask.sum().item()
                nucleus_mass = sorted_probs[nucleus_mask].sum().item()
                print(f"🌀 Top-P filtering: Nucleus size = {nucleus_size} tokens ({nucleus_mass:.1%} probability mass)")
            
            # Show final sampling distribution vs display distribution
            final_top5 = torch.topk(sampling_probs, 5)
            print(f"🎯 Final sampling distribution:")
            for i in range(5):
                if final_top5[0][i] > 0:
                    token = self.apertus.tokenizer.decode([final_top5[1][i].item()])
                    prob = final_top5[0][i].item()
                    print(f"   {i+1}. '{token}' → {prob:.1%}")
            
            # Make decision (sample next token)
            next_token_id = torch.multinomial(sampling_probs, 1)
            selected_token = self.apertus.tokenizer.decode([next_token_id.item()])
            
            # Find rank of selected token
            selected_rank = "N/A"
            if next_token_id in top_indices:
                selected_rank = (top_indices == next_token_id).nonzero().item() + 1
            
            print(f"🎲 SELECTED: '{selected_token}' (rank: {selected_rank})")
            
            # Store decision data
            decision_history.append({
                'step': step + 1,
                'text': current_text,
                'selected_token': selected_token,
                'selected_rank': selected_rank,
                'entropy': entropy,
                'max_confidence': max_prob,
                'hidden_magnitude': hidden_magnitude,
                'decision_quality': decision_quality
            })
            
            # Update for next step
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
        
        # Final analysis
        final_text = self.apertus.tokenizer.decode(input_ids[0])
        print(f"\n✨ FINAL GENERATED TEXT: '{final_text}'")
        
        avg_entropy = np.mean([d['entropy'] for d in decision_history])
        avg_confidence = np.mean([d['max_confidence'] for d in decision_history])
        
        print(f"\n📊 Generation Analysis:")
        print(f"   Average entropy: {avg_entropy:.2f}")
        print(f"   Average confidence: {avg_confidence:.1%}")
        print(f"   Generation quality: {'High' if avg_confidence > 0.3 else 'Medium' if avg_confidence > 0.15 else 'Low'}")
        
        return decision_history
    
    def comprehensive_tokenizer_comparison(self, test_word: str = "Bundesgesundheitsamt"):
        """Compare tokenization across different model tokenizers"""
        print(f"🔤 COMPREHENSIVE TOKENIZER COMPARISON")
        print("=" * 70)
        print(f"Test word: '{test_word}'")
        
        # Test with Apertus tokenizer
        print(f"\n🇨🇭 Apertus Tokenizer:")
        apertus_tokens = self.apertus.tokenizer.tokenize(test_word)
        apertus_ids = self.apertus.tokenizer.encode(test_word, add_special_tokens=False)
        
        print(f"   Tokens: {apertus_tokens}")
        print(f"   Token IDs: {apertus_ids}")
        print(f"   Token count: {len(apertus_tokens)}")
        print(f"   Efficiency: {len(test_word) / len(apertus_tokens):.1f} chars/token")
        
        # Detailed analysis of each token
        print(f"   Token breakdown:")
        for i, (token, token_id) in enumerate(zip(apertus_tokens, apertus_ids)):
            print(f"     {i+1}. '{token}' → ID: {token_id}")
        
        # Try to compare with other common tokenizers (if available)
        comparison_results = [
            {
                'model': 'Apertus Swiss AI',
                'tokens': apertus_tokens,
                'count': len(apertus_tokens),
                'efficiency': len(test_word) / len(apertus_tokens),
                'vocab_size': self.apertus.tokenizer.vocab_size
            }
        ]
        
        # Try GPT-2 style tokenizer
        try:
            from transformers import GPT2Tokenizer
            gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            gpt2_tokens = gpt2_tokenizer.tokenize(test_word)
            
            print(f"\n🤖 GPT-2 Tokenizer (for comparison):")
            print(f"   Tokens: {gpt2_tokens}")
            print(f"   Token count: {len(gpt2_tokens)}")
            print(f"   Efficiency: {len(test_word) / len(gpt2_tokens):.1f} chars/token")
            
            comparison_results.append({
                'model': 'GPT-2',
                'tokens': gpt2_tokens,
                'count': len(gpt2_tokens),
                'efficiency': len(test_word) / len(gpt2_tokens),
                'vocab_size': gpt2_tokenizer.vocab_size
            })
        except Exception as e:
            print(f"\n⚠️  GPT-2 tokenizer not available: {e}")
        
        # Try BERT tokenizer
        try:
            from transformers import BertTokenizer
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # Convert to lowercase for BERT
            bert_tokens = bert_tokenizer.tokenize(test_word.lower())
            
            print(f"\n📚 BERT Tokenizer (for comparison):")
            print(f"   Tokens: {bert_tokens}")
            print(f"   Token count: {len(bert_tokens)}")
            print(f"   Efficiency: {len(test_word) / len(bert_tokens):.1f} chars/token")
            
            comparison_results.append({
                'model': 'BERT',
                'tokens': bert_tokens,
                'count': len(bert_tokens),
                'efficiency': len(test_word) / len(bert_tokens),
                'vocab_size': bert_tokenizer.vocab_size
            })
        except Exception as e:
            print(f"\n⚠️  BERT tokenizer not available: {e}")
        
        # Analysis summary
        print(f"\n📊 TOKENIZATION COMPARISON SUMMARY:")
        print(f"{'Model':<20} {'Tokens':<8} {'Efficiency':<12} {'Vocab Size'}")
        print("-" * 60)
        
        for result in comparison_results:
            print(f"{result['model']:<20} {result['count']:<8} {result['efficiency']:<12.1f} {result['vocab_size']:,}")
        
        # Specific German compound word analysis
        if test_word == "Bundesgesundheitsamt":
            print(f"\n🇩🇪 GERMAN COMPOUND WORD ANALYSIS:")
            print(f"   Word parts: Bundes + gesundheits + amt")
            print(f"   Meaning: Federal Health Office")
            print(f"   Character count: {len(test_word)}")
            
            # Check if Apertus handles German compounds better
            apertus_efficiency = len(test_word) / len(apertus_tokens)
            print(f"   Apertus efficiency: {apertus_efficiency:.1f} chars/token")
            
            if apertus_efficiency > 8:
                print(f"   ✅ Excellent German compound handling!")
            elif apertus_efficiency > 6:
                print(f"   ✅ Good German compound handling")
            else:
                print(f"   ⚠️  Could be more efficient for German compounds")
        
        # Test additional German compound words
        additional_tests = [
            "Krankenversicherung",
            "Donaudampfschifffahrt", 
            "Rechtsschutzversicherung",
            "Arbeitsplatzcomputer"
        ]
        
        print(f"\n🧪 Additional German Compound Tests:")
        for word in additional_tests:
            tokens = self.apertus.tokenizer.tokenize(word)
            efficiency = len(word) / len(tokens)
            print(f"   '{word}' → {len(tokens)} tokens ({efficiency:.1f} chars/token)")
        
        print(f"\n✅ Tokenizer comparison completed!")
        return comparison_results
    
    def save_complete_log(self, filename: str = None):
        """Save complete command-line output log"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"apertus_transparency_log_{timestamp}.txt"
        
        # Get all captured output
        log_content = self.log_buffer.getvalue()
        
        # Add header with system info
        header = f"""# 🇨🇭 Apertus Advanced Transparency Analysis Log
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: {self.apertus.model_name}
GPU: {self.apertus.device_info['gpu_name'] if self.apertus.device_info['has_gpu'] else 'CPU'}
Memory: {self.apertus.device_info['gpu_memory_gb']:.1f} GB

====================================================================================
COMPLETE COMMAND-LINE OUTPUT CAPTURE:
====================================================================================

"""
        
        # Combine header with all captured output
        full_log = header + log_content
        
        # Save log
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_log)
        
        print(f"\n📝 Complete analysis log saved to: {filename}")
        print(f"📊 Log contains {len(log_content)} characters of output")
        
        return filename

def main():
    """Run the advanced transparency toolkit"""
    
    toolkit = AdvancedTransparencyToolkit()
    
    while True:
        print("\n🎯 ADVANCED TRANSPARENCY TOOLKIT MENU")
        print("=" * 50)
        print("1. Native Weights Inspection")
        print("2. Real-time Attention Visualization")
        print("3. Layer Evolution Real-time Tracking")
        print("4. Decision Process Analysis")
        print("5. Tokenizer Comparison (Bundesgesundheitsamt)")
        print("6. Run All Analyses")
        print("7. Save Complete Log")
        print("8. Custom Analysis")
        print("0. Exit")
        
        try:
            choice = input("\nSelect option (0-8): ").strip()
            
            if choice == "0":
                print("\n👋 Advanced Transparency Toolkit beendet. Auf Wiedersehen!")
                break
            
            elif choice == "1":
                layer = input("Enter layer pattern (e.g., 'layers.15.self_attn'): ").strip()
                if not layer:
                    layer = "layers.15.self_attn"
                toolkit.native_weights_inspection(layer)
                
            elif choice == "2":
                text = input("Enter text for attention analysis: ").strip()
                if not text:
                    text = "Apertus analysiert die Schweizer KI-Transparenz."
                toolkit.real_time_attention_visualization(text)
                
            elif choice == "3":
                text = input("Enter text for layer tracking: ").strip()
                if not text:
                    text = "Transparenz ist wichtig für Vertrauen."
                toolkit.layer_evolution_real_time_tracking(text)
                
            elif choice == "4":
                prompt = input("Enter prompt for decision analysis: ").strip()
                if not prompt:
                    prompt = "Die Schweizer KI-Forschung ist"
                toolkit.decision_process_analysis(prompt)
                
            elif choice == "5":
                word = input("Enter word for tokenizer comparison (default: Bundesgesundheitsamt): ").strip()
                if not word:
                    word = "Bundesgesundheitsamt"
                toolkit.comprehensive_tokenizer_comparison(word)
                
            elif choice == "6":
                print("\n🚀 Running all analyses...")
                
                # Start capturing ALL output
                toolkit.start_logging()
                
                toolkit.native_weights_inspection()
                toolkit.real_time_attention_visualization("Apertus ist transparent.")
                toolkit.layer_evolution_real_time_tracking("Schweizer KI-Innovation.")
                toolkit.decision_process_analysis("Die Schweizer KI-Forschung ist")
                toolkit.comprehensive_tokenizer_comparison("Bundesgesundheitsamt")
                
                # Stop capturing
                toolkit.stop_logging()
                print("✅ All analyses completed!")
                
            elif choice == "7":
                filename = input("Enter log filename (or press Enter for auto): ").strip()
                if not filename:
                    filename = None
                toolkit.save_complete_log(filename)
                
            elif choice == "8":
                print("Custom analysis - combine any methods as needed!")
                
            else:
                print("Invalid choice, please select 0-8.")
        
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Advanced toolkit session ended.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Returning to menu...")

if __name__ == "__main__":
    main()