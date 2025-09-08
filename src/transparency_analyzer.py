"""
Advanced transparency analysis tools for Apertus Swiss AI
Provides deep introspection into model decision-making processes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
try:
    from .apertus_core import ApertusCore
except ImportError:
    from apertus_core import ApertusCore

logger = logging.getLogger(__name__)


class ApertusTransparencyAnalyzer:
    """
    Advanced transparency analysis for Apertus models
    
    Enables complete introspection into neural network operations,
    attention patterns, hidden states, and decision processes.
    """
    
    def __init__(self, apertus_core: Optional[ApertusCore] = None):
        """
        Initialize transparency analyzer
        
        Args:
            apertus_core: Initialized ApertusCore instance, or None to create new
        """
        if apertus_core is None:
            self.apertus = ApertusCore(enable_transparency=True)
        else:
            self.apertus = apertus_core
            
        # Ensure transparency features are enabled
        if not (hasattr(self.apertus.model, 'config') and 
                getattr(self.apertus.model.config, 'output_attentions', False)):
            logger.warning("Model not configured for transparency analysis. Some features may not work.")
    
    def analyze_model_architecture(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of model architecture
        
        Returns:
            Dictionary containing detailed architecture information
        """
        logger.info("🔍 Analyzing Apertus model architecture...")
        
        config = self.apertus.model.config
        
        # Basic architecture info
        architecture = {
            "model_type": config.model_type,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_position_embeddings,
        }
        
        # Parameter analysis
        total_params = sum(p.numel() for p in self.apertus.model.parameters())
        trainable_params = sum(p.numel() for p in self.apertus.model.parameters() if p.requires_grad)
        
        architecture.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_gb": total_params * 2 / 1e9,  # Approximate for float16
        })
        
        # Layer breakdown
        layer_info = {}
        for name, module in self.apertus.model.named_modules():
            if hasattr(module, 'weight') and len(list(module.parameters())) > 0:
                params = sum(p.numel() for p in module.parameters())
                layer_info[name] = {
                    "parameters": params,
                    "shape": list(module.weight.shape) if hasattr(module, 'weight') else None,
                    "dtype": str(module.weight.dtype) if hasattr(module, 'weight') else None
                }
        
        architecture["layer_breakdown"] = layer_info
        
        # Print summary
        print("🏗️ APERTUS ARCHITECTURE ANALYSIS")
        print("=" * 60)
        print(f"Model Type: {architecture['model_type']}")
        print(f"Layers: {architecture['num_hidden_layers']}")
        print(f"Attention Heads: {architecture['num_attention_heads']}")
        print(f"Hidden Size: {architecture['hidden_size']}")
        print(f"Vocabulary: {architecture['vocab_size']:,} tokens")
        print(f"Total Parameters: {total_params:,}")
        print(f"Model Size: ~{architecture['model_size_gb']:.2f} GB")
        
        return architecture
    
    def visualize_attention_patterns(
        self,
        text: str,
        layer: int = 15,
        head: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Visualize attention patterns for given text
        
        Args:
            text: Input text to analyze
            layer: Which transformer layer to analyze (0 to num_layers-1)
            head: Specific attention head (None for average across heads)
            save_path: Optional path to save visualization
            
        Returns:
            Tuple of (attention_matrix, tokens)
        """
        logger.info(f"🎯 Analyzing attention patterns for: '{text}'")
        
        # Tokenize input
        inputs = self.apertus.tokenizer(text, return_tensors="pt")
        tokens = self.apertus.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Move inputs to model device
        device = next(self.apertus.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.apertus.model(**inputs, output_attentions=True)
        
        # Extract attention weights
        if layer >= len(outputs.attentions):
            layer = len(outputs.attentions) - 1
            logger.warning(f"Layer {layer} not available, using layer {len(outputs.attentions) - 1}")
        
        attention_weights = outputs.attentions[layer][0]  # [num_heads, seq_len, seq_len]
        
        # Average across heads or select specific head
        if head is None:
            attention_matrix = attention_weights.mean(dim=0).cpu().numpy()
            title_suffix = f"Layer {layer} (All Heads Average)"
        else:
            if head >= attention_weights.shape[0]:
                head = 0
                logger.warning(f"Head {head} not available, using head 0")
            attention_matrix = attention_weights[head].cpu().numpy()
            title_suffix = f"Layer {layer}, Head {head}"
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            attention_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Blues',
            cbar_kws={'label': 'Attention Weight'},
            square=True
        )
        
        plt.title(f'Attention Patterns - {title_suffix}')
        plt.xlabel('Key Tokens (what it looks at)')
        plt.ylabel('Query Tokens (what is looking)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention visualization saved to {save_path}")
        
        plt.show()
        
        # Print attention insights
        print(f"\n🔍 ATTENTION INSIGHTS FOR: '{text}'")
        print("=" * 60)
        print(f"Attention Matrix Shape: {attention_matrix.shape}")
        print(f"Max Attention Weight: {attention_matrix.max():.4f}")
        print(f"Average Attention Weight: {attention_matrix.mean():.4f}")
        print(f"Attention Spread (std): {attention_matrix.std():.4f}")
        
        # Show top attention patterns
        print("\n🎯 TOP ATTENTION PATTERNS:")
        for i, token in enumerate(tokens[:min(5, len(tokens))]):
            if i < attention_matrix.shape[0]:
                top_attention_idx = attention_matrix[i].argmax()
                top_attention_token = tokens[top_attention_idx] if top_attention_idx < len(tokens) else "N/A"
                attention_score = attention_matrix[i][top_attention_idx]
                print(f"  '{token}' → '{top_attention_token}' ({attention_score:.3f})")
        
        return attention_matrix, tokens
    
    def trace_hidden_states(
        self,
        text: str,
        analyze_layers: Optional[List[int]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Track evolution of hidden states through model layers
        
        Args:
            text: Input text to analyze
            analyze_layers: Specific layers to analyze (None for key layers)
            
        Returns:
            Dictionary mapping layer indices to analysis results
        """
        logger.info(f"🧠 Tracing hidden state evolution for: '{text}'")
        
        # Default to key layers if none specified
        if analyze_layers is None:
            num_layers = self.apertus.model.config.num_hidden_layers
            analyze_layers = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
        
        # Tokenize input
        inputs = self.apertus.tokenizer(text, return_tensors="pt")
        tokens = self.apertus.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Move inputs to model device
        device = next(self.apertus.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get hidden states
        with torch.no_grad():
            outputs = self.apertus.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        layer_analysis = {}
        
        print(f"\n🔄 HIDDEN STATE EVOLUTION FOR: '{text}'")
        print("=" * 60)
        
        for layer_idx in analyze_layers:
            if layer_idx >= len(hidden_states):
                continue
                
            layer_states = hidden_states[layer_idx][0]  # Remove batch dimension
            
            # Calculate statistics for each token
            token_stats = []
            for i, token in enumerate(tokens):
                if i < layer_states.shape[0]:
                    token_vector = layer_states[i].cpu().numpy()
                    stats = {
                        'token': token,
                        'mean_activation': np.mean(token_vector),
                        'std_activation': np.std(token_vector),
                        'max_activation': np.max(token_vector),
                        'min_activation': np.min(token_vector),
                        'l2_norm': np.linalg.norm(token_vector),
                        'activation_range': np.max(token_vector) - np.min(token_vector)
                    }
                    token_stats.append(stats)
            
            # Layer-level statistics
            layer_stats = {
                'avg_l2_norm': np.mean([s['l2_norm'] for s in token_stats]),
                'max_l2_norm': np.max([s['l2_norm'] for s in token_stats]),
                'avg_activation': np.mean([s['mean_activation'] for s in token_stats]),
                'activation_spread': np.std([s['mean_activation'] for s in token_stats])
            }
            
            layer_analysis[layer_idx] = {
                'token_stats': token_stats,
                'layer_stats': layer_stats,
                'hidden_state_shape': layer_states.shape
            }
            
            # Print layer summary
            print(f"\nLayer {layer_idx}:")
            print(f"  Hidden State Shape: {layer_states.shape}")
            print(f"  Average L2 Norm: {layer_stats['avg_l2_norm']:.4f}")
            print(f"  Peak L2 Norm: {layer_stats['max_l2_norm']:.4f}")
            print(f"  Average Activation: {layer_stats['avg_activation']:.4f}")
            
            # Show strongest tokens
            sorted_tokens = sorted(token_stats, key=lambda x: x['l2_norm'], reverse=True)
            print(f"  Strongest Tokens:")
            for i, stats in enumerate(sorted_tokens[:3]):
                print(f"    {i+1}. '{stats['token']}' (L2: {stats['l2_norm']:.4f})")
        
        # Visualize evolution
        self._plot_hidden_state_evolution(layer_analysis, analyze_layers, tokens)
        
        return layer_analysis
    
    def _plot_hidden_state_evolution(
        self,
        layer_analysis: Dict[int, Dict[str, Any]],
        layers: List[int],
        tokens: List[str]
    ):
        """Plot hidden state evolution across layers"""
        plt.figure(figsize=(14, 8))
        
        # Plot 1: Average L2 norms across layers
        plt.subplot(2, 2, 1)
        avg_norms = [layer_analysis[layer]['layer_stats']['avg_l2_norm'] for layer in layers]
        plt.plot(layers, avg_norms, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Layer')
        plt.ylabel('Average L2 Norm')
        plt.title('Representation Strength Evolution')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Token-specific evolution (first 5 tokens)
        plt.subplot(2, 2, 2)
        for token_idx in range(min(5, len(tokens))):
            token_norms = []
            for layer in layers:
                if token_idx < len(layer_analysis[layer]['token_stats']):
                    norm = layer_analysis[layer]['token_stats'][token_idx]['l2_norm']
                    token_norms.append(norm)
                else:
                    token_norms.append(0)
            
            plt.plot(layers, token_norms, 'o-', label=f"'{tokens[token_idx]}'", linewidth=1.5)
        
        plt.xlabel('Layer')
        plt.ylabel('L2 Norm')
        plt.title('Token-Specific Evolution')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Activation spread
        plt.subplot(2, 2, 3)
        spreads = [layer_analysis[layer]['layer_stats']['activation_spread'] for layer in layers]
        plt.plot(layers, spreads, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Layer')
        plt.ylabel('Activation Spread (std)')
        plt.title('Representation Diversity')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Peak vs Average activations
        plt.subplot(2, 2, 4)
        avg_norms = [layer_analysis[layer]['layer_stats']['avg_l2_norm'] for layer in layers]
        max_norms = [layer_analysis[layer]['layer_stats']['max_l2_norm'] for layer in layers]
        
        plt.plot(layers, avg_norms, 'bo-', label='Average', linewidth=2)
        plt.plot(layers, max_norms, 'ro-', label='Peak', linewidth=2)
        plt.xlabel('Layer')
        plt.ylabel('L2 Norm')
        plt.title('Peak vs Average Activations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_token_predictions(
        self,
        prompt: str,
        max_new_tokens: int = 5,
        temperature: float = 0.7,
        show_top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Analyze step-by-step token prediction process
        
        Args:
            prompt: Initial prompt
            max_new_tokens: Number of tokens to generate and analyze
            temperature: Sampling temperature
            show_top_k: Number of top candidates to show for each step
            
        Returns:
            List of prediction steps with probabilities and selections
        """
        logger.info(f"🎲 Analyzing token predictions for: '{prompt}'")
        
        print(f"\n🎲 TOKEN PREDICTION ANALYSIS")
        print("=" * 60)
        print(f"Prompt: '{prompt}'")
        print(f"Temperature: {temperature}")
        
        # Encode initial prompt
        input_ids = self.apertus.tokenizer.encode(prompt, return_tensors="pt")
        generation_steps = []
        
        for step in range(max_new_tokens):
            print(f"\n--- STEP {step + 1} ---")
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.apertus.model(input_ids)
                logits = outputs.logits[0, -1, :]  # Last token's predictions
            
            # Apply temperature and convert to probabilities
            scaled_logits = logits / temperature
            probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            # Get top candidates
            top_probs, top_indices = torch.topk(probabilities, show_top_k)
            
            # Create step data
            step_data = {
                'step': step + 1,
                'current_text': self.apertus.tokenizer.decode(input_ids[0]),
                'candidates': [],
                'logits_stats': {
                    'max_logit': logits.max().item(),
                    'min_logit': logits.min().item(),
                    'mean_logit': logits.mean().item(),
                    'std_logit': logits.std().item()
                }
            }
            
            print(f"Current text: '{step_data['current_text']}'")
            print(f"\nTop {show_top_k} Token Candidates:")
            
            for i in range(show_top_k):
                token_id = top_indices[i].item()
                token = self.apertus.tokenizer.decode([token_id])
                prob = top_probs[i].item()
                logit = logits[token_id].item()
                
                candidate = {
                    'rank': i + 1,
                    'token': token,
                    'token_id': token_id,
                    'probability': prob,
                    'logit': logit
                }
                step_data['candidates'].append(candidate)
                
                # Visual indicators for probability ranges
                if prob > 0.3:
                    indicator = "🔥"  # High confidence
                elif prob > 0.1:
                    indicator = "✅"  # Medium confidence
                elif prob > 0.05:
                    indicator = "⚠️"   # Low confidence
                else:
                    indicator = "❓"  # Very low confidence
                
                print(f"  {i+1:2d}. '{token}' - {prob:.1%} (logit: {logit:.2f}) {indicator}")
            
            # Sample next token
            next_token_id = torch.multinomial(probabilities, 1)
            next_token = self.apertus.tokenizer.decode([next_token_id.item()])
            
            # Find rank of selected token
            selected_rank = "N/A"
            if next_token_id in top_indices:
                selected_rank = (top_indices == next_token_id).nonzero().item() + 1
            
            step_data['selected_token'] = next_token
            step_data['selected_token_id'] = next_token_id.item()
            step_data['selected_rank'] = selected_rank
            
            print(f"\n🎯 SELECTED: '{next_token}' (rank: {selected_rank})")
            
            generation_steps.append(step_data)
            
            # Update input for next iteration
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
        
        # Final result
        final_text = self.apertus.tokenizer.decode(input_ids[0])
        print(f"\n✨ FINAL GENERATED TEXT: '{final_text}'")
        
        return generation_steps
    
    def weight_analysis(
        self,
        layer_name: str = "model.layers.15.self_attn.q_proj",
        sample_size: int = 100
    ) -> Optional[np.ndarray]:
        """
        Analyze specific layer weights
        
        Args:
            layer_name: Name of the layer to analyze
            sample_size: Size of sample for visualization
            
        Returns:
            Weight matrix if successful, None if layer not found
        """
        logger.info(f"⚖️ Analyzing weights for layer: {layer_name}")
        
        print(f"\n⚖️ WEIGHT ANALYSIS: {layer_name}")
        print("=" * 60)
        
        try:
            # Get the specified layer
            layer = dict(self.apertus.model.named_modules())[layer_name]
            weights = layer.weight.data.cpu().numpy()
            
            print(f"Weight Matrix Shape: {weights.shape}")
            print(f"Weight Statistics:")
            print(f"  Mean: {np.mean(weights):.6f}")
            print(f"  Std:  {np.std(weights):.6f}")
            print(f"  Min:  {np.min(weights):.6f}")
            print(f"  Max:  {np.max(weights):.6f}")
            print(f"  Total Parameters: {weights.size:,}")
            print(f"  Memory Usage: {weights.nbytes / 1024**2:.2f} MB")
            
            # Create visualizations
            self._plot_weight_analysis(weights, layer_name, sample_size)
            
            return weights
            
        except KeyError:
            print(f"❌ Layer '{layer_name}' not found!")
            print("\n📋 Available layers:")
            for name, module in self.apertus.model.named_modules():
                if hasattr(module, 'weight'):
                    print(f"  {name}")
            return None
    
    def _plot_weight_analysis(
        self,
        weights: np.ndarray,
        layer_name: str,
        sample_size: int
    ):
        """Plot weight analysis visualizations"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Weight distribution
        plt.subplot(2, 3, 1)
        plt.hist(weights.flatten(), bins=50, alpha=0.7, edgecolor='black', color='skyblue')
        plt.title(f'Weight Distribution\n{layer_name}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Weight matrix heatmap (sample)
        plt.subplot(2, 3, 2)
        if len(weights.shape) > 1:
            sample_weights = weights[:sample_size, :sample_size]
        else:
            sample_weights = weights[:sample_size].reshape(-1, 1)
        
        plt.imshow(sample_weights, cmap='RdBu', vmin=-0.1, vmax=0.1, aspect='auto')
        plt.title(f'Weight Matrix Sample\n({sample_size}x{sample_size})')
        plt.colorbar(label='Weight Value')
        
        # Plot 3: Row-wise statistics
        plt.subplot(2, 3, 3)
        if len(weights.shape) > 1:
            row_means = np.mean(weights, axis=1)
            row_stds = np.std(weights, axis=1)
            plt.plot(row_means, label='Row Means', alpha=0.7)
            plt.plot(row_stds, label='Row Stds', alpha=0.7)
            plt.title('Row-wise Statistics')
            plt.xlabel('Row Index')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 4: Weight magnitude distribution
        plt.subplot(2, 3, 4)
        weight_magnitudes = np.abs(weights.flatten())
        plt.hist(weight_magnitudes, bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
        plt.title('Weight Magnitude Distribution')
        plt.xlabel('|Weight Value|')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Sparsity analysis
        plt.subplot(2, 3, 5)
        threshold_range = np.logspace(-4, -1, 20)
        sparsity_ratios = []
        
        for threshold in threshold_range:
            sparse_ratio = np.mean(np.abs(weights) < threshold)
            sparsity_ratios.append(sparse_ratio)
        
        plt.semilogx(threshold_range, sparsity_ratios, 'o-', linewidth=2)
        plt.title('Sparsity Analysis')
        plt.xlabel('Threshold')
        plt.ylabel('Fraction of Weights Below Threshold')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Weight norm by layer section
        plt.subplot(2, 3, 6)
        if len(weights.shape) > 1:
            section_size = max(1, weights.shape[0] // 20)
            section_norms = []
            section_labels = []
            
            for i in range(0, weights.shape[0], section_size):
                end_idx = min(i + section_size, weights.shape[0])
                section = weights[i:end_idx]
                section_norm = np.linalg.norm(section)
                section_norms.append(section_norm)
                section_labels.append(f"{i}-{end_idx}")
            
            plt.bar(range(len(section_norms)), section_norms, alpha=0.7, color='lightgreen')
            plt.title('Section-wise L2 Norms')
            plt.xlabel('Weight Section')
            plt.ylabel('L2 Norm')
            plt.xticks(range(0, len(section_labels), max(1, len(section_labels)//5)))
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_available_layers(self) -> Dict[str, List[str]]:
        """
        Get list of all available layers for analysis
        
        Returns:
            Dictionary organizing layers by type
        """
        layers = {
            "attention": [],
            "mlp": [],
            "embedding": [],
            "norm": [],
            "other": []
        }
        
        for name, module in self.apertus.model.named_modules():
            if hasattr(module, 'weight'):
                if 'attn' in name:
                    layers["attention"].append(name)
                elif 'mlp' in name or 'feed_forward' in name:
                    layers["mlp"].append(name)
                elif 'embed' in name:
                    layers["embedding"].append(name)
                elif 'norm' in name or 'layer_norm' in name:
                    layers["norm"].append(name)
                else:
                    layers["other"].append(name)
        
        return layers
