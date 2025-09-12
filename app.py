"""
🇨🇭 Apertus Swiss AI Transparency Dashboard
Gradio-based HuggingFace Spaces application
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import os
import spaces

# Set environment variables to reduce verbosity and warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

warnings.filterwarnings('ignore')

# Try to import CUDA xIELU optimization for Apertus
try:
    from xielu.ops.wrappers import XIELU
    XIELU_AVAILABLE = True
    print("✅ CUDA xIELU optimization available - Apertus performance enhanced!")
except ImportError:
    XIELU_AVAILABLE = False
    print("ℹ️ CUDA xIELU not available - using fallback (install: pip install git+https://github.com/nickjbrowning/XIELU)")

# Global variables for model and tokenizer
model = None
tokenizer = None

@spaces.GPU
def load_model(hf_token):
    """Load Apertus model with HuggingFace token"""
    global model, tokenizer
    
    if not hf_token or not hf_token.startswith("hf_"):
        return "❌ Invalid HuggingFace token. Must start with 'hf_'"
    
    model_name = "swiss-ai/Apertus-8B-Instruct-2509"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
            output_attentions=True,
            output_hidden_states=True,
            trust_remote_code=True
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        memory_usage = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        # Check for xIELU optimization status
        xielu_status = "✅ CUDA xIELU Active" if XIELU_AVAILABLE and torch.cuda.is_available() else "⚠️ xIELU Fallback"
        
        if memory_usage > 0:
            return f"✅ Model loaded successfully!\n📊 Parameters: {total_params:,}\n💾 Memory: {memory_usage:.1f} GB\n🚀 Optimization: {xielu_status}"
        else:
            return f"✅ Model loaded successfully!\n📊 Parameters: {total_params:,}\n💾 CPU mode\n🚀 Optimization: {xielu_status}"
        
    except Exception as e:
        return f"❌ Failed to load model: {str(e)}\n💡 Check your token and model access permissions."

@spaces.GPU
def chat_with_apertus(message, max_tokens=300):
    """Simple chat function"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return "❌ Please load the model first by entering your HuggingFace token."
    
    try:
        formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### System:
You are Apertus, a helpful Swiss AI assistant. You are transparent, multilingual, and precise.

### Instruction:
{message}

### Response:
"""
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.split("### Response:")[-1].strip()
        
        return f"🇨🇭 **Apertus:** {response}"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

@spaces.GPU
def analyze_attention(text, layer=15):
    """Analyze attention patterns"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return None, "❌ Please load the model first."
    
    try:
        inputs = tokenizer(text, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        attention_weights = outputs.attentions[layer][0]
        avg_attention = attention_weights.mean(dim=0).cpu()
        
        if avg_attention.dtype == torch.bfloat16:
            avg_attention = avg_attention.float()
        
        avg_attention = avg_attention.numpy()
        
        # Create attention heatmap
        fig = px.imshow(
            avg_attention,
            x=tokens,
            y=tokens,
            color_continuous_scale='Blues',
            title=f"Attention Patterns - Layer {layer}",
            labels={'color': 'Attention Weight'}
        )
        fig.update_layout(height=500)
        
        # Get insights
        attention_received = avg_attention.sum(axis=0)
        top_indices = np.argsort(attention_received)[-3:][::-1]
        
        insights = "**🎯 Top Attended Tokens:**\n\n"
        for i, idx in enumerate(top_indices):
            if idx < len(tokens):
                score = attention_received[idx]
                token = tokens[idx]
                
                # Use markdown code blocks to prevent any formatting issues
                insights += f"{i+1}. Token: `{token}` • Score: {score:.3f}\n\n"
        
        return fig, insights
        
    except Exception as e:
        return None, f"❌ Error analyzing attention: {str(e)}"

@spaces.GPU
def analyze_token_predictions(text):
    """Analyze next token predictions"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return None, "❌ Please load the model first."
    
    try:
        inputs = tokenizer(text, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
        
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probabilities, 10)
        
        # Create prediction data
        pred_data = []
        for i in range(10):
            token_id = top_indices[i].item()
            token = tokenizer.decode([token_id])
            # Keep original tokens - they show important tokenization info
            if not token.strip():
                token = f"[ID:{token_id}]"
            prob = top_probs[i].item()
            pred_data.append({"Rank": i+1, "Token": token, "Probability": prob})
        
        df = pd.DataFrame(pred_data)
        
        fig = px.bar(df, x="Token", y="Probability",
                   title="Top 10 Most Likely Next Tokens",
                   color="Probability", color_continuous_scale="viridis")
        fig.update_layout(height=400)
        
        # Create insights
        insights = "**🏆 Prediction Details:**\n\n"
        for _, row in df.iterrows():
            prob_pct = row["Probability"] * 100
            confidence = "🔥" if prob_pct > 20 else "✅" if prob_pct > 5 else "⚠️"
            confidence_text = "Very confident" if prob_pct > 20 else "Confident" if prob_pct > 5 else "Uncertain"
            
            token = str(row['Token'])
            # Use markdown code blocks to prevent formatting issues
            insights += f"{row['Rank']}. Token: `{token}` • {prob_pct:.1f}% {confidence} ({confidence_text})\n\n"
        
        return fig, insights
        
    except Exception as e:
        return None, f"❌ Error analyzing predictions: {str(e)}"

@spaces.GPU
def analyze_layer_evolution(text):
    """Analyze how representations evolve through layers"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return None, "❌ Please load the model first."
    
    try:
        inputs = tokenizer(text, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        
        # Sample key layers
        sample_layers = [0, 4, 8, 12, 16, 20, 24, 28, 31]
        layer_stats = []
        
        for layer_idx in sample_layers:
            if layer_idx < len(hidden_states):
                layer_state = hidden_states[layer_idx][0]
                
                layer_cpu = layer_state.cpu()
                if layer_cpu.dtype == torch.bfloat16:
                    layer_cpu = layer_cpu.float()
                
                l2_norms = torch.norm(layer_cpu, dim=-1)
                
                layer_stats.append({
                    "Layer": layer_idx,
                    "L2_Norm_Mean": l2_norms.mean().item(),
                    "L2_Norm_Max": l2_norms.max().item(),
                    "Hidden_Mean": layer_cpu.mean().item(),
                    "Hidden_Std": layer_cpu.std().item()
                })
        
        df = pd.DataFrame(layer_stats)
        
        # Create evolution plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('L2 Norm Evolution', 'Hidden State Mean',
                          'Hidden State Std', 'Layer Comparison'),
            vertical_spacing=0.12
        )
        
        fig.add_trace(go.Scatter(x=df['Layer'], y=df['L2_Norm_Mean'],
                               mode='lines+markers', name='L2 Mean'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Layer'], y=df['Hidden_Mean'],
                               mode='lines+markers', name='Hidden Mean'), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['Layer'], y=df['Hidden_Std'],
                               mode='lines+markers', name='Hidden Std'), row=2, col=1)
        fig.add_trace(go.Bar(x=df['Layer'], y=df['L2_Norm_Max'],
                           name='L2 Max'), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, title="Neural Representation Evolution")
        
        # Create table
        table_html = df.round(4).to_html(index=False, classes='table table-striped')
        
        return fig, f"**📊 Layer Statistics:**\n{table_html}"
        
    except Exception as e:
        return None, f"❌ Error analyzing layer evolution: {str(e)}"

@spaces.GPU
def analyze_weights(layer_num, layer_type):
    """Analyze weight distribution with research-based metrics"""
    global model
    
    if model is None:
        return None, "❌ Please load the model first."
    
    try:
        selected_layer = f"model.layers.{layer_num}.{layer_type}"
        
        # Get weights directly
        layer_dict = dict(model.named_modules())
        if selected_layer not in layer_dict:
            return None, f"❌ Layer '{selected_layer}' not found"
        
        layer_obj = layer_dict[selected_layer]
        if not hasattr(layer_obj, 'weight'):
            return None, f"❌ Layer has no weights"
        
        weights = layer_obj.weight.data.cpu()
        if weights.dtype == torch.bfloat16:
            weights = weights.float()
        weights = weights.numpy()
        
        # Research-based analysis
        l1_norm = np.sum(np.abs(weights))
        l2_norm = np.sqrt(np.sum(weights**2))
        zero_weights = np.sum(np.abs(weights) < 1e-8)
        dead_ratio = zero_weights / weights.size * 100
        weight_range = np.max(weights) - np.min(weights)
        
        # Sparsity analysis with LLM-appropriate thresholds
        sparse_001 = np.mean(np.abs(weights) < 0.001) * 100  # Tiny weights
        sparse_01 = np.mean(np.abs(weights) < 0.01) * 100    # Very small weights  
        sparse_1 = np.mean(np.abs(weights) < 0.1) * 100      # Small weights
        
        # Percentiles
        p25, p50, p75, p95 = np.percentile(np.abs(weights), [25, 50, 75, 95])
        
        # Smart visualization for different layer sizes
        if weights.size < 500000:  # Small layers - full histogram
            fig = px.histogram(weights.flatten(), bins=50, 
                             title=f"Weight Distribution - {selected_layer}",
                             labels={'x': 'Weight Value', 'y': 'Frequency'},
                             color_discrete_sequence=['#2E86AB'])
            fig.add_vline(x=np.mean(weights), line_dash="dash", line_color="red", 
                        annotation_text=f"Mean: {np.mean(weights):.6f}")
            
        elif weights.size < 2000000:  # Medium layers - sampled histogram
            # Sample 100k weights for visualization
            sample_size = min(100000, weights.size)
            sampled_weights = np.random.choice(weights.flatten(), sample_size, replace=False)
            fig = px.histogram(sampled_weights, bins=50,
                             title=f"Weight Distribution - {selected_layer} (Sampled: {sample_size:,}/{weights.size:,})",
                             labels={'x': 'Weight Value', 'y': 'Frequency'},
                             color_discrete_sequence=['#2E86AB'])
            fig.add_vline(x=np.mean(weights), line_dash="dash", line_color="red",
                        annotation_text=f"Mean: {np.mean(weights):.6f}")
                        
        else:  # Large layers - statistical summary plot
            # Create a multi-panel statistical visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Weight Statistics Summary',
                    'Sparsity Analysis', 
                    'Distribution Percentiles',
                    'Health Indicators'
                ),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "indicator"}]]
            )
            
            # Panel 1: Basic statistics
            fig.add_trace(go.Bar(
                x=['Mean', 'Std', 'Min', 'Max'],
                y=[np.mean(weights), np.std(weights), np.min(weights), np.max(weights)],
                name='Statistics',
                marker_color='#2E86AB'
            ), row=1, col=1)
            
            # Panel 2: Sparsity levels (Updated for 8B LLM standards)
            fig.add_trace(go.Bar(
                x=['<0.001', '<0.01', '<0.1'],
                y=[sparse_001, sparse_01, sparse_1],
                name='Sparsity %',
                marker_color=[
                    '#28a745' if sparse_001 < 25 else '#ffc107' if sparse_001 < 40 else '#ff8c00' if sparse_001 < 55 else '#dc3545',
                    '#28a745' if sparse_01 < 50 else '#ffc107' if sparse_01 < 65 else '#ff8c00' if sparse_01 < 80 else '#dc3545',
                    '#28a745' if sparse_1 < 75 else '#ffc107' if sparse_1 < 85 else '#ff8c00' if sparse_1 < 92 else '#dc3545'
                ]
            ), row=1, col=2)
            
            # Panel 3: Percentiles
            fig.add_trace(go.Bar(
                x=['25th', '50th', '75th', '95th'],
                y=[p25, p50, p75, p95],
                name='Percentiles',
                marker_color='#17a2b8'
            ), row=2, col=1)
            
            # Panel 4: Health score gauge
            health_score = 100
            if dead_ratio > 15: health_score -= 30
            elif dead_ratio > 5: health_score -= 15
            if sparse_001 > 30: health_score -= 20
            elif sparse_001 > 10: health_score -= 10
            if weight_range < 0.001: health_score -= 25
            if weight_range > 10: health_score -= 25
            
            fig.add_trace(go.Indicator(
                mode = "gauge+number",
                value = health_score,
                title = {'text': "Health Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': '#2E86AB'},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}
            ), row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False, 
                            title=f"Statistical Analysis - {selected_layer} ({weights.size:,} parameters)")
            
        fig.update_layout(height=500, showlegend=False)
        
        # Health assessment (updated for 8B LLM standards)
        health_score = 100
        
        # Dead weights - very strict since truly dead weights are bad
        if dead_ratio > 15: health_score -= 30
        elif dead_ratio > 5: health_score -= 15
        
        # Tiny weights (<0.001) - updated thresholds based on LLM research
        if sparse_001 > 55: health_score -= 25  # >55% is concerning
        elif sparse_001 > 40: health_score -= 15  # >40% needs attention
        elif sparse_001 > 25: health_score -= 5   # >25% is acceptable
        
        # Weight range - extreme ranges indicate problems
        if weight_range < 0.001: health_score -= 20  # Too compressed
        elif weight_range > 10: health_score -= 20   # Too wide
        
        health_color = "🟢" if health_score >= 80 else "🟡" if health_score >= 60 else "🔴"
        health_status = "Excellent" if health_score >= 90 else "Good" if health_score >= 80 else "Fair" if health_score >= 60 else "Poor"
        
        # Format results
        results = f"""
## ⚖️ Weight Analysis: {selected_layer}

### 📊 Core Statistics
- **Shape:** {weights.shape}
- **Parameters:** {weights.size:,}
- **Mean:** {np.mean(weights):+.6f}
- **Std:** {np.std(weights):.6f}

### 🔬 Weight Health Analysis
- **L1 Norm:** {l1_norm:.3f} (Manhattan distance - sparsity indicator)
- **L2 Norm:** {l2_norm:.3f} (Euclidean distance - magnitude measure)
- **Dead Weights:** {dead_ratio:.1f}% (weights ≈ 0)
- **Range:** {weight_range:.6f} (Max - Min weight values)

### 🕸️ Sparsity Analysis (8B LLM Research-Based Thresholds)
- **Tiny (<0.001):** {sparse_001:.1f}% {'🟢 Excellent' if sparse_001 < 25 else '🟡 Good' if sparse_001 < 40 else '⚠️ Watch' if sparse_001 < 55 else '🔴 Concerning'}
- **Very Small (<0.01):** {sparse_01:.1f}% {'🟢 Excellent' if sparse_01 < 50 else '🟡 Good' if sparse_01 < 65 else '⚠️ Acceptable' if sparse_01 < 80 else '🔴 High'}
- **Small (<0.1):** {sparse_1:.1f}% {'🟢 Excellent' if sparse_1 < 75 else '🟡 Good' if sparse_1 < 85 else '⚠️ Normal' if sparse_1 < 92 else '🔴 Very High'}

### 📈 Distribution Characteristics
- **25th Percentile:** {p25:.6f}
- **Median:** {p50:.6f}
- **75th Percentile:** {p75:.6f}
- **95th Percentile:** {p95:.6f}

### 🏥 Layer Health Assessment: {health_color} {health_status} ({health_score}/100)

**Key Insights (8B LLM Standards):**
- **Weight Activity:** {100-dead_ratio:.1f}% of weights are active (target: >95%)
- **Sparsity Pattern:** {sparse_1:.1f}% small weights (8B LLMs: 70-85% is normal)
- **Distribution Health:** L2/L1 ratio = {l2_norm/l1_norm:.3f} (balanced ≈ 0.1-1.0)
- **Learning Capacity:** Weight range suggests {'good' if 0.01 < weight_range < 5 else 'limited'} learning capacity

💡 **Research Note:** High sparsity (70-90%) is **normal** for large transformers and indicates efficient learned representations, not poor health.
        """
        
        return fig, results
        
    except Exception as e:
        return None, f"❌ Error analyzing weights: {str(e)}"

# Create Gradio interface with custom CSS
def create_interface():
    # Custom CSS for dark Swiss theme
    custom_css = """
    /* Dark Swiss-inspired styling */
    .gradio-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        font-family: 'Helvetica Neue', 'Arial', sans-serif;
        color: #f8f9fa;
    }
    
    .main-header {
        background: linear-gradient(135deg, #dc3545 0%, #8B0000 100%);
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(220, 53, 69, 0.4);
        border: 1px solid rgba(220, 53, 69, 0.3);
    }
    
    .feature-box {
        background: rgba(25, 25, 46, 0.95);
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border-left: 4px solid #dc3545;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .auth-section {
        background: rgba(25, 25, 46, 0.9);
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #dc3545;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.2);
    }
    
    .footer-section {
        background: linear-gradient(135deg, #0d1421 0%, #1a1a2e 100%);
        padding: 30px;
        border-radius: 15px;
        margin-top: 40px;
        color: #f8f9fa;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Tab styling */
    .tab-nav {
        background: rgba(25, 25, 46, 0.95);
        border-radius: 10px;
        padding: 5px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Button improvements */
    .gr-button {
        background: linear-gradient(135deg, #dc3545 0%, #8B0000 100%);
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        color: white;
        box-shadow: 0 2px 8px rgba(220, 53, 69, 0.3);
    }
    
    .gr-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(220, 53, 69, 0.6);
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
    }
    
    /* Input field styling */
    .gr-textbox, .gr-dropdown {
        background: rgba(25, 25, 46, 0.8);
        border-radius: 8px;
        border: 2px solid rgba(255, 255, 255, 0.2);
        transition: border-color 0.3s ease;
        color: #f8f9fa;
    }
    
    .gr-textbox:focus, .gr-dropdown:focus {
        border-color: #dc3545;
        box-shadow: 0 0 0 3px rgba(220, 53, 69, 0.2);
        background: rgba(25, 25, 46, 0.9);
    }
    
    /* Tab content styling */
    .gr-tab-item {
        background: rgba(25, 25, 46, 0.5);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Text color improvements */
    .gr-markdown, .gr-html, .gr-textbox label {
        color: #f8f9fa;
    }
    
    /* Plot background */
    .gr-plot {
        background: rgba(25, 25, 46, 0.8);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    """
    
    with gr.Blocks(
        title="🇨🇭 Apertus Swiss AI Transparency Dashboard", 
        theme=gr.themes.Default(
            primary_hue="red",
            secondary_hue="gray",
            neutral_hue="gray",
            font=gr.themes.GoogleFont("Inter")
        ),
        css=custom_css
    ) as demo:
        
        # Main Header
        gr.HTML("""
        <div class="main-header">
            <div style="text-align: center; max-width: 1200px; margin: 0 auto;">
                <h1 style="color: white; font-size: 3em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    🇨🇭 Apertus Swiss AI Transparency Dashboard
                </h1>
                <h2 style="color: white; margin: 10px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                    The World's Most Transparent Language Model
                </h2>
                <p style="color: white; font-size: 1.2em; margin: 15px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                    <strong>Explore the internal workings of Switzerland's open-source 8B parameter AI model</strong>
                </p>
            </div>
        </div>
        """)
        
        # Feature Overview
        gr.HTML("""
        <div class="feature-box">
            <h3 style="color: #ff6b6b; margin-bottom: 20px; font-size: 1.5em;">🎯 What makes Apertus special?</h3>
            <p style="font-size: 1.1em; margin-bottom: 15px; color: #f8f9fa; font-weight: 500;">
                Unlike ChatGPT or Claude, you can see <strong>EVERYTHING</strong> happening inside the AI model:
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; margin: 20px 0;">
                <div style="background: rgba(13, 20, 33, 0.8); padding: 20px; border-radius: 10px; border-left: 4px solid #4dabf7; box-shadow: 0 4px 12px rgba(77, 171, 247, 0.2); border: 1px solid rgba(77, 171, 247, 0.3);">
                    <strong style="color: #74c0fc; font-size: 1.1em;">🧠 Attention Patterns</strong><br>
                    <span style="color: #ced4da; line-height: 1.4;">Which words the AI focuses on (like eye-tracking during reading)</span>
                </div>
                <div style="background: rgba(13, 20, 33, 0.8); padding: 20px; border-radius: 10px; border-left: 4px solid #51cf66; box-shadow: 0 4px 12px rgba(81, 207, 102, 0.2); border: 1px solid rgba(81, 207, 102, 0.3);">
                    <strong style="color: #8ce99a; font-size: 1.1em;">⚖️ Neural Weights</strong><br>
                    <span style="color: #ced4da; line-height: 1.4;">The "brain connections" that control decisions</span>
                </div>
                <div style="background: rgba(13, 20, 33, 0.8); padding: 20px; border-radius: 10px; border-left: 4px solid #ffd43b; box-shadow: 0 4px 12px rgba(255, 212, 59, 0.2); border: 1px solid rgba(255, 212, 59, 0.3);">
                    <strong style="color: #ffec99; font-size: 1.1em;">🎲 Prediction Probabilities</strong><br>
                    <span style="color: #ced4da; line-height: 1.4;">How confident the AI is about each word choice</span>
                </div>
                <div style="background: rgba(13, 20, 33, 0.8); padding: 20px; border-radius: 10px; border-left: 4px solid #22b8cf; box-shadow: 0 4px 12px rgba(34, 184, 207, 0.2); border: 1px solid rgba(34, 184, 207, 0.3);">
                    <strong style="color: #66d9ef; font-size: 1.1em;">🔍 Thinking Process</strong><br>
                    <span style="color: #ced4da; line-height: 1.4;">Step-by-step how responses are generated</span>
                </div>
                <div style="background: rgba(13, 20, 33, 0.8); padding: 20px; border-radius: 10px; border-left: 4px solid #ff6b6b; box-shadow: 0 4px 12px rgba(255, 107, 107, 0.2); border: 1px solid rgba(255, 107, 107, 0.3);">
                    <strong style="color: #ff8a8a; font-size: 1.1em;">🚀 CUDA xIELU</strong><br>
                    <span style="color: #ced4da; line-height: 1.4;">Swiss innovation: learnable activation function with GPU acceleration</span>
                </div>
            </div>
            <p style="text-align: center; font-size: 1.3em; margin-top: 25px; color: #ff6b6b; font-weight: 600;">
                <strong>This is complete AI transparency + Swiss innovations! 🇨🇭</strong>
            </p>
        </div>
        """)
        
        # Authentication Section
        gr.HTML("""
        <div class="auth-section">
            <h3 style="color: #ff6b6b; margin-bottom: 15px; text-align: center; font-size: 1.4em;">🔐 Model Authentication</h3>
            <p style="text-align: center; color: #f8f9fa; margin-bottom: 20px; font-size: 1.1em; font-weight: 500;">
                Enter your HuggingFace token to access the Apertus-8B-Instruct-2509 model
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                hf_token = gr.Textbox(
                    label="🗝️ HuggingFace Token",
                    placeholder="hf_...",
                    type="password",
                    info="Required to access swiss-ai/Apertus-8B-Instruct-2509. Get your token from: https://huggingface.co/settings/tokens",
                    container=True
                )
            with gr.Column(scale=1):
                load_btn = gr.Button(
                    "🇨🇭 Load Apertus Model", 
                    variant="primary", 
                    size="lg",
                    elem_classes="auth-button"
                )
        
        with gr.Row():
            model_status = gr.Textbox(
                label="📊 Model Status", 
                interactive=False,
                container=True
            )
        
        load_btn.click(load_model, inputs=[hf_token], outputs=[model_status])
        
        # Main Interface Tabs
        with gr.Tabs():
            # Chat Tab
            with gr.TabItem("💬 Chat with Apertus"):
                with gr.Row():
                    with gr.Column(scale=2):
                        chat_input = gr.Textbox(
                            label="Your message (any language)",
                            placeholder="Erkläre mir Transparenz in der KI...\nExplique-moi la transparence en IA...\nSpiegami la trasparenza nell'IA...",
                            lines=3
                        )
                        max_tokens = gr.Slider(50, 500, value=300, label="Max Tokens")
                        chat_btn = gr.Button("🇨🇭 Chat", variant="primary")
                    with gr.Column(scale=3):
                        chat_output = gr.Markdown(label="Apertus Response")
                
                chat_btn.click(chat_with_apertus, inputs=[chat_input, max_tokens], outputs=[chat_output])
            
            # Attention Analysis Tab
            with gr.TabItem("👁️ Attention Patterns"):
                gr.HTML("<p><strong>🔍 What you'll see:</strong> Heatmap showing which words the AI 'looks at' while thinking - like tracking eye movements during reading</p>")
                with gr.Row():
                    with gr.Column(scale=1):
                        attention_text = gr.Textbox(
                            label="Text to analyze",
                            value="Die Schweiz ist",
                            info="Enter text to see internal model processing"
                        )
                        attention_layer = gr.Slider(0, 31, value=15, step=1, label="Attention Layer")
                        attention_btn = gr.Button("👁️ Analyze Attention", variant="secondary")
                    with gr.Column(scale=2):
                        attention_plot = gr.Plot(label="Attention Heatmap")
                        attention_insights = gr.Markdown(label="Attention Insights")
                
                attention_btn.click(
                    analyze_attention, 
                    inputs=[attention_text, attention_layer], 
                    outputs=[attention_plot, attention_insights]
                )
            
            # Token Predictions Tab
            with gr.TabItem("🎲 Token Predictions"):
                gr.HTML("<p><strong>🔍 What you'll see:</strong> Top-10 most likely next words with confidence levels - see the AI's 'thought process' for each word</p>")
                with gr.Row():
                    with gr.Column(scale=1):
                        prediction_text = gr.Textbox(
                            label="Text to analyze",
                            value="Die wichtigste Eigenschaft von Apertus ist",
                            info="Enter partial text to see next word predictions"
                        )
                        prediction_btn = gr.Button("🎲 Analyze Predictions", variant="secondary")
                    with gr.Column(scale=2):
                        prediction_plot = gr.Plot(label="Prediction Probabilities")
                        prediction_insights = gr.Markdown(label="Prediction Details")
                
                prediction_btn.click(
                    analyze_token_predictions, 
                    inputs=[prediction_text], 
                    outputs=[prediction_plot, prediction_insights]
                )
            
            # Layer Evolution Tab
            with gr.TabItem("🧠 Layer Evolution"):
                gr.HTML("<p><strong>🔍 What you'll see:</strong> How the AI's 'understanding' develops through 32 neural layers - from basic recognition to deep comprehension</p>")
                with gr.Row():
                    with gr.Column(scale=1):
                        evolution_text = gr.Textbox(
                            label="Text to analyze",
                            value="Schweizer KI-Innovation revolutioniert Transparenz.",
                            info="Enter text to see layer evolution"
                        )
                        evolution_btn = gr.Button("🧠 Analyze Evolution", variant="secondary")
                    with gr.Column(scale=2):
                        evolution_plot = gr.Plot(label="Layer Evolution")
                        evolution_stats = gr.HTML(label="Layer Statistics")
                
                evolution_btn.click(
                    analyze_layer_evolution, 
                    inputs=[evolution_text], 
                    outputs=[evolution_plot, evolution_stats]
                )
            
            # Weight Analysis Tab
            with gr.TabItem("⚖️ Weight Analysis"):
                gr.HTML("<p><strong>🔍 What you'll see:</strong> The actual 'brain connections' (neural weights) that control AI decisions - the learned parameters</p>")
                gr.HTML("<p><em>Real-time analysis of neural network weights following research best practices</em></p>")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        weight_layer_num = gr.Dropdown(
                            choices=list(range(32)), 
                            value=15, 
                            label="Layer Number"
                        )
                        weight_layer_type = gr.Dropdown(
                            choices=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj"],
                            value="self_attn.q_proj",
                            label="Layer Component"
                        )
                        weight_btn = gr.Button("⚖️ Analyze Weights", variant="secondary")
                    
                    with gr.Column(scale=2):
                        weight_plot = gr.Plot(label="Weight Distribution")
                        weight_analysis = gr.Markdown(label="Weight Analysis")
                
                # Gradio handles state much better - no disappearing output!
                weight_btn.click(
                    analyze_weights,
                    inputs=[weight_layer_num, weight_layer_type],
                    outputs=[weight_plot, weight_analysis]
                )
        
        # Footer
        gr.HTML("""
        <div class="footer-section">
            <h2 style="color: white; margin-bottom: 20px; font-size: 2.2em;">🇨🇭 Apertus Swiss AI</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 30px; margin: 30px 0;">
                <div>
                    <h4 style="color: #f8f9fa; margin-bottom: 10px;">🏔️ Swiss Excellence</h4>
                    <p style="color: #bdc3c7; line-height: 1.6;">
                        Built with Swiss precision engineering principles - reliable, transparent, and innovative.
                    </p>
                </div>
                <div>
                    <h4 style="color: #f8f9fa; margin-bottom: 10px;">🔬 Research Grade</h4>
                    <p style="color: #bdc3c7; line-height: 1.6;">
                        Complete model transparency with research-based metrics and analysis tools.
                    </p>
                </div>
                <div>
                    <h4 style="color: #f8f9fa; margin-bottom: 10px;">🌍 Multilingual</h4>
                    <p style="color: #bdc3c7; line-height: 1.6;">
                        Supports German, French, Italian, English, Romansh and Swiss dialects.
                    </p>
                </div>
                <div>
                    <h4 style="color: #f8f9fa; margin-bottom: 10px;">🎓 Educational</h4>
                    <p style="color: #bdc3c7; line-height: 1.6;">
                        Perfect for students, researchers, and anyone curious about AI internals.
                    </p>
                </div>
            </div>
            <div style="border-top: 1px solid #546e7a; padding-top: 20px; margin-top: 30px;">
                <p style="color: #ecf0f1; font-size: 1.3em; margin: 0;">
                    <strong>Experience true AI transparency - Swiss precision meets artificial intelligence</strong>
                </p>
                <p style="color: #95a5a6; margin: 10px 0 0 0;">
                    Powered by Apertus-8B-Instruct-2509 • 8B Parameters • Complete Transparency
                </p>
            </div>
        </div>
        """)
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_port=8501, server_name="0.0.0.0")