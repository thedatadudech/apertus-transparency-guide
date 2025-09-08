"""
🇨🇭 Live Apertus Transparency Dashboard
Real-time visualization of all model internals
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from apertus_core import ApertusCore
from transparency_analyzer import ApertusTransparencyAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit
st.set_page_config(
    page_title="🇨🇭 Apertus Transparency Dashboard",
    page_icon="🇨🇭",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_apertus_model():
    """Load Apertus model with caching"""
    with st.spinner("🧠 Loading Apertus model..."):
        apertus = ApertusCore(enable_transparency=True)
        analyzer = ApertusTransparencyAnalyzer(apertus)
    return apertus, analyzer

def create_attention_heatmap(attention_weights, tokens):
    """Create interactive attention heatmap"""
    fig = px.imshow(
        attention_weights,
        x=tokens,
        y=tokens,
        color_continuous_scale='Blues',
        title="Attention Pattern Heatmap",
        labels={'x': 'Key Tokens', 'y': 'Query Tokens', 'color': 'Attention Weight'}
    )
    
    fig.update_layout(
        width=600,
        height=600,
        xaxis={'side': 'bottom', 'tickangle': 45},
        yaxis={'side': 'left'}
    )
    
    return fig

def create_layer_evolution_plot(layer_stats):
    """Create layer-by-layer evolution plot"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('L2 Norms', 'Mean Activations', 'Std Deviations', 'Activation Ranges'),
        vertical_spacing=0.12
    )
    
    layers = [stat['layer'] for stat in layer_stats]
    
    # L2 Norms
    fig.add_trace(
        go.Scatter(x=layers, y=[stat['l2_norm'] for stat in layer_stats],
                   mode='lines+markers', name='L2 Norm', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Mean Activations
    fig.add_trace(
        go.Scatter(x=layers, y=[stat['mean'] for stat in layer_stats],
                   mode='lines+markers', name='Mean', line=dict(color='red')),
        row=1, col=2
    )
    
    # Std Deviations
    fig.add_trace(
        go.Scatter(x=layers, y=[stat['std'] for stat in layer_stats],
                   mode='lines+markers', name='Std Dev', line=dict(color='green')),
        row=2, col=1
    )
    
    # Activation Ranges
    fig.add_trace(
        go.Scatter(x=layers, y=[stat['max'] - stat['min'] for stat in layer_stats],
                   mode='lines+markers', name='Range', line=dict(color='purple')),
        row=2, col=2
    )
    
    fig.update_layout(height=500, showlegend=False, title="Layer-by-Layer Neural Evolution")
    return fig

def create_prediction_bar_chart(predictions):
    """Create token prediction bar chart"""
    tokens = [pred['token'] for pred in predictions[:10]]
    probs = [pred['probability'] for pred in predictions[:10]]
    
    fig = px.bar(
        x=tokens, y=probs,
        title="Top 10 Token Predictions",
        labels={'x': 'Tokens', 'y': 'Probability'},
        color=probs,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_architecture_overview(model_info):
    """Create model architecture visualization"""
    fig = go.Figure()
    
    # Create architecture diagram
    layers = model_info['num_layers']
    hidden_size = model_info['hidden_size']
    
    # Add layer blocks
    for i in range(min(8, layers)):  # Show first 8 layers
        fig.add_shape(
            type="rect",
            x0=i, y0=0, x1=i+0.8, y1=1,
            fillcolor="lightblue",
            line=dict(color="darkblue", width=2)
        )
        
        fig.add_annotation(
            x=i+0.4, y=0.5,
            text=f"L{i}",
            showarrow=False,
            font=dict(size=10)
        )
    
    if layers > 8:
        fig.add_annotation(
            x=8.5, y=0.5,
            text=f"... {layers-8} more",
            showarrow=False,
            font=dict(size=12)
        )
    
    fig.update_layout(
        title=f"Model Architecture ({layers} layers, {hidden_size}d hidden)",
        xaxis=dict(range=[-0.5, 9], showgrid=False, showticklabels=False),
        yaxis=dict(range=[-0.5, 1.5], showgrid=False, showticklabels=False),
        height=200,
        showlegend=False
    )
    
    return fig

def main():
    """Main dashboard application"""
    
    # Header
    st.title("🇨🇭 Apertus Swiss AI Transparency Dashboard")
    st.markdown("### Real-time visualization of all model internals")
    
    # Sidebar
    st.sidebar.title("🔧 Analysis Settings")
    
    # Load model
    try:
        apertus, analyzer = load_apertus_model()
        st.sidebar.success("✅ Model loaded successfully!")
        
        # Model info in sidebar
        model_info = apertus.get_model_info()
        st.sidebar.markdown("### 📊 Model Info")
        st.sidebar.write(f"**Model**: {model_info['model_name']}")
        st.sidebar.write(f"**Parameters**: {model_info['total_parameters']:,}")
        st.sidebar.write(f"**Layers**: {model_info['num_layers']}")
        st.sidebar.write(f"**Hidden Size**: {model_info['hidden_size']}")
        
        if 'gpu_memory_allocated_gb' in model_info:
            st.sidebar.write(f"**GPU Memory**: {model_info['gpu_memory_allocated_gb']:.1f} GB")
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Input text
    st.markdown("### 📝 Input Text")
    
    example_texts = [
        "Apertus ist ein transparentes KI-Modell aus der Schweiz.",
        "Machine learning requires transparency for trust and understanding.",
        "La Suisse développe des modèles d'intelligence artificielle transparents.",
        "Artificial intelligence should be explainable and interpretable.",
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        input_text = st.text_area(
            "Enter text to analyze:",
            value=example_texts[0],
            height=100
        )
    
    with col2:
        st.markdown("**Examples:**")
        for i, example in enumerate(example_texts):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                input_text = example
                st.rerun()
    
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
        st.stop()
    
    # Analysis settings
    st.sidebar.markdown("### ⚙️ Analysis Options")
    show_architecture = st.sidebar.checkbox("Show Architecture", True)
    show_tokenization = st.sidebar.checkbox("Show Tokenization", True)
    show_layers = st.sidebar.checkbox("Show Layer Analysis", True)
    show_attention = st.sidebar.checkbox("Show Attention", True)
    show_predictions = st.sidebar.checkbox("Show Predictions", True)
    
    attention_layer = st.sidebar.slider("Attention Layer", 0, model_info['num_layers']-1, 15)
    num_predictions = st.sidebar.slider("Top-K Predictions", 5, 20, 10)
    
    # Run analysis
    if st.button("🔍 Analyze Transparency", type="primary"):
        
        with st.spinner("🧠 Analyzing model internals..."):
            
            # Architecture Overview
            if show_architecture:
                st.markdown("## 🏗️ Model Architecture")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    arch_fig = create_architecture_overview(model_info)
                    st.plotly_chart(arch_fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Architecture Details:**")
                    st.write(f"• **Type**: Transformer Decoder")
                    st.write(f"• **Layers**: {model_info['num_layers']}")
                    st.write(f"• **Attention Heads**: {model_info['num_attention_heads']}")
                    st.write(f"• **Hidden Size**: {model_info['hidden_size']}")
                    st.write(f"• **Parameters**: {model_info['total_parameters']:,}")
                    st.write(f"• **Context**: {model_info['max_position_embeddings']:,} tokens")
            
            # Tokenization
            if show_tokenization:
                st.markdown("## 🔤 Tokenization Analysis")
                
                tokens = apertus.tokenizer.tokenize(input_text)
                token_ids = apertus.tokenizer.encode(input_text)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Token Breakdown:**")
                    token_df = pd.DataFrame({
                        'Position': range(1, len(tokens) + 1),
                        'Token': tokens,
                        'Token ID': token_ids[1:] if len(token_ids) > len(tokens) else token_ids
                    })
                    st.dataframe(token_df, use_container_width=True)
                
                with col2:
                    st.markdown("**Statistics:**")
                    st.write(f"• **Original Text**: '{input_text}'")
                    st.write(f"• **Token Count**: {len(tokens)}")
                    st.write(f"• **Characters**: {len(input_text)}")
                    st.write(f"• **Tokens/Characters**: {len(tokens)/len(input_text):.2f}")
            
            # Layer Analysis
            if show_layers:
                st.markdown("## 🧠 Layer-by-Layer Processing")
                
                # Get hidden states
                inputs = apertus.tokenizer(input_text, return_tensors="pt")
                with torch.no_grad():
                    outputs = apertus.model(**inputs, output_hidden_states=True)
                
                hidden_states = outputs.hidden_states
                
                # Analyze sampled layers
                layer_stats = []
                sample_layers = list(range(0, len(hidden_states), max(1, len(hidden_states)//8)))
                
                for layer_idx in sample_layers:
                    layer_state = hidden_states[layer_idx][0]
                    
                    layer_stats.append({
                        'layer': layer_idx,
                        'l2_norm': torch.norm(layer_state, dim=-1).mean().item(),
                        'mean': layer_state.mean().item(),
                        'std': layer_state.std().item(),
                        'max': layer_state.max().item(),
                        'min': layer_state.min().item()
                    })
                
                # Plot evolution
                evolution_fig = create_layer_evolution_plot(layer_stats)
                st.plotly_chart(evolution_fig, use_container_width=True)
                
                # Layer statistics table
                st.markdown("**Layer Statistics:**")
                stats_df = pd.DataFrame(layer_stats)
                stats_df = stats_df.round(4)
                st.dataframe(stats_df, use_container_width=True)
            
            # Attention Analysis
            if show_attention:
                st.markdown("## 👁️ Attention Pattern Analysis")
                
                # Get attention weights
                with torch.no_grad():
                    outputs = apertus.model(**inputs, output_attentions=True)
                
                attentions = outputs.attentions
                tokens = apertus.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                
                if attention_layer < len(attentions):
                    attention_weights = attentions[attention_layer][0]  # Remove batch dim
                    avg_attention = attention_weights.mean(dim=0).cpu().numpy()  # Average heads
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        attention_fig = create_attention_heatmap(avg_attention, tokens)
                        st.plotly_chart(attention_fig, use_container_width=True)
                    
                    with col2:
                        st.markdown(f"**Layer {attention_layer} Statistics:**")
                        st.write(f"• **Attention Heads**: {attention_weights.shape[0]}")
                        st.write(f"• **Matrix Size**: {avg_attention.shape}")
                        st.write(f"• **Entropy**: {-np.sum(avg_attention * np.log(avg_attention + 1e-12)):.2f}")
                        
                        # Most attended tokens
                        attention_received = avg_attention.sum(axis=0)
                        top_tokens = np.argsort(attention_received)[-3:][::-1]
                        
                        st.markdown("**Most Attended Tokens:**")
                        for i, token_idx in enumerate(top_tokens):
                            if token_idx < len(tokens):
                                st.write(f"{i+1}. '{tokens[token_idx]}' ({attention_received[token_idx]:.3f})")
                else:
                    st.error(f"Layer {attention_layer} not available. Max layer: {len(attentions)-1}")
            
            # Prediction Analysis
            if show_predictions:
                st.markdown("## 🎲 Next Token Predictions")
                
                # Get predictions
                with torch.no_grad():
                    outputs = apertus.model(**inputs)
                    logits = outputs.logits[0, -1, :]
                
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probabilities, num_predictions)
                
                # Prepare prediction data
                predictions = []
                for i in range(num_predictions):
                    token_id = top_indices[i].item()
                    token = apertus.tokenizer.decode([token_id])
                    prob = top_probs[i].item()
                    logit = logits[token_id].item()
                    
                    predictions.append({
                        'rank': i + 1,
                        'token': token,
                        'probability': prob,
                        'logit': logit
                    })
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    pred_fig = create_prediction_bar_chart(predictions)
                    st.plotly_chart(pred_fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Prediction Statistics:**")
                    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12)).item()
                    max_prob = probabilities.max().item()
                    top_k_sum = top_probs.sum().item()
                    
                    st.write(f"• **Entropy**: {entropy:.2f}")
                    st.write(f"• **Max Probability**: {max_prob:.1%}")
                    st.write(f"• **Top-{num_predictions} Sum**: {top_k_sum:.1%}")
                    
                    confidence = "High" if max_prob > 0.5 else "Medium" if max_prob > 0.2 else "Low"
                    st.write(f"• **Confidence**: {confidence}")
                    
                    # Predictions table
                    st.markdown("**Top Predictions:**")
                    pred_df = pd.DataFrame(predictions)
                    pred_df['probability'] = pred_df['probability'].apply(lambda x: f"{x:.1%}")
                    pred_df['logit'] = pred_df['logit'].apply(lambda x: f"{x:+.2f}")
                    st.dataframe(pred_df[['rank', 'token', 'probability']], use_container_width=True)
            
            # Summary
            st.markdown("## 📊 Transparency Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Tokens Analyzed", len(tokens))
            
            with col2:
                st.metric("Layers Processed", len(hidden_states))
            
            with col3:
                st.metric("Attention Heads", model_info['num_attention_heads'])
            
            with col4:
                if 'gpu_memory_allocated_gb' in model_info:
                    st.metric("GPU Memory", f"{model_info['gpu_memory_allocated_gb']:.1f} GB")
                else:
                    st.metric("Parameters", f"{model_info['total_parameters']:,}")
            
            st.success("✅ Complete transparency analysis finished!")
            st.info("🇨🇭 This demonstrates the full transparency capabilities of Apertus Swiss AI - "
                   "every layer, attention pattern, and prediction is completely visible!")

    # Footer
    st.markdown("---")
    st.markdown("🇨🇭 **Apertus Swiss AI** - The world's most transparent language model")

if __name__ == "__main__":
    main()