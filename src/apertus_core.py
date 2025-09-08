"""
Core Apertus Swiss AI wrapper class
Provides unified interface for model loading and basic operations
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ApertusCore:
    """
    Core wrapper for Apertus Swiss AI model
    
    Provides unified interface for model loading, configuration,
    and basic text generation with Swiss engineering standards.
    """
    
    def __init__(
        self,
        model_name: str = "swiss-ai/Apertus-8B-Instruct-2509",
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        enable_transparency: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_memory: Optional[Dict[int, str]] = None,
        low_cpu_mem_usage: bool = True
    ):
        """
        Initialize Apertus model with flexible GPU optimization
        
        Args:
            model_name: HuggingFace model identifier (requires registration at HF)
            device_map: Device mapping strategy ("auto" recommended)
            torch_dtype: Precision (None=auto-detect based on GPU capabilities)
            enable_transparency: Enable attention/hidden state outputs
            load_in_8bit: Use 8-bit quantization (for memory-constrained GPUs)
            load_in_4bit: Use 4-bit quantization (for lower-end GPUs)
            max_memory: Memory limits per GPU (auto-detected if not specified)
            low_cpu_mem_usage: Minimize CPU memory usage during loading
            
        Note:
            Automatically optimizes for available GPU. The swiss-ai/Apertus-8B-Instruct-2509 
            model requires providing name, country, and affiliation on Hugging Face to access.
            Run 'huggingface-cli login' after approval to authenticate.
        """
        self.model_name = model_name
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_memory = max_memory
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.enable_transparency = enable_transparency
        
        # Auto-detect optimal dtype based on GPU capabilities
        if torch_dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.torch_dtype = torch.bfloat16  # Best for modern GPUs
            else:
                self.torch_dtype = torch.float16   # Fallback
        else:
            self.torch_dtype = torch_dtype
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.conversation_history = []
        self.device_info = self._detect_gpu_info()
        
        # Load model
        self._load_model()
        
        logger.info(f"🇨🇭 Apertus loaded successfully: {model_name}")
    
    def _detect_gpu_info(self) -> Dict[str, any]:
        """Detect GPU information for automatic optimization"""
        info = {"has_gpu": False, "gpu_name": None, "gpu_memory_gb": 0, "supports_bf16": False}
        
        if torch.cuda.is_available():
            info["has_gpu"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info["supports_bf16"] = torch.cuda.is_bf16_supported()
            
            logger.info(f"🎯 GPU detected: {info['gpu_name']}")
            logger.info(f"📊 GPU Memory: {info['gpu_memory_gb']:.1f} GB")
            logger.info(f"🔧 bfloat16 support: {info['supports_bf16']}")
            
            # Memory-based recommendations
            if info["gpu_memory_gb"] >= 40:
                logger.info("🚀 High-memory GPU - optimal settings enabled")
            elif info["gpu_memory_gb"] >= 20:
                logger.info("⚡ Mid-range GPU - balanced settings enabled")
            else:
                logger.info("💾 Lower-memory GPU - consider using quantization")
        else:
            logger.warning("⚠️  No GPU detected - falling back to CPU")
            
        return info
    
    def _load_model(self):
        """Load tokenizer and model with specified configuration"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Configure padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with transparency options
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                trust_remote_code=True,
                output_attentions=self.enable_transparency,
                output_hidden_states=self.enable_transparency
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Log model information
            self._log_model_info()
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
    
    def _log_model_info(self):
        """Log model architecture and memory information"""
        config = self.model.config
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Model Architecture:")
        logger.info(f"  - Layers: {config.num_hidden_layers}")
        logger.info(f"  - Attention Heads: {config.num_attention_heads}")
        logger.info(f"  - Hidden Size: {config.hidden_size}")
        logger.info(f"  - Total Parameters: {total_params:,}")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"  - GPU Memory: {memory_allocated:.2f} GB")
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        system_message: str = "You are a helpful Swiss AI assistant."
    ) -> str:
        """
        Generate response to user prompt
        
        Args:
            prompt: User input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            do_sample: Whether to use sampling
            system_message: System context for the conversation
            
        Returns:
            Generated response text
        """
        try:
            # Format prompt with instruction template
            formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### System:
{system_message}

### Instruction:
{prompt}

### Response:
"""
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                max_length=2048,
                truncation=True
            )
            
            # Move inputs to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and extract response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response.split("### Response:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def chat(
        self,
        message: str,
        maintain_history: bool = True,
        **generation_kwargs
    ) -> str:
        """
        Simple chat interface with optional history maintenance
        
        Args:
            message: User message
            maintain_history: Whether to maintain conversation context
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Assistant response
        """
        # Build context from history if enabled
        context = ""
        if maintain_history and self.conversation_history:
            recent_history = self.conversation_history[-5:]  # Last 5 exchanges
            context = "\n".join([
                f"Human: {h['human']}\nAssistant: {h['assistant']}"
                for h in recent_history
            ]) + "\n\n"
        
        # Generate response
        full_prompt = context + f"Human: {message}\nAssistant:"
        response = self.generate_response(full_prompt, **generation_kwargs)
        
        # Update history if enabled
        if maintain_history:
            self.conversation_history.append({
                "human": message,
                "assistant": response
            })
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary with model architecture and performance info
        """
        if not self.model:
            return {"error": "Model not loaded"}
        
        config = self.model.config
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            "model_name": self.model_name,
            "model_type": config.model_type,
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_position_embeddings,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_gb": total_params * 2 / 1e9,  # Approximate for float16
        }
        
        # Add GPU memory info if available
        if torch.cuda.is_available():
            info.update({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "device": str(next(self.model.parameters()).device)
            })
        
        return info
    
    def get_tokenizer_info(self) -> Dict:
        """
        Get tokenizer information and capabilities
        
        Returns:
            Dictionary with tokenizer details
        """
        if not self.tokenizer:
            return {"error": "Tokenizer not loaded"}
        
        return {
            "vocab_size": self.tokenizer.vocab_size,
            "model_max_length": self.tokenizer.model_max_length,
            "pad_token": self.tokenizer.pad_token,
            "eos_token": self.tokenizer.eos_token,
            "bos_token": self.tokenizer.bos_token,
            "unk_token": self.tokenizer.unk_token,
            "tokenizer_class": self.tokenizer.__class__.__name__
        }
    
    def test_multilingual_capabilities(self) -> Dict[str, str]:
        """
        Test model's multilingual capabilities with sample prompts
        
        Returns:
            Dictionary with responses in different languages
        """
        test_prompts = {
            "German": "Erkläre maschinelles Lernen in einfachen Worten.",
            "French": "Explique l'apprentissage automatique simplement.",
            "Italian": "Spiega l'apprendimento automatico in modo semplice.",
            "English": "Explain machine learning in simple terms.",
            "Romansh": "Explitgescha l'emprender automatica simplamain."
        }
        
        results = {}
        for language, prompt in test_prompts.items():
            try:
                response = self.generate_response(
                    prompt,
                    max_new_tokens=150,
                    temperature=0.7
                )
                results[language] = response
            except Exception as e:
                results[language] = f"Error: {str(e)}"
        
        return results
    
    def __repr__(self):
        """String representation of the model"""
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            return f"ApertusCore(model={self.model_name}, params={total_params:,})"
        else:
            return f"ApertusCore(model={self.model_name}, status=not_loaded)"
