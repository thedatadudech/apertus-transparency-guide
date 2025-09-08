"""
Multilingual Swiss Assistant
Specialized implementation for Swiss multilingual use cases
"""

from typing import Dict, List, Optional, Any
import logging
from .apertus_core import ApertusCore

logger = logging.getLogger(__name__)


class SwissMultilingualAssistant:
    """
    Swiss multilingual assistant with language detection and context switching
    
    Handles seamless conversation across German, French, Italian, English,
    and Romansh with Swiss cultural context and precision.
    """
    
    def __init__(self, apertus_core: Optional[ApertusCore] = None):
        """
        Initialize multilingual assistant
        
        Args:
            apertus_core: Initialized ApertusCore instance, or None to create new
        """
        if apertus_core is None:
            self.apertus = ApertusCore()
        else:
            self.apertus = apertus_core
        
        self.conversation_history = []
        self.language_context = {}
        
        # Swiss language configurations
        self.supported_languages = {
            "German": "de",
            "French": "fr", 
            "Italian": "it",
            "English": "en",
            "Romansh": "rm"
        }
        
        # Language-specific system messages
        self.system_messages = {
            "de": """Du bist ein hilfsreicher Schweizer AI-Assistent. Du verstehst die Schweizer Kultur, 
                     Gesetze und Gepflogenheiten. Antworte präzise und höflich auf Deutsch. 
                     Berücksichtige schweizerische Besonderheiten in deinen Antworten.""",
            
            "fr": """Tu es un assistant IA suisse utile. Tu comprends la culture, les lois et 
                     les coutumes suisses. Réponds de manière précise et polie en français. 
                     Prends en compte les spécificités suisses dans tes réponses.""",
            
            "it": """Sei un utile assistente IA svizzero. Comprendi la cultura, le leggi e 
                     le usanze svizzere. Rispondi in modo preciso e cortese in italiano. 
                     Considera le specificità svizzere nelle tue risposte.""",
            
            "en": """You are a helpful Swiss AI assistant. You understand Swiss culture, 
                     laws, and customs. Respond precisely and politely in English. 
                     Consider Swiss specificities in your responses.""",
            
            "rm": """Ti es in assistent IA svizzer d'agid. Ti chapeschas la cultura, 
                     las legas e las usanzas svizras. Respunda precis e curtes en rumantsch. 
                     Consideresch las specificitads svizras en tias respostas."""
        }
        
        logger.info("🇨🇭 Swiss Multilingual Assistant initialized")
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on common patterns
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detected language code
        """
        text_lower = text.lower()
        
        # German indicators
        if any(word in text_lower for word in ['der', 'die', 'das', 'und', 'ist', 'sind', 'haben', 'können', 'schweiz']):
            return "de"
        
        # French indicators  
        elif any(word in text_lower for word in ['le', 'la', 'les', 'et', 'est', 'sont', 'avoir', 'être', 'suisse']):
            return "fr"
        
        # Italian indicators
        elif any(word in text_lower for word in ['il', 'la', 'gli', 'le', 'è', 'sono', 'avere', 'essere', 'svizzera']):
            return "it"
        
        # Romansh indicators (limited)
        elif any(word in text_lower for word in ['il', 'la', 'els', 'las', 'è', 'èn', 'avair', 'esser', 'svizra']):
            return "rm"
        
        # Default to English
        else:
            return "en"
    
    def chat(
        self,
        message: str,
        target_language: Optional[str] = None,
        maintain_context: bool = True,
        **generation_kwargs
    ) -> str:
        """
        Chat with automatic language detection and appropriate response
        
        Args:
            message: User message in any supported language
            target_language: Force specific language (None for auto-detection)
            maintain_context: Whether to maintain conversation context
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Assistant response in appropriate language
        """
        # Use intelligent system message that lets Apertus detect language automatically
        if target_language:
            # If specific language requested, use that system message
            system_message = self.system_messages.get(target_language, self.system_messages["en"])
        else:
            # Let Apertus automatically detect and respond in appropriate language
            system_message = """You are a helpful Swiss AI assistant. You understand all Swiss languages: German, French, Italian, English, and Romansh. 
            Detect the language of the user's message and respond in the SAME language. 
            If the message is in German (including Swiss German), respond in German.
            If the message is in French, respond in French.
            If the message is in Italian, respond in Italian.
            If the message is in Romansh, respond in Romansh.
            If the message is in English, respond in English.
            Consider Swiss cultural context and be precise and helpful."""
        
        # Build context if maintaining history
        context = ""
        if maintain_context and self.conversation_history:
            recent_history = self.conversation_history[-3:]  # Last 3 exchanges
            context = "\n".join([
                f"Human: {h['human']}\nAssistant: {h['assistant']}"
                for h in recent_history
            ]) + "\n\n"
        
        # Create full prompt
        full_prompt = f"{context}Human: {message}\nAssistant:"
        
        # Generate response
        response = self.apertus.generate_response(
            full_prompt,
            system_message=system_message,
            **generation_kwargs
        )
        
        # Update conversation history  
        if maintain_context:
            self.conversation_history.append({
                "human": message,
                "assistant": response,
                "language": "auto-detected",
                "timestamp": self._get_timestamp()
            })
        
        # Update language context
        self.language_context["auto"] = self.language_context.get("auto", 0) + 1
        
        logger.info(f"Response generated via auto-detection")
        return response
    
    def translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> str:
        """
        Translate text between supported languages
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Translated text
        """
        # Language name mapping
        lang_names = {
            "de": "German", "fr": "French", "it": "Italian", 
            "en": "English", "rm": "Romansh"
        }
        
        source_name = lang_names.get(source_language, source_language)
        target_name = lang_names.get(target_language, target_language)
        
        translation_prompt = f"""Translate the following text from {source_name} to {target_name}. 
        Maintain the original meaning, tone, and any Swiss cultural context.
        
        Text to translate: {text}
        
        Translation:"""
        
        response = self.apertus.generate_response(
            translation_prompt,
            temperature=0.3,  # Lower temperature for more consistent translation
            system_message="You are a professional Swiss translator with expertise in all Swiss languages."
        )
        
        return response.strip()
    
    def get_swiss_context_response(
        self,
        question: str,
        context_type: str = "general"
    ) -> str:
        """
        Get response with specific Swiss context
        
        Args:
            question: User question
            context_type: Type of Swiss context (legal, cultural, business, etc.)
            
        Returns:
            Context-aware response
        """
        context_prompts = {
            "legal": "Consider Swiss legal framework, cantonal differences, and federal regulations.",
            "cultural": "Consider Swiss cultural values, traditions, and regional differences.",
            "business": "Consider Swiss business practices, work culture, and economic environment.",
            "healthcare": "Consider Swiss healthcare system, insurance, and medical practices.",
            "education": "Consider Swiss education system, universities, and vocational training.",
            "government": "Consider Swiss political system, direct democracy, and federalism."
        }
        
        context_instruction = context_prompts.get(context_type, "Consider general Swiss context.")
        detected_lang = self.detect_language(question)
        
        swiss_prompt = f"""Answer the following question with specific Swiss context. 
        {context_instruction}
        
        Question: {question}
        
        Answer:"""
        
        return self.apertus.generate_response(
            swiss_prompt,
            system_message=self.system_messages.get(detected_lang, self.system_messages["en"])
        )
    
    def switch_language(self, target_language: str) -> str:
        """
        Switch conversation language and confirm
        
        Args:
            target_language: Target language code
            
        Returns:
            Confirmation message in target language
        """
        confirmations = {
            "de": "Gerne! Ich antworte ab jetzt auf Deutsch. Wie kann ich Ihnen helfen?",
            "fr": "Avec plaisir! Je répondrai désormais en français. Comment puis-je vous aider?",
            "it": "Volentieri! D'ora in poi risponderò in italiano. Come posso aiutarla?",
            "en": "Certainly! I'll now respond in English. How can I help you?",
            "rm": "Gugent! Jau rispund ussa en rumantsch. Co poss jau gidar a vus?"
        }
        
        return confirmations.get(target_language, confirmations["en"])
    
    def get_language_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about language usage in conversation
        
        Returns:
            Dictionary with language usage statistics
        """
        total_exchanges = len(self.conversation_history)
        
        if total_exchanges == 0:
            return {"total_exchanges": 0, "languages_used": {}}
        
        language_counts = {}
        for exchange in self.conversation_history:
            lang = exchange.get("language", "unknown")
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Calculate percentages
        language_percentages = {
            lang: (count / total_exchanges) * 100 
            for lang, count in language_counts.items()
        }
        
        return {
            "total_exchanges": total_exchanges,
            "languages_used": language_counts,
            "language_percentages": language_percentages,
            "most_used_language": max(language_counts.items(), key=lambda x: x[1])[0] if language_counts else None
        }
    
    def clear_history(self):
        """Clear conversation history and language context"""
        self.conversation_history = []
        self.language_context = {}
        logger.info("Conversation history and language context cleared")
    
    def export_conversation(self, format: str = "text") -> str:
        """
        Export conversation history in specified format
        
        Args:
            format: Export format ('text', 'json', 'csv')
            
        Returns:
            Formatted conversation data
        """
        if format == "text":
            return self._export_as_text()
        elif format == "json":
            return self._export_as_json()
        elif format == "csv":
            return self._export_as_csv()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_as_text(self) -> str:
        """Export conversation as formatted text"""
        if not self.conversation_history:
            return "No conversation history available."
        
        output = "🇨🇭 Swiss Multilingual Conversation Export\n"
        output += "=" * 50 + "\n\n"
        
        for i, exchange in enumerate(self.conversation_history, 1):
            lang_flag = {"de": "🇩🇪", "fr": "🇫🇷", "it": "🇮🇹", "en": "🇬🇧", "rm": "🏔️"}.get(exchange.get("language", "en"), "🌍")
            output += f"Exchange {i} {lang_flag} ({exchange.get('language', 'unknown')})\n"
            output += f"Human: {exchange['human']}\n"
            output += f"Assistant: {exchange['assistant']}\n"
            output += f"Time: {exchange.get('timestamp', 'N/A')}\n\n"
        
        return output
    
    def _export_as_json(self) -> str:
        """Export conversation as JSON"""
        import json
        return json.dumps({
            "conversation_history": self.conversation_history,
            "language_statistics": self.get_language_statistics()
        }, indent=2, ensure_ascii=False)
    
    def _export_as_csv(self) -> str:
        """Export conversation as CSV"""
        if not self.conversation_history:
            return "exchange_id,language,human_message,assistant_response,timestamp\n"
        
        output = "exchange_id,language,human_message,assistant_response,timestamp\n"
        for i, exchange in enumerate(self.conversation_history, 1):
            # Escape CSV fields
            human_msg = exchange['human'].replace('"', '""').replace('\n', ' ')
            assistant_msg = exchange['assistant'].replace('"', '""').replace('\n', ' ')
            
            output += f'{i},"{exchange.get("language", "unknown")}","{human_msg}","{assistant_msg}","{exchange.get("timestamp", "")}"\n'
        
        return output
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def demo_multilingual_capabilities(self) -> Dict[str, str]:
        """
        Demonstrate multilingual capabilities with sample responses
        
        Returns:
            Dictionary with sample responses in each language
        """
        demo_prompts = {
            "de": "Erkläre mir das Schweizer Bildungssystem.",
            "fr": "Explique-moi le système politique suisse.",
            "it": "Descrivi la cultura svizzera.",
            "en": "What makes Swiss engineering special?",
            "rm": "Co è special tar la Svizra?"
        }
        
        results = {}
        for lang, prompt in demo_prompts.items():
            try:
                response = self.chat(prompt, target_language=lang, maintain_context=False)
                results[lang] = response
            except Exception as e:
                results[lang] = f"Error: {str(e)}"
        
        return results
    
    def __repr__(self):
        """String representation of the assistant"""
        total_exchanges = len(self.conversation_history)
        most_used = "None"
        
        if self.language_context:
            most_used = max(self.language_context.items(), key=lambda x: x[1])[0]
        
        return f"SwissMultilingualAssistant(exchanges={total_exchanges}, primary_language={most_used})"
