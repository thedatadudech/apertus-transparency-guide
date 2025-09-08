"""
Multilingual Demo with Apertus Swiss AI
Demonstrates seamless language switching and cultural context
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from multilingual_assistant import SwissMultilingualAssistant


def demo_language_switching():
    """Demonstrate automatic language switching"""
    print("🌍 Multilingual Language Switching Demo")
    print("=" * 50)
    
    assistant = SwissMultilingualAssistant()
    
    # Test prompts in different languages
    test_prompts = [
        ("Guten Tag! Wie funktioniert das Schweizer Bildungssystem?", "German"),
        ("Bonjour! Comment puis-je ouvrir un compte bancaire en Suisse?", "French"),
        ("Ciao! Puoi spiegarmi il sistema sanitario svizzero?", "Italian"),
        ("Hello! What are the benefits of living in Switzerland?", "English"),
        ("Allegra! Co poss far per emprender il rumantsch?", "Romansh")
    ]
    
    for prompt, language in test_prompts:
        print(f"\n🗣️ Testing {language}:")
        print(f"User: {prompt}")
        print("🤔 Processing...")
        
        try:
            response = assistant.chat(prompt, maintain_context=False)
            print(f"🇨🇭 Apertus: {response}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 40)
    
    # Show language statistics
    stats = assistant.get_language_statistics()
    print(f"\n📊 Language Usage Statistics:")
    for lang, count in stats['languages_used'].items():
        percentage = stats['language_percentages'][lang]
        print(f"  {lang}: {count} messages ({percentage:.1f}%)")


def demo_context_switching():
    """Demonstrate context-aware language switching"""
    print("\n🔄 Context-Aware Language Switching Demo")
    print("=" * 50)
    
    assistant = SwissMultilingualAssistant()
    
    # Conversation with language switching
    conversation_flow = [
        ("Kannst du mir bei meinen Steuern helfen?", "Starting in German"),
        ("Actually, can you explain it in English please?", "Switching to English"),
        ("Merci, mais peux-tu maintenant l'expliquer en français?", "Switching to French"),
        ("Perfetto! Ora continua in italiano per favore.", "Switching to Italian")
    ]
    
    print("Starting contextual conversation...")
    
    for message, description in conversation_flow:
        print(f"\n📝 {description}")
        print(f"User: {message}")
        
        try:
            response = assistant.chat(message, maintain_context=True)
            print(f"🇨🇭 Apertus: {response}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n💾 Exporting conversation...")
    conversation_export = assistant.export_conversation("text")
    print(conversation_export[:500] + "..." if len(conversation_export) > 500 else conversation_export)


def demo_swiss_context():
    """Demonstrate Swiss cultural context understanding"""
    print("\n🏔️ Swiss Cultural Context Demo")
    print("=" * 50)
    
    assistant = SwissMultilingualAssistant()
    
    swiss_context_questions = [
        ("Wie funktioniert die direkte Demokratie in der Schweiz?", "legal"),
        ("Was sind typisch schweizerische Werte?", "cultural"),
        ("Wie gründe ich ein Unternehmen in der Schweiz?", "business"),
        ("Welche Krankenversicherung brauche ich?", "healthcare"),
        ("Comment fonctionne le système de formation dual?", "education")
    ]
    
    for question, context_type in swiss_context_questions:
        print(f"\n🎯 Context: {context_type}")
        print(f"Question: {question}")
        
        try:
            response = assistant.get_swiss_context_response(question, context_type)
            print(f"🇨🇭 Swiss Context Response: {response}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 30)


def demo_translation_capabilities():
    """Demonstrate translation between Swiss languages"""
    print("\n🔄 Translation Demo")
    print("=" * 50)
    
    assistant = SwissMultilingualAssistant()
    
    original_text = "Die Schweiz ist ein mehrsprachiges Land mit vier Amtssprachen."
    
    translations = [
        ("de", "fr", "German to French"),
        ("de", "it", "German to Italian"),
        ("de", "en", "German to English"),
        ("de", "rm", "German to Romansh")
    ]
    
    print(f"Original text: {original_text}")
    
    for source, target, description in translations:
        print(f"\n🔄 {description}:")
        
        try:
            translated = assistant.translate_text(original_text, source, target)
            print(f"Translation: {translated}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def interactive_demo():
    """Interactive multilingual chat"""
    print("\n💬 Interactive Multilingual Chat")
    print("=" * 50)
    print("Chat in any language! Type 'stats' for statistics, 'quit' to exit")
    
    assistant = SwissMultilingualAssistant()
    
    while True:
        user_input = input("\n🙋 You: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() == 'quit':
            break
            
        if user_input.lower() == 'stats':
            stats = assistant.get_language_statistics()
            print("📊 Conversation Statistics:")
            print(f"Total exchanges: {stats['total_exchanges']}")
            for lang, count in stats['languages_used'].items():
                print(f"  {lang}: {count} messages")
            continue
        
        try:
            response = assistant.chat(user_input)
            print(f"🇨🇭 Apertus: {response}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Final statistics
    stats = assistant.get_language_statistics()
    if stats['total_exchanges'] > 0:
        print(f"\n📊 Final Statistics:")
        print(f"Total exchanges: {stats['total_exchanges']}")
        print(f"Most used language: {stats['most_used_language']}")


def main():
    """Main demo function"""
    print("🇨🇭 Apertus Swiss AI - Multilingual Demo")
    print("Loading Swiss Multilingual Assistant...")
    
    demos = [
        ("1", "Language Switching Demo", demo_language_switching),
        ("2", "Context Switching Demo", demo_context_switching),
        ("3", "Swiss Cultural Context Demo", demo_swiss_context),
        ("4", "Translation Demo", demo_translation_capabilities),
        ("5", "Interactive Chat", interactive_demo)
    ]
    
    print("\nAvailable demos:")
    for num, name, _ in demos:
        print(f"  {num}. {name}")
    
    print("  0. Run all demos")
    
    choice = input("\nChoose demo (0-5): ").strip()
    
    if choice == "0":
        for num, name, demo_func in demos[:-1]:  # Exclude interactive demo
            print(f"\n{'='*20} {name} {'='*20}")
            try:
                demo_func()
            except Exception as e:
                print(f"❌ Demo failed: {str(e)}")
    else:
        for num, name, demo_func in demos:
            if choice == num:
                try:
                    demo_func()
                except Exception as e:
                    print(f"❌ Demo failed: {str(e)}")
                break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
