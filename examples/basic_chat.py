"""
Basic Chat Example with Apertus Swiss AI
Simple conversation interface demonstrating core functionality
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from apertus_core import ApertusCore


def main():
    """Main chat interface"""
    print("🇨🇭 Apertus Swiss AI - Basic Chat Example")
    print("=" * 50)
    print("Loading model... (this may take a few minutes)")
    
    try:
        # Initialize Apertus
        apertus = ApertusCore()
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model info: {apertus.get_model_info()['total_parameters']:,} parameters")
        print("\nType 'quit' to exit, 'clear' to clear history")
        print("Try different languages: German, French, Italian, English")
        print("-" * 50)
        
        while True:
            # Get user input
            user_input = input("\n🙋 You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == 'quit':
                print("👋 Auf Wiedersehen! Au revoir! Goodbye!")
                break
                
            if user_input.lower() == 'clear':
                apertus.clear_history()
                print("🗑️ Conversation history cleared!")
                continue
            
            # Generate response
            print("🤔 Thinking...")
            try:
                response = apertus.chat(user_input)
                print(f"🇨🇭 Apertus: {response}")
                
            except Exception as e:
                print(f"❌ Error generating response: {str(e)}")
    
    except Exception as e:
        print(f"❌ Failed to initialize Apertus: {str(e)}")
        print("Make sure you have the required dependencies installed:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
