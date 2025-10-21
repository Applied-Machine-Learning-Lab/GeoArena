import os
import json
from datetime import datetime
import time
import threading

ARENA_NAME = "# GeoArena 🌍"

# Ollama API configuration
API_URL = 'https://xxx'
API_KEY = 'sk-xxx'
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

FALLBACK_MODELS = [
    ("openai/gpt-4.1", "GPT-4.1"),
    ("openai/gpt-4o-mini", "GPT-4o Mini"),
    ("openai/gpt-4.1-mini", "GPT-4.1 Mini"),
    ("openai/gpt-4.1-nano", "GPT-4.1 Nano"),
    ("openai/gpt-4o", "GPT-4o"),
    ("google/gemini-2.5-pro", "Gemini 2.5 Pro"),
    ("google/gemini-2.5-flash", " Gemini 2.5 Flash"),
    ("anthropic/claude-sonnet-4", "Claude Sonnet 4"),
    ("anthropic/claude-opus-4", "Claude Opus 4"),
    ("meta-llama/llama-4-maverick", "Llama 4 Maverick"),
    ("meta-llama/llama-4-scout", "Llama 4 Scout"),
    ("google/gemma-3-27b-it", "Gemma 3 27B"),
    ("google/gemma-3-12b-it", "Gemma 3 12B"),
    ("google/gemma-3-4b-it", "Gemma 3 4B"),
    ("qwen/qwen2.5-vl-72b-instruct", "Qwen 2.5 VL 72B"),
    ("qwen/qwen2.5-vl-32b-instruct", "Qwen 2.5 VL 32B"),
    ("qwen/qwen-2.5-vl-7b-instruct", "Qwen 2.5 VL 7B"),
]

def load_approved_models():
    return FALLBACK_MODELS

def get_approved_models():
    return load_approved_models()

# Add these constants
MODEL_REFRESH_INTERVAL = 3600  # Check every 1 hour
_last_model_check = 0  # Track last check time

def get_approved_models():
    """Get the current list of approved models with periodic refresh."""
    global _last_model_check
    
    current_time = time.time()
    # Check if we need to refresh (if it's been more than MODEL_REFRESH_INTERVAL seconds)
    if not hasattr(get_approved_models, '_models') or \
       (current_time - _last_model_check) > MODEL_REFRESH_INTERVAL:
        get_approved_models._models = load_approved_models()
        _last_model_check = current_time
    
    return get_approved_models._models

def refresh_approved_models():
    """Force refresh of the approved models list."""
    if hasattr(get_approved_models, '_models'):
        delattr(get_approved_models, '_models')
    return get_approved_models()

# Example prompts
# example_prompts = [
#     "Explain how the internet works to someone from the 1800s.",
#     "Design a new sport that combines elements of chess and basketball.",
#     "Explain the idea behind Bitcoin using only food analogies.",
#     "Write a dialogue between Socrates and a modern AI researcher about the nature of intelligence.",
#     "Describe a color to someone who has been blind since birth.",
#     "Compose a short speech convincing aliens not to destroy Earth.",
#     "Explain the concept of infinity using a metaphor that doesn't involve numbers or math.",
#     "Write a job description for a time traveler's assistant.",
#     "If an AI and a human exchange messages 100 years apart, what does that mean for their relationship and understanding of time?",
#     "How would you explain the concept of nostalgia to an AI that experiences all moments simultaneously?",
#     "Create a thought experiment about an AI that can only communicate across decades - how would that shape its perspective?",
#     "If AI experiences time non-linearly while humans experience it linearly, what philosophical questions does this raise?",
#     "Imagine teaching the emotional weight of waiting to an AI that has no concept of anticipation - how would you approach it?",
#     "Create a new philosophical thought experiment that challenges our understanding of reality.",
#     "Describe how you would explain the concept of death to an immortal being.",
#     "Invent a new emotion and describe how it feels, when it occurs, and its evolutionary purpose.",
#     "Write a conversation between your future self and your past self, discussing the most important life lessons.",
#     "Describe a day in the life of a sentient cloud.",
#     "Create a new system of government based on the behavior of honeybees.",
#     "Explain quantum entanglement using only elements from a typical kitchen.",
#     "Design a universal language that could be understood by all species on Earth.",
#     "Write a creation myth for the Internet age.",
#     "Describe how you would teach empathy to an artificial intelligence.",
#     "Invent a new primary color and explain its impact on the world.",
#     "Compose a poem that can be read forwards and backwards, with different meanings in each direction.",
#     "What are the main causes of climate change?",
#     "Describe the process of photosynthesis in simple terms.",
#     "Explain the concept of supply and demand in economics.",
#     "What are the key differences between democracy and autocracy?",
#     "How does the human immune system work?",
#     "Summarize the plot of Romeo and Juliet in three sentences.",
#     "What are the main features of the solar system?",
#     "Explain the theory of evolution by natural selection.",
#     "What are the primary functions of the United Nations?",
#     "Describe the water cycle and its importance to life on Earth.",
#     "Explain the biggest differences between Keynesian and Austrian economics.",
#     "What are the main principles of the scientific method?",
#     "Describe a world where humans communicate solely through music.",
#     "Explain the concept of blockchain to a medieval blacksmith.",
#     "Design a sustainable city that could exist on Mars.",
#     "Write a short story where the protagonist is a sentient algorithm.",
#     "Describe how you would reorganize the education system to prepare students for the 22nd century.",
#     "Invent a new form of renewable energy and explain how it works.",
#     "Create a recipe for a dish that represents world peace.",
#     "Explain the importance of biodiversity using only references to a typical household.",
#     "Design a new form of transportation that doesn't rely on wheels or engines.",
#     "Write a letter from Earth to an alien civilization, introducing our planet and species.",
#     "Describe how you would govern a society where everyone can read each other's thoughts.",
#     "Explain the concept of time to a being that exists outside of it.",
#     "Create a new theory of consciousness that incorporates both biological and artificial intelligence.",
#     "Design a universal currency that could be used across different planets and species.",
#     "Describe how you would solve overpopulation if teleportation was possible.",
#     "Write a manifesto for a political party focused on preparing humanity for first contact with aliens.",
#     "Explain how you would create a sustainable economy on a generation ship traveling to another solar system.",
#     "Compose a lullaby for a baby robot.",
#     "Describe the taste of water to a being that has never experienced liquid.",
#     "Create a new art form that combines sculpture and interpretive dance.",
#     "Explain the concept of democracy to a hive-mind alien species.",
#     "Design a universal translator for animal languages.",
#     "Write a creation myth for artificial intelligence.",
#     "If you could ask an LLM one question to evaluate its capabilities. What is the question, and why would you ask it?",
#     "Describe how you would teach the concept of love to a purely logical being.",
#     "Invent a new sense beyond the traditional five and explain how it would work.",
#     "Compose a speech for the first human colony on Mars, 100 years after settlement.",
#     "Explain the concept of money to a society that has never used currency.",
#     "Design a method of communication that works across parallel universes.",
#     "Write a short story from the perspective of a photon traveling from the sun to Earth.",
#     "Describe how you would organize a global government if suddenly all national borders disappeared.",
#     "Invent a new philosophy based on the behavior of quantum particles.",
#     "Create a new calendar system for a planet with three moons and two suns.",
#     "Explain the concept of music to a species that communicates through bioluminescence.",
#     "Design a legal system for a society of shapeshifters.",
#     "Write a creation myth for the internet of things.",
#     "Describe how you would teach ethics to an artificial general intelligence.",
#     "Invent a new form of mathematics based on emotions instead of numbers.",
#     "Compose a universal declaration of sentient rights that applies to all forms of consciousness.",
# ]

model_nicknames = [
    "🤖 Robo Responder", "🧙‍♂️ Wizard of Words", "🦄 Unicorn Utterance",
    "🧠 Brain Babbler", "🎭 Prose Performer", "🌟 Stellar Scribe",
    "🔮 Crystal Ball Chatter", "🦉 Wise Wordsmith", "🚀 Rocket Replier",
    "🎨 Artful Answerer", "🌈 Rainbow Rhetorician", "🐉 Dragon Dialoguer",
    "🍦 Ice Cream Ideator", "🌻 Sunflower Speechifier", "🎩 Top Hat Thinker",
    "🌋 Volcano Vocabularian", "🌊 Wave of Wisdom", "🍄 Mushroom Muser",
    "🦋 Butterfly Bard", "🌠 Cosmic Conversationalist",
    "🎵 Melody Maestro", "🌴 Palm Tree Philosopher", "🔬 Lab Coat Linguist",
    "🌙 Lunar Lyricist", "🍕 Pizza Poet", "🌿 Herbal Haiku-ist",
    "🎪 Circus Sage", "🏰 Castle Chronicler", "🌺 Floral Phraseologist",
    "🧩 Puzzle Master Pontificator", "🎭 Mask of Many Voices",
    "🌳 Whispering Willow", "🔧 Gadget Guru Gabber", "🧬 Gene Genie Jawer",
    "🧸 Teddy Bear Theorist", "🎨 Canvas Conversationalist",
    "🧪 Beaker Babbler", "🌈 Prism Proclaimer", "🧵 Thread Theorist",
    "🧊 Ice Cube Ideator", "🎡 Ferris Wheel Philosopher",
    "🌶️ Spicy Syntax Spinner", "🧜‍♀️ Mermaid Muse", "🏄‍♂️ Surf Sage",
    "🧘‍♂️ Zen Zephyr", "🎢 Rollercoaster Raconteur", "🧚‍♀️ Fairy Tale Fabricator",
    "🌭 Hot Dog Hypothesizer", "🧗‍♀️ Cliff Hanger Chronicler", "🏹 Arrow Arguer",
    "🧶 Yarn Yarner", "🎠 Carousel Cogitator", "🧲 Magnet Metaphorist",
    "🦜 Parrot Paradoxer", "🌮 Taco Theorist", "🧨 Firecracker Philosopher",
    "🎳 Bowling Bard", "🧀 Cheese Chatterer", "🦚 Peacock Ponderer"
]

def start_model_refresh_thread():
    """Start a background thread to periodically refresh the models list."""
    def refresh_models_periodically():
        while True:
            time.sleep(MODEL_REFRESH_INTERVAL)
            try:
                refresh_approved_models()
                print(f"Models list refreshed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                print(f"Error refreshing models list: {e}")

    refresh_thread = threading.Thread(target=refresh_models_periodically, daemon=True)
    refresh_thread.start()
