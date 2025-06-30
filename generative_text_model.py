import logging
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict
import torch

# Configure logging for tracking execution and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextGenerator:
    def __init__(self, model_name: str = "gpt2"):
        # Initialize GPT-2 tokenizer and model
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            logger.info("GPT-2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            sys.exit(1)

    def generate_text(self, prompt: str, max_length: int = 200, temperature: float = 0.7, top_k: int = 50) -> str:
        # Generate text based on the input prompt
        try:
            if not prompt.strip():
                logger.warning("Empty prompt provided.")
                return "Error: Prompt is empty."
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info("Text generated successfully")
            return generated_text
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            return f"Error: {str(e)}"

    def batch_generate(self, prompts: List[Dict]) -> List[Dict]:
        # Process multiple prompts and return generated texts
        results = []
        for prompt_dict in prompts:
            topic = prompt_dict["topic"]
            prompt_text = prompt_dict["prompt"]
            generated = self.generate_text(prompt_text)
            results.append({"topic": topic, "prompt": prompt_text, "generated_text": generated})
        return results

def main():
    # Initialize generator and process example prompts
    generator = TextGenerator()
    prompts = [
        {
            "topic": "Space Exploration",
            "prompt": "In the year 2050, humanity has established a colony on Mars. Describe a day in the life of a Martian colonist."
        },
        {
            "topic": "Artificial Intelligence",
            "prompt": "Artificial intelligence has transformed education by 2030. Explain how AI tutors enhance learning experiences."
        },
        {
            "topic": "Sustainable Cities",
            "prompt": "A futuristic city runs entirely on renewable energy. Describe its infrastructure and daily operations."
        }
    ]
    results = generator.batch_generate(prompts)
    for result in results:
        print(f"\n{'='*60}\nTopic: {result['topic']}\n{'='*60}")
        print(f"Prompt: {result['prompt']}\n")
        print(f"Generated Text:\n{result['generated_text']}\n")

if __name__ == "__main__":
    main()
