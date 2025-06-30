import logging
from transformers import pipeline
from typing import List, Tuple
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArticleSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the summarizer with a pre-trained model.
        
        Args:
            model_name (str): Name of the pre-trained model to use for summarization.
        """
        try:
            logger.info("Loading summarization model...")
            self.summarizer = pipeline("summarization", model=model_name)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            sys.exit(1)

    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """
        Generate a summary for the given text.
        
        Args:
            text (str): Input text to summarize.
            max_length (int): Maximum length of the summary.
            min_length (int): Minimum length of the summary.
        
        Returns:
            str: Generated summary or error message if summarization fails.
        """
        try:
            if not text.strip():
                logger.warning("Empty input text provided.")
                return "Error: Input text is empty."
            
            logger.info("Generating summary...")
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]['summary_text']
            logger.info("Summary generated successfully.")
            return summary
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            return f"Error: Failed to generate summary - {str(e)}"

def main():
    # Initialize summarizer
    summarizer = ArticleSummarizer()

    # Example articles
    articles: List[Tuple[str, str]] = [
        (
            "Climate Change Impacts",
            """
            Climate change is significantly affecting ecosystems and human societies worldwide. 
            Rising global temperatures, primarily due to greenhouse gas emissions from human 
            activities like burning fossil fuels, are causing more frequent and intense heatwaves, 
            droughts, and wildfires. Polar ice caps are melting at unprecedented rates, leading to 
            rising sea levels that threaten coastal communities and small island nations. Extreme 
            weather events, such as hurricanes and floods, are becoming more severe, causing 
            billions of dollars in damages annually and displacing millions of people. 
            Additionally, shifts in climate patterns are disrupting agriculture, reducing crop 
            yields, and threatening food security in many regions. Biodiversity loss is 
            accelerating as species struggle to adapt to rapidly changing environments. 
            International efforts, such as the Paris Agreement, aim to limit global warming to 
            1.5°C above pre-industrial levels, but current commitments fall short. Transitioning 
            to renewable energy, improving energy efficiency, and adopting sustainable practices 
            are critical steps to mitigate these impacts. Public awareness and policy changes are 
            essential to drive collective action.
            """
        ),
        (
            "Advancements in Artificial Intelligence",
            """
            Artificial intelligence (AI) has seen remarkable advancements in recent years, 
            transforming industries and daily life. Machine learning, a subset of AI, enables 
            computers to learn from data and improve performance without explicit programming. 
            Deep learning, using neural networks, has achieved breakthroughs in image recognition, 
            natural language processing, and autonomous vehicles. AI applications now include 
            personalized recommendations on streaming platforms, medical diagnostics, and fraud 
            detection in finance. However, challenges remain, such as ethical concerns around bias 
            in AI algorithms, data privacy issues, and the potential for job displacement in 
            certain sectors. The development of explainable AI aims to make models more 
            transparent and trustworthy. Governments and organizations are investing heavily in AI 
            research, with global AI market projections reaching hundreds of billions of dollars 
            by the end of the decade. Collaboration between technologists, policymakers, and 
            ethicists is crucial to ensure AI's benefits are maximized while minimizing risks.
            """
        )
    ]

    # Process each article
    for title, text in articles:
        print(f"\n{'='*50}\nArticle: {title}\n{'='*50}")
        print(f"Original Text ({len(text.split())} words):\n{text}\n")
        summary = summarizer.summarize(text)
        print(f"Summary ({len(summary.split())} words):\n{summary}\n")

if __name__ == "__main__":
    main()
