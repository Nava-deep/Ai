import logging
import re
import sys
from transformers import pipeline
from typing import List, Tuple, Dict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from datetime import datetime

# Download required NLTK data for text processing
nltk.download('punkt')
nltk.download('stopwords')

# Configure logging for debugging and tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArticleSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        # Initialize the summarizer with a pre-trained model
        try:
            logger.info("Loading summarization model...")
            self.summarizer = pipeline("summarization", model=model_name)
            self.stop_words = set(stopwords.words('english'))
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            sys.exit(1)

    def preprocess_text(self, text: str) -> str:
        # Clean and preprocess text by removing extra spaces and stop words
        try:
            text = re.sub(r'\s+', ' ', text.strip())
            sentences = sent_tokenize(text)
            processed_sentences = []
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                words = [word for word in words if word.isalnum() and word not in self.stop_words]
                processed_sentences.append(' '.join(words))
            return ' '.join(processed_sentences)
        except Exception as e:
            logger.error(f"Text preprocessing failed: {str(e)}")
            return text

    def summarize(self, text: str, max_length: int = 150, min_length: int = 50, ratio: float = 0.3) -> str:
        # Generate a summary for the input text with specified length constraints
        try:
            if not text.strip():
                logger.warning("Empty input text provided.")
                return "Error: Input text is empty."
            preprocessed_text = self.preprocess_text(text)
            max_len = min(max_length, int(len(word_tokenize(preprocessed_text)) * ratio))
            logger.info("Generating summary...")
            summary = self.summarizer(
                preprocessed_text,
                max_length=max_len,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )[0]['summary_text']
            logger.info("Summary generated successfully.")
            return summary
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            return f"Error: Failed to generate summary - {str(e)}"

    def batch_summarize(self, articles: List[Tuple[str, str]], max_length: int = 150, min_length: int = 50) -> List[Dict]:
        # Process multiple articles and return summaries with metadata
        results = []
        for title, text in articles:
            summary = self.summarize(text, max_length, min_length)
            word_count_original = len(word_tokenize(text))
            word_count_summary = len(word_tokenize(summary))
            results.append({
                "title": title,
                "original_text": text,
                "summary": summary,
                "original_word_count": word_count_original,
                "summary_word_count": word_count_summary,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        return results

    def export_to_csv(self, results: List[Dict], filename: str = "summaries.csv"):
        # Export summarization results to a CSV file
        try:
            df = pd.DataFrame(results)
            df.to_csv(filename, index=False)
            logger.info(f"Summaries exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")

    def format_output(self, result: Dict, format_type: str = "plain") -> str:
        # Format the output in plain text or markdown
        if format_type == "markdown":
            output = f"### {result['title']}\n"
            output += f"**Original Text ({result['original_word_count']} words)**:\n{result['original_text']}\n\n"
            output += f"**Summary ({result['summary_word_count']} words)**:\n{result['summary']}\n\n"
            output += f"**Generated on**: {result['timestamp']}\n"
        else:
            output = f"\n{'='*50}\nArticle: {result['title']}\n{'='*50}\n"
            output += f"Original Text ({result['original_word_count']} words):\n{result['original_text']}\n\n"
            output += f"Summary ({result['summary_word_count']} words):\n{result['summary']}\n"
            output += f"Generated on: {result['timestamp']}\n"
        return output

def main():
    # Initialize summarizer and process example articles
    summarizer = ArticleSummarizer()
    articles = [
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
        ),
        (
            "Quantum Computing Breakthroughs",
            """
            Quantum computing represents a revolutionary leap in computational capabilities. Unlike 
            classical computers, which process bits as 0s or 1s, quantum computers use quantum bits 
            or qubits that can exist in superpositions, enabling parallel computations on an 
            unprecedented scale. Recent breakthroughs include achieving quantum supremacy, where a 
            quantum computer solved a problem infeasible for classical computers in a reasonable 
            time frame. Companies like IBM, Google, and startups like Rigetti are advancing quantum 
            hardware and software, targeting applications in cryptography, drug discovery, and 
            optimization problems. Challenges include maintaining qubit stability and scaling systems 
            to handle practical tasks. Quantum error correction and fault-tolerant designs are active 
            research areas. The potential to solve complex problems faster than classical computers 
            could transform industries, but significant technical hurdles remain before widespread 
            adoption.
            """
        )
    ]
    results = summarizer.batch_summarize(articles, max_length=200, min_length=60)
    for result in results:
        print(summarizer.format_output(result, format_type="plain"))
        print(summarizer.format_output(result, format_type="markdown"))
    summarizer.export_to_csv(results)

if __name__ == "__main__":
    main()
