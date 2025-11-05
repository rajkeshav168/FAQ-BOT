# app/bot.py
from __future__ import annotations

import logging
import os
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any, List, Dict, Tuple
import json

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

warnings.filterwarnings("ignore")


class RequirementError(RuntimeError):
    pass


class JupiterFAQBot:
    # ------------------------------------------------------------------ #
    # Free Models Configuration
    # ------------------------------------------------------------------ #
    MODELS = {
        "bi": "sentence-transformers/all-MiniLM-L6-v2",  # Fast semantic search
        "cross": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Reranking
        "qa": "deepset/roberta-base-squad2",  # Better QA model
        "summarizer": "facebook/bart-large-cnn",  # Better summarization
    }

    # Retrieval parameters
    TOP_K = 15          # More candidates for better coverage
    HIGH_SIM = 0.85     # High confidence threshold
    CROSS_OK = 0.50     # Cross-encoder threshold
    MIN_SIM = 0.40      # Minimum similarity to consider
    
    # Paths
    EMB_CACHE = Path("data/faq_embeddings.npy")
    FAQ_PATH = Path("data/faqs.csv")

    # Response templates for better UX
    CONFIDENCE_LEVELS = {
    "high": "This information matches your query based on our FAQs:\n\n",
    "medium": "This appears to be relevant to your question:\n\n",
    "low": "This may be related to your query and could be helpful:\n\n",
    "none": (
        "We couldn't find a direct match for your question. "
        "However, we can assist with topics such as:\n"
        "• Account opening and KYC\n"
        "• Payments and UPI\n"
        "• Rewards and cashback\n"
        "• Credit cards and loans\n"
        "• Investments and savings\n\n"
        "Please try rephrasing your question or selecting a topic above."
    )
    }

    # ------------------------------------------------------------------ #
    def __init__(self, csv_path: str = None) -> None:
        logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
        
        # Use provided path or default
        self.csv_path = csv_path or str(self.FAQ_PATH)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe_dev = 0 if self.device.type == "cuda" else -1

        self._load_data(self.csv_path)
        self._setup_models()
        self._setup_embeddings()
        
        logging.info("Jupiter FAQ Bot ready ✔")

    # ------------------------ Text Processing ------------------------- #
    @staticmethod
    def _clean(text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        text = str(text)
        text = unicodedata.normalize("NFC", text)
        # Remove extra whitespace but keep sentence structure
        text = re.sub(r'\s+', ' ', text)
        # Keep bullet points and formatting
        text = re.sub(r'•\s*', '\n• ', text)
        return text.strip()

    @staticmethod
    def _preprocess_query(query: str) -> str:
        """Preprocess user query for better matching"""
        # Expand common abbreviations
        abbreviations = {
            'kyc': 'know your customer verification',
            'upi': 'unified payments interface',
            'fd': 'fixed deposit',
            'sip': 'systematic investment plan',
            'neft': 'national electronic funds transfer',
            'rtgs': 'real time gross settlement',
            'imps': 'immediate payment service',
            'emi': 'equated monthly installment',
            'apr': 'annual percentage rate',
            'atm': 'automated teller machine',
            'pin': 'personal identification number',
        }
        
        query_lower = query.lower()
        for abbr, full in abbreviations.items():
            if abbr in query_lower.split():
                query_lower = query_lower.replace(abbr, full)
        
        return query_lower

    # ------------------------ Initialization -------------------------- #
    def _load_data(self, path: str):
        """Load and preprocess FAQ data"""
        if not Path(path).exists():
            raise RequirementError(f"CSV not found: {path}")

        df = pd.read_csv(path)
        
        # Clean all text fields
        df["question"] = df["question"].apply(self._clean)
        df["answer"] = df["answer"].apply(self._clean)
        df["category"] = df["category"].fillna("General")
        
        # Create searchable text combining question and category
        df["searchable"] = df["question"].str.lower() + " " + df["category"].str.lower()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)
        
        self.faq = df
        logging.info(f"Loaded {len(self.faq)} FAQ entries from {len(df['category'].unique())} categories")

    def _setup_models(self):
        """Initialize all models"""
        logging.info("Loading models...")
        
        # Sentence transformer for embeddings
        self.bi = SentenceTransformer(self.MODELS["bi"], device=self.device)
        
        # Cross-encoder for reranking
        self.cross = CrossEncoder(self.MODELS["cross"], device=self.device)
        
        # QA model
        self.qa = pipeline(
            "question-answering",
            model=self.MODELS["qa"],
            device=self.pipe_dev,
            handle_impossible_answer=True
        )
        
        # Summarization model - using BART for better quality
        self.summarizer = pipeline(
            "summarization",
            model=self.MODELS["summarizer"],
            device=self.pipe_dev,
            max_length=150,
            min_length=50
        )
        
        logging.info("All models loaded successfully")

    def _setup_embeddings(self):
        """Create or load embeddings"""
        questions = self.faq["searchable"].tolist()
        
        if self.EMB_CACHE.exists():
            emb = np.load(self.EMB_CACHE)
            if len(emb) != len(questions):
                logging.info("Regenerating embeddings due to data change...")
                emb = self.bi.encode(questions, show_progress_bar=True, convert_to_tensor=False)
                np.save(self.EMB_CACHE, emb)
        else:
            logging.info("Creating embeddings for the first time...")
            emb = self.bi.encode(questions, show_progress_bar=True, convert_to_tensor=False)
            self.EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.EMB_CACHE, emb)
        
        self.embeddings = emb

    # ------------------------- Retrieval ------------------------------ #
    def _retrieve_candidates(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve top candidates using semantic search"""
        if top_k is None:
            top_k = self.TOP_K
            
        # Preprocess query
        processed_query = self._preprocess_query(query)
        
        # Encode query
        query_emb = self.bi.encode([processed_query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        
        # Get top indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Filter by minimum similarity
        candidates = []
        for idx in top_indices:
            if similarities[idx] >= self.MIN_SIM:
                candidates.append({
                    "idx": int(idx),
                    "question": self.faq.iloc[idx]["question"],
                    "answer": self.faq.iloc[idx]["answer"],
                    "category": self.faq.iloc[idx]["category"],
                    "similarity": float(similarities[idx])
                })
        
        return candidates

    def _rerank_candidates(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Rerank candidates using cross-encoder"""
        if not candidates:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, c["question"]] for c in candidates]
        
        # Get cross-encoder scores
        scores = self.cross.predict(pairs, convert_to_numpy=True)
        
        # Add scores to candidates
        for c, score in zip(candidates, scores):
            c["cross_score"] = float(score)
        
        # Filter and sort by cross-encoder score
        reranked = [c for c in candidates if c["cross_score"] >= self.CROSS_OK]
        reranked.sort(key=lambda x: x["cross_score"], reverse=True)
        
        return reranked

    def _extract_answer(self, query: str, context: str) -> Dict[str, Any]:
        """Extract specific answer using QA model"""
        try:
            result = self.qa(question=query, context=context)
            return {
                "answer": result["answer"],
                "score": result["score"],
                "start": result.get("start", 0),
                "end": result.get("end", len(result["answer"]))
            }
        except Exception as e:
            logging.warning(f"QA extraction failed: {e}")
            return {"answer": context, "score": 0.5}

    def _create_friendly_response(self, answers: List[str], confidence: str = "medium") -> str:
        """Create a user-friendly response from multiple answers"""
        if not answers:
            return self.CONFIDENCE_LEVELS["none"]
        
        # Remove duplicates while preserving order
        unique_answers = []
        seen = set()
        for ans in answers:
            normalized = ans.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_answers.append(ans)
        
        if len(unique_answers) == 1:
            # Single answer - return as is with confidence prefix
            return self.CONFIDENCE_LEVELS[confidence] + unique_answers[0]
        
        # Multiple answers - need to summarize
        combined_text = " ".join(unique_answers)
        
        # If text is short enough, format it nicely
        if len(combined_text) < 300:
            response = self.CONFIDENCE_LEVELS[confidence]
            for i, answer in enumerate(unique_answers):
                if "•" in answer:
                    # Already has bullets
                    response += answer + "\n\n"
                else:
                    # Add as paragraph
                    response += answer + "\n\n"
            return response.strip()
        
        # Long text - summarize it
        try:
            # Prepare text for summarization
            summary_input = f"Summarize the following information about Jupiter banking services: {combined_text}"
            
            # Generate summary
            summary = self.summarizer(summary_input, max_length=150, min_length=50, do_sample=False)
            summarized_text = summary[0]['summary_text']
            
            # Make it more conversational
            response = self.CONFIDENCE_LEVELS[confidence]
            response += self._make_conversational(summarized_text)
            
            return response
            
        except Exception as e:
            logging.warning(f"Summarization failed: {e}")
            # Fallback to formatted response
            return self._format_multiple_answers(unique_answers, confidence)

    def _make_conversational(self, text: str) -> str:
        """Make response more conversational and friendly"""
        # Add appropriate punctuation if missing
        if text and text[-1] not in '.!?':
            text += '.'
        
        # Replace robotic phrases
        replacements = {
            "The user": "You",
            "the user": "you",
            "It is": "It's",
            "You will": "You'll",
            "You can not": "You can't",
            "Do not": "Don't",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text

    def _format_multiple_answers(self, answers: List[str], confidence: str) -> str:
        """Format multiple answers nicely"""
        response = self.CONFIDENCE_LEVELS[confidence]
        
        if len(answers) <= 3:
            # Few answers - show all
            for answer in answers:
                if "•" in answer:
                    response += answer + "\n\n"
                else:
                    response += f"• {answer}\n\n"
        else:
            # Many answers - group by category
            response += "Here are the key points:\n\n"
            for i, answer in enumerate(answers[:5]):  # Limit to 5
                response += f"{i+1}. {answer}\n\n"
        
        return response.strip()

    # ------------------------- Main API ------------------------------- #
    def generate_response(self, query: str) -> str:
        """Generate response for user query"""
        query = self._clean(query)
        
        # Step 1: Retrieve candidates
        candidates = self._retrieve_candidates(query)
        
        if not candidates:
            return self.CONFIDENCE_LEVELS["none"]
        
        # Step 2: Check for high similarity match
        if candidates[0]["similarity"] >= self.HIGH_SIM:
            return self.CONFIDENCE_LEVELS["high"] + candidates[0]["answer"]
        
        # Step 3: Rerank candidates
        reranked = self._rerank_candidates(query, candidates)
        
        if not reranked:
            # Use original candidates with lower confidence
            reranked = candidates[:3]
            confidence = "low"
        else:
            confidence = "high" if reranked[0]["cross_score"] > 0.8 else "medium"
        
        # Step 4: Extract relevant answers
        relevant_answers = []
        
        for candidate in reranked[:5]:  # Top 5 reranked
            # Try QA extraction for more specific answer
            qa_result = self._extract_answer(query, candidate["answer"])
            
            if qa_result["score"] > 0.3:
                # Good QA match
                relevant_answers.append(qa_result["answer"])
            else:
                # Use full answer if QA didn't find specific part
                relevant_answers.append(candidate["answer"])
        
        # Step 5: Create final response
        final_response = self._create_friendly_response(relevant_answers, confidence)
        
        return final_response

    def suggest_related_queries(self, query: str) -> List[str]:
        """Suggest related queries based on similar questions"""
        candidates = self._retrieve_candidates(query, top_k=10)
        
        related = []
        seen = set()
        
        for candidate in candidates:
            if candidate["similarity"] >= 0.5 and candidate["similarity"] < 0.9:
                # Clean question for display
                clean_q = candidate["question"].strip()
                if clean_q.lower() not in seen and clean_q.lower() != query.lower():
                    seen.add(clean_q.lower())
                    related.append(clean_q)
        
        # Return top 5 related queries
        return related[:5]

    def get_categories(self) -> List[str]:
        """Get all available FAQ categories"""
        return sorted(self.faq["category"].unique().tolist())

    def get_faqs_by_category(self, category: str) -> List[Dict[str, str]]:
        """Get all FAQs for a specific category"""
        cat_faqs = self.faq[self.faq["category"].str.lower() == category.lower()]
        
        return [
            {
                "question": row["question"],
                "answer": row["answer"]
            }
            for _, row in cat_faqs.iterrows()
        ]

    def search_faqs(self, keyword: str) -> List[Dict[str, str]]:
        """Simple keyword search in FAQs"""
        keyword_lower = keyword.lower()
        
        matches = []
        for _, row in self.faq.iterrows():
            if (keyword_lower in row["question"].lower() or 
                keyword_lower in row["answer"].lower()):
                matches.append({
                    "question": row["question"],
                    "answer": row["answer"],
                    "category": row["category"]
                })
        
        return matches[:10]  # Limit to 10 results


# Evaluation module
class BotEvaluator:
    """Evaluate bot performance"""
    
    def __init__(self, bot: JupiterFAQBot):
        self.bot = bot
        
    def create_test_queries(self) -> List[Dict[str, str]]:
        """Create test queries based on FAQ categories"""
        test_queries = [
            # Account queries
            {"query": "How do I open an account?", "expected_category": "Account"},
            {"query": "What is Jupiter savings account?", "expected_category": "Account"},
            
            # Payment queries
            {"query": "How to make UPI payment?", "expected_category": "Payments"},
            {"query": "What is the daily transaction limit?", "expected_category": "Payments"},
            
            # Rewards queries
            {"query": "How do I earn cashback?", "expected_category": "Rewards"},
            {"query": "What are Jewels?", "expected_category": "Rewards"},
            
            # Investment queries
            {"query": "Can I invest in mutual funds?", "expected_category": "Investments"},
            {"query": "What is Magic Spends?", "expected_category": "Magic Spends"},
            
            # Loan queries
            {"query": "How to apply for personal loan?", "expected_category": "Jupiter Loans"},
            {"query": "What is the interest rate?", "expected_category": "Jupiter Loans"},
            
            # Card queries
            {"query": "How to get credit card?", "expected_category": "Edge+ Credit Card"},
            {"query": "Is there any annual fee?", "expected_category": "Edge+ Credit Card"},
        ]
        
        return test_queries
    
    def evaluate_retrieval_accuracy(self) -> Dict[str, float]:
        """Evaluate how well the bot retrieves relevant information"""
        test_queries = self.create_test_queries()
        
        correct = 0
        total = len(test_queries)
        
        results = []
        
        for test in test_queries:
            response = self.bot.generate_response(test["query"])
            
            # Check if response mentions expected category content
            is_correct = test["expected_category"].lower() in response.lower()
            
            if is_correct:
                correct += 1
            
            results.append({
                "query": test["query"],
                "expected_category": test["expected_category"],
                "response": response[:200] + "..." if len(response) > 200 else response,
                "correct": is_correct
            })
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }
    
    def evaluate_response_quality(self) -> Dict[str, Any]:
        """Evaluate the quality of responses"""
        test_queries = [
            "What is Jupiter?",
            "How do I earn rewards?",
            "Tell me about credit cards",
            "Can I get a loan?",
            "How to invest money?"
        ]
        
        quality_metrics = []
        
        for query in test_queries:
            response = self.bot.generate_response(query)
            
            # Check quality indicators
            has_greeting = any(phrase in response for phrase in ["Based on", "Here's", "I found"])
            has_structure = "\n" in response or "•" in response
            appropriate_length = 50 < len(response) < 500
            
            quality_score = sum([has_greeting, has_structure, appropriate_length]) / 3
            
            quality_metrics.append({
                "query": query,
                "response_length": len(response),
                "has_greeting": has_greeting,
                "has_structure": has_structure,
                "appropriate_length": appropriate_length,
                "quality_score": quality_score
            })
        
        avg_quality = sum(m["quality_score"] for m in quality_metrics) / len(quality_metrics)
        
        return {
            "average_quality_score": avg_quality,
            "metrics": quality_metrics
        }


# Utility functions for data preparation
def prepare_faq_data(csv_path: str = "data/faqs.csv") -> pd.DataFrame:
    """Prepare and validate FAQ data"""
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    required_cols = ["question", "answer", "category"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    # Basic stats
    print(f"Total FAQs: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())
    
    return df


# Main execution example
if __name__ == "__main__":
    # Initialize bot
    bot = JupiterFAQBot()
    
    # Test some queries
    test_queries = [
        "How do I open a savings account?",
        "What are the cashback rates?",
        "Can I get a personal loan?",
        "How to use UPI?",
        "Tell me about investments"
    ]
    
    print("\n" + "="*50)
    print("Testing Jupiter FAQ Bot")
    print("="*50 + "\n")
    
    for query in test_queries:
        print(f"Q: {query}")
        response = bot.generate_response(query)
        print(f"A: {response}\n")
        
        # Show related queries
        related = bot.suggest_related_queries(query)
        if related:
            print("Related questions:")
            for r in related[:3]:
                print(f"  - {r}")
        print("\n" + "-"*50 + "\n")
    
    # Run evaluation
    print("\n" + "="*50)
    print("Running Evaluation")
    print("="*50 + "\n")
    
    evaluator = BotEvaluator(bot)
    
    # Retrieval accuracy
    accuracy_results = evaluator.evaluate_retrieval_accuracy()
    print(f"Retrieval Accuracy: {accuracy_results['accuracy']:.2%}")
    print(f"Correct: {accuracy_results['correct']}/{accuracy_results['total']}")
    
    # Response quality
    quality_results = evaluator.evaluate_response_quality()
    print(f"\nAverage Response Quality: {quality_results['average_quality_score']:.2%}")