# app/scraper.py (Enhanced version)
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import json
from typing import List, Dict, Optional
from pathlib import Path
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JupiterFAQScraper:
    """Enhanced scraper for Jupiter Money website"""
    
    def __init__(self):
        self.base_url = "https://jupiter.money"
        self.target_urls = [
            "https://jupiter.money/savings-account/",
            "https://jupiter.money/pro-salary-account/",
            # ... other URLs
        ]
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def test_scraping(self) -> Dict[str, bool]:
        """Test if scraping actually works"""
        results = {}
        
        # Test 1: Can we access the website?
        try:
            response = self.session.get(self.base_url, timeout=10)
            results['website_accessible'] = response.status_code == 200
            logger.info(f"Website accessible: {results['website_accessible']}")
        except:
            results['website_accessible'] = False
            logger.error("Cannot access Jupiter website")
        
        # Test 2: Can we find FAQ content?
        if results['website_accessible']:
            try:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Look for common FAQ indicators
                faq_indicators = ['faq', 'help', 'support', 'question', 'answer']
                content = soup.get_text().lower()
                results['faq_content_found'] = any(indicator in content for indicator in faq_indicators)
                logger.info(f"FAQ content found: {results['faq_content_found']}")
            except:
                results['faq_content_found'] = False
        
        return results
    
    def scrape_with_fallback(self) -> pd.DataFrame:
        """Try multiple scraping methods with fallback"""
        all_faqs = []
        
        # Method 1: Try actual scraping
        logger.info("Attempting to scrape Jupiter website...")
        test_results = self.test_scraping()
        
        if test_results.get('website_accessible'):
            # Try basic scraping
            for url in self.target_urls[:3]:  # Test with first 3 URLs
                try:
                    faqs = self.scrape_page_safe(url)
                    all_faqs.extend(faqs)
                    if faqs:
                        logger.info(f"Successfully scraped {len(faqs)} FAQs from {url}")
                except Exception as e:
                    logger.warning(f"Failed to scrape {url}: {e}")
        
        # Method 2: If scraping fails or gets too little data, use fallback
        if len(all_faqs) < 10:
            logger.warning("Actual scraping yielded insufficient data. Using fallback FAQ data...")
            all_faqs = self.get_fallback_faqs()
        
        # Create DataFrame
        df = pd.DataFrame(all_faqs)
        if not df.empty:
            df = df.drop_duplicates(subset=['question'])
        
        return df
    
    def scrape_page_safe(self, url: str) -> List[Dict]:
        """Safely scrape a page with error handling"""
        faqs = []
        
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Got status code {response.status_code} for {url}")
                return faqs
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Strategy 1: Look for structured data
            scripts = soup.find_all('script', type='application/ld+json')
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if '@type' in data and 'FAQ' in str(data.get('@type')):
                        # Extract FAQ structured data
                        faqs.extend(self.extract_structured_faqs(data))
                except:
                    continue
            
            # Strategy 2: Look for FAQ sections
            faq_sections = soup.find_all(['div', 'section'], 
                                       class_=re.compile(r'faq|question|help|support', re.I))
            
            for section in faq_sections[:5]:  # Limit to prevent too many
                faqs.extend(self.extract_section_faqs(section, url))
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
        
        return faqs
    
    def extract_structured_faqs(self, data: dict) -> List[Dict]:
        """Extract FAQs from structured data"""
        faqs = []
        
        if isinstance(data, dict):
            if data.get('@type') == 'FAQPage':
                for item in data.get('mainEntity', []):
                    if item.get('@type') == 'Question':
                        faqs.append({
                            'question': self._clean_text(item.get('name', '')),
                            'answer': self._clean_text(
                                item.get('acceptedAnswer', {}).get('text', '')
                            ),
                            'category': 'General'
                        })
        
        return faqs
    
    def extract_section_faqs(self, section, url: str) -> List[Dict]:
        """Extract FAQs from a page section"""
        faqs = []
        category = self._get_category_from_url(url)
        
        # Look for Q&A pairs
        questions = section.find_all(['h2', 'h3', 'h4', 'dt', 'div'], 
                                   class_=re.compile(r'question|title|header', re.I))
        
        for q in questions[:10]:  # Limit to prevent too many
            # Try to find corresponding answer
            answer = None
            
            # Check next sibling
            next_elem = q.find_next_sibling()
            if next_elem and next_elem.name in ['p', 'div', 'dd']:
                answer = next_elem
            
            # Check parent's next sibling
            if not answer:
                parent = q.parent
                if parent:
                    next_elem = parent.find_next_sibling()
                    if next_elem:
                        answer = next_elem.find(['p', 'div'])
            
            if answer:
                faqs.append({
                    'question': self._clean_text(q.get_text()),
                    'answer': self._clean_text(answer.get_text()),
                    'category': category
                })
        
        return faqs
    
    def get_fallback_faqs(self) -> List[Dict]:
        """Return comprehensive fallback FAQs based on Jupiter's services"""
        # This is the fallback data that will be used if scraping fails
        # Based on the FAQs you provided earlier
        return [
            # Account
            {
                'question': 'What is the Jupiter All-in-1 Savings Account?',
                'answer': 'The All-in-1 Savings Account on Jupiter powered by Federal Bank helps you manage your money better with faster payments, smart saving tools, and investment insights—all in one place.',
                'category': 'Account'
            },
            {
                'question': 'How do I open a Jupiter Savings Account?',
                'answer': 'You can open your Jupiter digital account by following a few simple steps: 1. Install the Jupiter app 2. Tap "Open an all-in-1 Savings Account" while selecting a Jupiter experience 3. Complete your video KYC',
                'category': 'Account'
            },
            # Add more FAQs here from your data...
            # (Include all the FAQs you provided)
        ]
    
    def _clean_text(self, text: str) -> str:
        """Clean text"""
        if not text:
            return ""
        text = ' '.join(text.split())
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()
    
    def _get_category_from_url(self, url: str) -> str:
        """Get category from URL"""
        url_lower = url.lower()
        if 'account' in url_lower:
            return 'Account'
        elif 'payment' in url_lower or 'upi' in url_lower:
            return 'Payments'
        elif 'card' in url_lower:
            return 'Cards'
        elif 'loan' in url_lower:
            return 'Loans'
        elif 'invest' in url_lower or 'mutual' in url_lower:
            return 'Investments'
        return 'General'
    
    def run_complete_scraping(self) -> pd.DataFrame:
        """Main method to run scraping with all fallbacks"""
        logger.info("Starting Jupiter FAQ scraping process...")
        
        # Try scraping with fallback
        df = self.scrape_with_fallback()
        
        if df.empty:
            logger.error("No FAQ data could be obtained!")
        else:
            logger.info(f"Total FAQs collected: {len(df)}")
            
            # Save to CSV
            self.save_to_csv(df)
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = "data/faqs.csv"):
        """Save FAQs to CSV"""
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        if not df.empty:
            df = df[['question', 'answer', 'category']]
            df.to_csv(filename, index=False)
            logger.info(f"Saved {len(df)} FAQs to {filename}")


class FAQUpdater:
    """Manages FAQ updates with reliability checks"""
    
    def __init__(self):
        self.scraper = JupiterFAQScraper()
        self.faq_file = "data/faqs.csv"
    
    def check_and_update(self, force_update: bool = False) -> pd.DataFrame:
        """Check and update FAQs with verification"""
        import os
        from datetime import datetime, timedelta
        
        # First, check if we have existing FAQ data
        if os.path.exists(self.faq_file) and not force_update:
            # Load existing data
            existing_df = pd.read_csv(self.faq_file)
            
            # Check file age
            file_time = datetime.fromtimestamp(os.path.getmtime(self.faq_file))
            if datetime.now() - file_time < timedelta(days=7):
                logger.info(f"FAQ data is recent (updated {file_time.strftime('%Y-%m-%d')})")
                return existing_df
        
        # Try to update
        logger.info("Attempting to update FAQ data...")
        
        # Test if scraping works
        test_results = self.scraper.test_scraping()
        
        if not test_results.get('website_accessible'):
            logger.warning("Cannot access Jupiter website. Using existing/fallback data.")
            if os.path.exists(self.faq_file):
                return pd.read_csv(self.faq_file)
            else:
                # Use fallback data
                fallback_faqs = self.scraper.get_fallback_faqs()
                df = pd.DataFrame(fallback_faqs)
                self.scraper.save_to_csv(df)
                return df
        
        # Try scraping
        new_df = self.scraper.run_complete_scraping()
        
        # Verify the scraped data
        if self.verify_scraped_data(new_df):
            # Backup old file if exists
            if os.path.exists(self.faq_file):
                backup_name = f"data/faqs_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                os.rename(self.faq_file, backup_name)
                logger.info(f"Backed up old FAQs to {backup_name}")
            
            return new_df
        else:
            logger.warning("Scraped data failed verification. Using existing/fallback data.")
            if os.path.exists(self.faq_file):
                return pd.read_csv(self.faq_file)
            else:
                # Use comprehensive fallback
                return self.create_comprehensive_fallback()
    
    def verify_scraped_data(self, df: pd.DataFrame) -> bool:
        """Verify if scraped data is valid"""
        if df.empty:
            return False
        
        # Check minimum requirements
        if len(df) < 20:  # Expecting at least 20 FAQs
            logger.warning(f"Only {len(df)} FAQs scraped, seems too low")
            return False
        
        # Check if we have multiple categories
        if df['category'].nunique() < 3:
            logger.warning("Not enough FAQ categories")
            return False
        
        # Check answer quality
        avg_answer_length = df['answer'].str.len().mean()
        if avg_answer_length < 50:
            logger.warning("Answers seem too short")
            return False
        
        return True
    
    def create_comprehensive_fallback(self) -> pd.DataFrame:
        """Create comprehensive fallback FAQ data"""
        # This includes ALL the FAQs you provided
        fallback_data = [
            # Account FAQs
            {"question": "What is the Jupiter All-in-1 Savings Account?", "answer": "The All-in-1 Savings Account on Jupiter powered by Federal Bank helps you manage your money better with faster payments, smart saving tools, and investment insights—all in one place.", "category": "Account"},
            {"question": "How do I open a Jupiter Savings Account?", "answer": "You can open your Jupiter digital account by following a few simple steps: 1. Install the Jupiter app 2. Tap 'Open an all-in-1 Savings Account' while selecting a Jupiter experience 3. Complete your video KYC", "category": "Account"},
            {"question": "Do I earn Jewels for making payments?", "answer": "Yes! You earn up to 1% cashback as Jewels on: • UPI payments • Debit Card spends (online & offline) • Investments in Digital Gold", "category": "Rewards"},
            {"question": "Can I use my Jupiter Debit Card outside India?", "answer": "Absolutely. You can spend in over 120 countries with 0% forex fee on international transactions using your Jupiter Debit Card.", "category": "Card"},
            {"question": "Do I earn Jewels on International payments?", "answer": "Yes, you also earn up to 1% cashback on online and offline international spends.", "category": "Rewards"},
            {"question": "What payment modes are available with the Jupiter account?", "answer": "You can make superfast payments with UPI, IMPS, and debit card—whether it's for recharges, bills, or merchant transactions.", "category": "Payments"},
            {"question": "Can I invest using my Jupiter account?", "answer": "Yes! You can invest in Mutual Funds and Digital Gold with up to 1.5% extra returns on curated mutual fund plans.", "category": "Investments"},
            {"question": "What additional benefits do I get with the Savings Account?", "answer": "You earn up to 1% cashback as Jewels on: • Free cheque book • Free IMPS transfers • ATM withdrawals", "category": "Account"},
            # Include ALL other FAQs from your data here...
        ]
        
        df = pd.DataFrame(fallback_data)
        self.scraper.save_to_csv(df)
        return df
    
    def get_scraping_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics about FAQ data"""
        return {
            'total_faqs': len(df),
            'categories': df['category'].nunique(),
            'category_distribution': df['category'].value_counts().to_dict(),
            'avg_question_length': df['question'].str.len().mean(),
            'avg_answer_length': df['answer'].str.len().mean(),
            'data_source': 'scraped' if len(df) > 50 else 'fallback'
        }


# Create a simple test script
def test_scraper():
    """Test if the scraper can actually get data"""
    print("Testing Jupiter FAQ Scraper...")
    print("-" * 50)
    
    scraper = JupiterFAQScraper()
    
    # Test 1: Website accessibility
    test_results = scraper.test_scraping()
    print(f"Website accessible: {test_results.get('website_accessible', False)}")
    print(f"FAQ content found: {test_results.get('faq_content_found', False)}")
    
    # Test 2: Try scraping one page
    print("\nTesting page scraping...")
    test_url = "https://jupiter.money/savings-account/"
    faqs = scraper.scrape_page_safe(test_url)
    print(f"FAQs found on {test_url}: {len(faqs)}")
    
    if faqs:
        print("\nSample FAQ:")
        print(f"Q: {faqs[0]['question'][:100]}...")
        print(f"A: {faqs[0]['answer'][:100]}...")
    
    # Test 3: Full scraping
    print("\nRunning full scraping process...")
    df = scraper.run_complete_scraping()
    print(f"Total FAQs collected: {len(df)}")
    
    if not df.empty:
        print(f"Categories: {df['category'].unique()}")
        print(f"Data saved to: data/faqs.csv")
    
    return df


if __name__ == "__main__":
    # Run the test
    test_scraper()