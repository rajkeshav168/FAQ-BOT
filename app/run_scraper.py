# run_scraper.py
import sys
import logging
from app.scraper import FAQUpdater

def main():
    """Run the FAQ scraping process"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if force update is requested
    force_update = '--force' in sys.argv
    
    # Run updater
    updater = FAQUpdater()
    df = updater.check_and_update(force_update=force_update)
    
    # Display stats
    stats = updater.get_scraping_stats(df)
    print(f"\nScraping Statistics:")
    print(f"Total FAQs: {stats['total_faqs']}")
    print(f"Categories: {stats['categories']}")
    print(f"Category Distribution: {stats['category_distribution']}")

if __name__ == "__main__":
    main()