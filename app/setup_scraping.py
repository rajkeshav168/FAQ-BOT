# setup_scraping.py
import subprocess
import sys

def setup_selenium_driver():
    """Install Chrome driver for Selenium"""
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        
        # This will download the driver if needed
        service = Service(ChromeDriverManager().install())
        print("Chrome driver installed successfully!")
    except Exception as e:
        print(f"Error setting up Chrome driver: {e}")
        print("You may need to install Chrome/Chromium browser")

def main():
    print("Setting up scraping environment...")
    
    # Install requirements
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'selenium', 'webdriver-manager'])
    
    # Setup driver
    setup_selenium_driver()
    
    print("Setup complete! You can now run the scraper.")

if __name__ == "__main__":
    main()