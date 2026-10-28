# scraper package
from .amazon_scraper           import AmazonScraper
from .flipkart_scraper         import FlipkartScraper
from .croma_scraper            import CromaScraper
from .reliance_digital_scraper import RelianceDigitalScraper
from .mock_data_generator      import MockDataGenerator

__all__ = [
    "AmazonScraper",
    "FlipkartScraper",
    "CromaScraper",
    "RelianceDigitalScraper",
    "MockDataGenerator",
]
