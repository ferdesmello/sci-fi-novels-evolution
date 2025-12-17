"""
This script scrapes data from book pages on the Goodreads website using a list of 
bookshelf URLs as a starting point.

Modules:
    - requests
    - BeautifulSoup
    - pandas
    - time
    - json
    - os
    - urllib3
    - random
    - logging
    - tenacity
    - re
    - typing
    - datetime
    - winsound (only for Windows)
    - pathlib
    - sys
"""

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import os
from requests.exceptions import RequestException, Timeout
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import random
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import re
from typing import List, Dict, Any, Union
from requests import Session
from datetime import datetime
import winsound

from pathlib import Path
import sys

#----------------------------------------------------------------------------------
# Make relative path imports work
# Make project root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
# Import paths
from paths import BRUTE

#----------------------------------------------------------------------------------
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#----------------------------------------------------------------------------------
def load_progress() -> Dict[str, Dict[str, Any]]:
    """
    Loads scraping progress data from a JSON file.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing progress data.
        If the file exists, returns its contents;
        If the file doesn't exist, returns a default dictionary with an empty 'urls' key.
        
    Example:
        Returns: {'urls': {}} or {'urls': {'http://example.com': {...}}}
    """

    if os.path.exists(BRUTE / 'scraping_progress.json'):
        with open(BRUTE / 'scraping_progress.json', 'r') as f:
            return json.load(f)
    return {'urls': {}}

#----------------------------------------------------------------------------------
def save_progress(progress: Dict[str, Dict[str, Any]]):
    """
    Saves scraping progress data to a JSON file.

    Args:
        progress (Dict[str, Dict[str, Any]]): The progress dictionary to be saved.
        Typically contains information about scraped URLs and their status.

    Note:
        Overwrites the existing file at BRUTE / 'scraping_progress.json'
    """

    with open(BRUTE / 'scraping_progress.json', 'w') as f:
        json.dump(progress, f)

#----------------------------------------------------------------------------------
def get_session() -> Session:
    """
    Creates a configured requests Session with retry mechanisms and custom headers.

    Returns:
        requests.Session: A session object configured with:
        - Retry mechanism for specific HTTP status codes
        - 10 total retries with an exponential backoff
        - Custom User-Agent header
        - HTTPS connection adapter with retry support
    """

    session = requests.Session()
    retries = Retry(total=10, backoff_factor=0.2, status_forcelist=[413, 429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

#----------------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def make_request(session: Session, url: str, timeout: int) -> requests.Response:
    """
    Makes an HTTP GET request with error handling and logging.

    Args:
        session (requests.Session): Configured requests session to use for the request.
        url (str): The URL to send the GET request to.
        timeout (int): Maximum time in seconds to wait for the request to complete.

    Returns:
        requests.Response: The response object from the successful request.
    """

    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return response
    except Timeout:
        logging.warning(f"Timeout occurred while requesting {url}. Retrying...")
        raise
    except RequestException as e:
        logging.error(f"Error occurred while requesting {url}: {e}")
        raise

#----------------------------------------------------------------------------------
def scrape_list(session: Session, url: str, page: int) -> List[Dict[str, Any]]:
    """
    Scrapes data from a Goodreads list page.

    Args:
        session (requests.Session): Configured requests session to use for the request.
        url (str): The URL for the list on Goodreads.
        page (int): Number of the present page at the list.

    Returns:
        books (List[Dict[str, Any]]): list of dictionaries of book data containing:
        - 'title': book title
        - 'author': book author
        - 'url': book page address on Goodreads
    """

    full_url = f"{url}?page={page}"
    response = make_request(session=session, url=full_url, timeout=30)
    soup = BeautifulSoup(response.content, 'html.parser')
    books = []

    #-----------------------------------------
    # Approach 1: Extract using <div> elements with class "elementList"
    for book in soup.find_all('div', class_='elementList'):
        title_elem = book.find('a', class_='bookTitle')
        author_elem = book.find('a', class_='authorName')

        if title_elem and author_elem:
            books.append({
                'title': title_elem.text.strip(),
                'author': author_elem.text.strip(),
                'url': "https://www.goodreads.com" + title_elem['href']
            })

    #-----------------------------------------
    # Approach 2: Extract using <tr> elements with itemtype="http://schema.org/Book"
    for book in soup.find_all('tr', itemtype='http://schema.org/Book'):
        title_elem = book.find('a', class_='bookTitle')
        author_elem = book.find('a', class_='authorName')

        if title_elem and author_elem:
            books.append({
                'title': title_elem.find('span', itemprop='name').text.strip(),
                'author': author_elem.find('span', itemprop='name').text.strip(),
                'url': "https://www.goodreads.com" + title_elem['href']
            })

    return books

#----------------------------------------------------------------------------------
def scrape_book_page(session: Session, url: str) -> Union[Dict[str, Any], None]:
    """
    Scrapes data from a Goodreads book page.

    Args:
        session (requests.Session): Configured requests session to use for the request.
        url (str): address for the book page on Goodreads.

    Returns:
        book_data (Dict[str, float, int]): A dictionary containing:
        - 'series': simple data whether the book is part of a series
        - 'pages': number of pages of that edition
        - 'year': first year published
        - 'rate': average rate
        - 'ratings': number of ratings
        - 'genres': listed genres
        - 'synopsis': synopses text
        - 'review': longer review of the first three
        None: Error scraping the data.
    """

    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        #-----------------------------------------
        # Extract series
        series_tag = soup.find('h3', class_='Text Text__title3 Text__italic Text__regular Text__subdued')
        series = series_tag.text.strip() if series_tag else None

        #-----------------------------------------
        # Extract pages
        pages_tag = soup.find('p', attrs={'data-testid': 'pagesFormat'})
        pages = pages_tag.text.strip() if pages_tag else None

        #-----------------------------------------
        # Extract year
        year = None
        publication_info = soup.find('p', {'data-testid': 'publicationInfo'})
        if publication_info:
            year_text = publication_info.get_text()
            match = re.search(r'(\d{4})', year_text)
            if match:
                year = match.group(1)

        #-----------------------------------------
        # Extract genres
        genres = []
        genre_elem = soup.find('div', {'data-testid': 'genresList'})
        if genre_elem:
            genres = [a.text.strip() for a in genre_elem.find_all('a')]

        #-----------------------------------------
        # Extract synopsis
        synopsis = "No synopsis available"
        description_elem = soup.find('div', {'data-testid': 'description'})
        if description_elem:
            synopsis = description_elem.text.strip()

        #-----------------------------------------
        # Extract mean rate
        rate = 0
        rating_div = soup.find('div', class_='RatingStatistics__rating')
        if rating_div:
            rate_text = rating_div.text.strip()
            rate = float(rate_text)

        #-----------------------------------------
        # Extract number of ratings
        ratings = 0
        ratings_elem = soup.find('span', {'data-testid': 'ratingsCount'})
        if ratings_elem:
            ratings_text = ratings_elem.text.strip().split()[0].replace(',', '')
            ratings = int(ratings_text)

        #-----------------------------------------
        # Find all review sections
        review_sections = soup.find_all('section', class_='ReviewText')

        """len(review_sections) > n is used to check if the list has at least n+1 
        elements before accessing the nth index. This ensures we don't try to access 
        an index that doesn't exist."""
        # Extract the first review, if it exists
        review_1 = review_sections[0].find('span', class_='Formatted').text.strip() if len(review_sections) > 0 else "No review available"
        # Extract the second review, if it exists
        review_2 = review_sections[1].find('span', class_='Formatted').text.strip() if len(review_sections) > 1 else "No review available"
        # Extract the third review, if it exists
        review_3 = review_sections[2].find('span', class_='Formatted').text.strip() if len(review_sections) > 2 else "No review available"
        # Get the longest review
        review = max([review_1, review_2, review_3], key=len)

        # Display the extracted reviews
        #print(f"\nFirst review: {review_1}")
        #print(f"\nSecond review: {review_2}")
        #print(f"\nThird review: {review_3}")
        #print(f"\nChosen review: {review}")

        #-----------------------------------------
        # Book data extracted
        book_data = {
            'series': series,
            'pages': pages,
            'year': year,
            'rate': rate,
            'ratings': ratings,
            'genres': genres,
            'synopsis': synopsis,
            'review': review,
        }

        logging.info(f"Successfully scraped book:\n  {url}")
        logging.debug(f"Book data: {book_data}")

        return book_data

    except Exception as e:
        logging.error(f"Error scraping:\n!!!{url}: {e}")
        return None

#----------------------------------------------------------------------------------
def scrape_goodreads_lists(urls: List[str], max_pages: int) -> List[Dict[str, Any]]:
    """
    Main scraping function that calls the other scraping functions.

    Args:
        urls (List[str]): List of URL Goodreads lists to use.
        max_pages (int): Maximun number os pages to use per Goodreads list.

    Returns:
        all_books (List[Dict[str, Any]]): List of dictionaries with book data.
    """

    progress = load_progress()
    session = get_session()
    all_books = []
    
    for url in urls:
        books = []
        last_page = progress['urls'].get(url, {}).get('last_page', 0)
        
        for page in range(last_page + 1, max_pages + 1):
            logging.info(f"Scraping:\n{url} - page {page}")
            try:
                page_books = scrape_list(session, url, page)
                
                if not page_books:
                    logging.info(f"No more books found on:\n{url} - page {page}\nMoving to next URL.----------------")
                    break
                
                for book in page_books:
                    book_data = scrape_book_page(session, book['url'])
                    if book_data:
                        book_data.update(book)
                        books.append(book_data)
                    else:
                        logging.warning(f"Failed to scrape book:\n!!!{book['url']}")
                    
                # Implement a random delay between 5 to 15 seconds after every page (not at every book)
                time.sleep(random.uniform(5, 15))
                
                # Save progress after each page
                progress['urls'][url] = {'last_page': page, 'books': books}
                save_progress(progress)
                
            except Exception as e:
                logging.error(f"Error scraping:\n!!!{url} - page {page}: {e}")
                time.sleep(60) # Wait a minute before retrying
        
        all_books.extend(books)
    #print(books)
    return all_books

#----------------------------------------------------------------------------------
def main():
    """
    Main execution function for the scraping script.
    Calls the main scraping function, transforms the JSON file in a CSV, and saves the data.
    """

    # Webpages to start the scraping
    urls = [
        "https://www.goodreads.com/list/show/43374.Classic_Science_Fiction_1930_1939",
        "https://www.goodreads.com/list/show/40744.Classic_Science_Fiction_1940_1949",
        "https://www.goodreads.com/list/show/5152.Classic_Science_Fiction_1950_1959",
        "https://www.goodreads.com/list/show/5158.Classic_Science_Fiction_1960_1969",
        "https://www.goodreads.com/list/show/42069.Classic_Science_Fiction_1970_1979",
        "https://www.goodreads.com/list/show/42417.Classic_Science_Fiction_1980_1989",
        "https://www.goodreads.com/list/show/42875.Classic_Science_Fiction_1990_1999",
        "https://www.goodreads.com/list/show/43319.Classic_Science_Fiction_2000_2009",
        "https://www.goodreads.com/list/show/75182.Science_Fiction_2010_2019",
        "https://www.goodreads.com/list/show/146613.Science_Fiction_2020_2029",
        "https://www.goodreads.com/list/show/155483",
        "https://www.goodreads.com/list/show/155401",
        "https://www.goodreads.com/list/show/155391",
        "https://www.goodreads.com/list/show/155378",
        "https://www.goodreads.com/list/show/155343",
        "https://www.goodreads.com/list/show/155307",
        "https://www.goodreads.com/list/show/155296",
        "https://www.goodreads.com/list/show/155261",
        "https://www.goodreads.com/list/show/155402",
        "https://www.goodreads.com/list/show/155440",
        "https://www.goodreads.com/list/show/155447",
        "https://www.goodreads.com/list/show/155799",
        "https://www.goodreads.com/list/show/171985",
        "https://www.goodreads.com/list/show/196123",
        "https://www.goodreads.com/list/show/196124",
        "https://www.goodreads.com/list/show/195299.Scifi_published_in_2024",
        "https://www.goodreads.com/list/show/79670.Best_Science_Fiction_on_Goodreads_with_fewer_than_100_ratings",
        "https://www.goodreads.com/list/show/78128.Best_Science_Fiction_on_Goodreads_with_between_100_and_999_ratings",
        "https://www.goodreads.com/list/show/77875.Best_Science_Fiction_on_Goodreads_with_between_1000_and_9999_ratings",
        "https://www.goodreads.com/list/show/46769.Popular_Science_Fiction_on_GoodReads_with_between_10000_and_24999_ratings",
        "https://www.goodreads.com/list/show/39287.Popular_Science_Fiction_on_GoodReads_with_between_25000_and_50000_ratings",
        "https://www.goodreads.com/list/show/138257.Popular_Science_Fiction_on_Goodreads_with_between_50000_and_99999_ratings",
        "https://www.goodreads.com/list/show/35776.Most_Popular_Science_Fiction_on_Goodreads",
        "https://www.goodreads.com/list/show/115331.Nineteenth_Century_Science_Fiction",
        "https://www.goodreads.com/list/show/18864.Genetics_in_Science_Fiction",
        "https://www.goodreads.com/list/show/549.Most_Under_rated_Science_Fiction",
        "https://www.goodreads.com/list/show/6228.SF_Masterworks",
        "https://www.goodreads.com/list/show/6934.Science_Fiction_Books_by_Female_Authors",
        "https://www.goodreads.com/list/show/9951.best_hard_science_fiction",
        "https://www.goodreads.com/list/show/17148.Space_Horror",
        "https://www.goodreads.com/list/show/6032.Best_Aliens",
        "https://www.goodreads.com/list/show/485.Best_Books_on_Artificial_Intelligence_",
        "https://www.goodreads.com/list/show/487.Best_of_Cyberpunk",
        "https://www.goodreads.com/list/show/17324.Transhuman_Science_Fiction_",
        "https://www.goodreads.com/list/show/114349.Best_Forgotten_Science_Fiction_of_the_20th_Century",
        "https://www.goodreads.com/list/show/47.Best_Dystopian_and_Post_Apocalyptic_Fiction",
        "https://www.goodreads.com/list/show/7239.Best_Utopian_Dystopian_Fiction",
        "https://www.goodreads.com/list/show/25823",
        "https://www.goodreads.com/list/show/154763.The_Amazing_Colossal_Science_Fiction_Ketchup_Pre_1900s",
        "https://www.goodreads.com/list/show/101755.Radium_Age_Sci_Fi",
        "https://www.goodreads.com/list/show/113093.Golden_Age_and_New_Wave_Science_Fiction_novels",
        "https://www.goodreads.com/list/show/83753.Eclipse_Phase_Recommended_Reading",
        "https://www.goodreads.com/list/show/1127.Excellent_Space_Opera",
        "https://www.goodreads.com/list/show/47731.Best_HARD_SCIENCE_FICTION_of_the_21st_Century",
        "https://www.goodreads.com/list/show/78971.Best_Space_Opera_of_the_21st_Century",
        ]

    all_books = scrape_goodreads_lists(urls, max_pages=50)

    # Create DataFrame with specified column order
    df = pd.DataFrame(all_books)

    column_order = [
        'title', 
        'author', 
        'year', 
        'pages', 
        'rate', 
        'ratings', 
        'series', 
        'genres', 
        'synopsis',
        'review',
        'url'
    ]
    df = df.reindex(columns=column_order)
    
    df.to_csv(BRUTE / 'sci-fi_books_PARTIAL_LISTS.csv', index=False, sep=';', encoding='utf-8-sig')
    
    logging.info(f"Scraped {len(all_books)} books. Data saved to BRUTE / sci-fi_books_PARTIAL_LISTS.csv")

    #----------------------------------------------------------------------------------
    # Reading the complete json file and saving it as a CSV file

    # Step 1: Load the JSON file into a Python object
    with open(BRUTE / 'scraping_progress.json', 'r') as f:
        data = json.load(f) # Load JSON into a dictionary

    # Step 2: Extract the "books" data from within the "urls" layer
    books_data = []

    # Navigate through the URLs dictionary and extract the "books" lists
    for _, url_content in data['urls'].items(): # url_key, url_content
        # Check if "books" is present and is a list
        if 'books' in url_content and isinstance(url_content['books'], list):
            books_data.extend(url_content['books']) # Add the list of books to books_data

    # Step 3: Flatten the books data into a DataFrame
    # Each element in books_data is a book's details
    df_books = pd.json_normalize(books_data)
    
    # Inspect the DataFrame structure
    print("\n",df_books.head())
    print(df_books.info())
    
    #--------------------------------------------
    # Chose the right columns and their order
    df_books = df_books.reindex(columns=column_order)

    #--------------------------------------------
    # Save the flattened and final DataFrame to a CSV file
    df_books.to_csv(BRUTE / 'sci-fi_books_LISTS.csv', index=False, sep=';', encoding='utf-8-sig')

    logging.info(f"Data saved to BRUTE / sci-fi_books_LISTS.csv")

#----------------------------------------------------------------------------------
# Execution
if __name__ == "__main__":

    # Record start time
    start = datetime.now()

    main()

    # Record end time
    end = datetime.now()

    # How long did it take?
    print(f"Script started at {start}")
    print(f"Script finished at {end}")
    print(f"Total runtime: {end - start}")

    winsound.Beep(800, 500) # Play a 800 Hz beep for 500 milliseconds
    winsound.Beep(500, 500) # Play a 500 Hz beep for 500 milliseconds
    winsound.Beep(300, 500) # Play a 300 Hz beep for 500 milliseconds