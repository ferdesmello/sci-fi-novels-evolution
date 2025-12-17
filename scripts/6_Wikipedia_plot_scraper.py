
"""
This script scrapes plot data from novel articles on Wikipedia using the data 
scraped from the Goodreads website.

Modules:
    - pandas
    - wikipedia
    - re
    - requests
    - BeautifulSoup
    - tenacity
    - typing
    - os
    - csv
    - pathlib
    - time
    - winsound (only for Windows)
    - pathlib
    - sys
"""

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
import pandas as pd
import wikipedia
import re
import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type
)
from typing import Dict, Any
import os
import csv
from pathlib import Path
import time
from datetime import datetime
import winsound

from pathlib import Path
import sys

#----------------------------------------------------------------------------------
# Make relative path imports work
# Make project root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
# Import paths
from paths import FILTERED

#----------------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Cleans up the retrieved Wikipedia text by removing references, external links, and other unwanted formatting.

    Args:
        text (str): Text to be cleaned.

    Returns:
        text (str): Cleaned text.
    """

    # Remove reference markers like [1], [2], etc.
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Strip leading and trailing whitespace
    return text.strip()

#----------------------------------------------------------------------------------
def build_unwanted_terms(author: str) -> list[str]:

    """
    Generate unwanted terms for filtering results, including author variations.
    
    Args:
        author (str): Name of the author as given.

    Returns:
        unwanted_terms (list): List of unwanted terms to filter out from search results.
    """
    
    terms = [
        ' series', 
        '(series)', 
        '(novel series)', 
        ' novels', 
        '(novels)', 
        ' trilogy',
        'film)', 
        'movie)', 
        'adaptation)', 
        'franchise)', 
        '(disambiguation)',
        '(comics)', 
        'miniseries)', 
        '(author)',
        '(disambiguation)',
    ]

    unwanted_terms = terms.copy()

    separated_author_name = author.replace(".", ". ").replace(".  ", ". ").lower()
    unwanted_terms.append(separated_author_name)

    separated_author_name = author.replace("Jr.", "").replace(".", ". ").replace(".  ", ". ").lower()
    unwanted_terms.append(separated_author_name)

    first_name = author.split()[0]
    last_name = author.split()[-1]
    unwanted_terms.append((first_name + " " + last_name).lower())

    author_name_without = author.replace("Jr.", "")
    first_name = author_name_without.split()[0]
    last_name = author_name_without.split()[-1]
    unwanted_terms.append((first_name + " " + last_name).lower())

    return unwanted_terms

#----------------------------------------------------------------------------------
def filter_results(search_results: list, unwanted_terms: list) -> list:

    """
    Remove unwanted search results.

    Args:
        search_results (list): List of search results from Wikipedia.
        unwanted_terms (list): List of unwanted terms to filter out.

    Returns:
        filtered (list): Filtered list of search results.
    """

    filtered = []

    for result in search_results:
        result_lower = result.lower()
        has_unwanted = False

        for term in unwanted_terms:
            if term in result_lower:
                has_unwanted = True
                break

        if not has_unwanted:
            filtered.append(result)

    return filtered

#----------------------------------------------------------------------------------
def choose_result(title: str, filtered_results: list) -> str:

    """
    Choose the most appropriate result from the filtered list.

    Args:
        title (str): Title of the novel.
        filtered_results (list): List of filtered search results.

    Returns:
        chosen_result (str): Chosen result from the filtered list.
    """

    if not filtered_results:
        return None

    title_lower = title.lower()
    novel_pattern_1 = re.compile(rf"^{re.escape(title_lower)}.*\(.*novel\)$", re.IGNORECASE)
    novel_pattern_2 = re.compile(rf"^{re.escape(title_lower)}.*\(.*novella\)$", re.IGNORECASE)

    # 1. Prefer results that look like "Title (novel)"
    for result in filtered_results:
        if novel_pattern_1.match(result.lower()):
            return result

    # 2. Prefer results that look like "Title (novella)"
    for result in filtered_results:
        if novel_pattern_2.match(result.lower()):
            return result

    # 3. Exact match
    for result in filtered_results:
        if result.lower() == title_lower:
            return result
    
    # 4. Exact match ignoring colons
    title_no_colon = title_lower.replace(":", "")
    for result in filtered_results:
        if result.lower().replace(":", "") == title_no_colon:
            return result

    # 5. Fallback to first
    return filtered_results[0]

#----------------------------------------------------------------------------------
def validate_result(title: str, chosen_result: str, filtered_results: list) -> str:
    """
    Validate the chosen result by comparing the first two words of the title and the result.
    Returns the best matching result or None if no validation passes.

    Args:
        title (str): Title of the novel.
        chosen_result (str): Chosen result from the filtered list.
        filtered_results (list): List of filtered search results.

    Returns:
        validated_result (str): Validated result or None if validation fails.
    """
    
    if not chosen_result:
        return None

    # Function to get the first two words, padded with "Blank" if needed
    def get_first_two_words(text: str) -> list:
        words = text.split()
        words = [w.lower().replace(":", "") for w in words[:2]]
        while len(words) < 2:
            words.append("Blank")
        return words

    # Function to compare two texts based on first two words
    def is_valid_match(text1: str, text2: str) -> bool:
        words1 = get_first_two_words(text1)
        words2 = get_first_two_words(text2)
        
        #print(f'Comparing "{text1}" vs "{text2}"')
        print(f'  Words1: {words1}, Words2: {words2}')
        
        # First word must always match
        if words1[0] != words2[0]:
            print("  First words don't match - FAIL")
            return False
        
        # If first words match, check second words
        if words1[1] == words2[1]:
            print("  Both first and second words match - PASS")
            return True
        elif (words1[1] == "Blank") or (words2[1] == "Blank"):
            print("  First word matches, one second word is Blank - PASS")
            return True
        else:
            print("  First words match but second words differ - FAIL")
            return False

    # Test the initially chosen result first
    if is_valid_match(title, chosen_result):
        print(f"  Chosen result '{chosen_result}' is valid")
        return chosen_result
    
    print(f"  Chosen result '{chosen_result}' failed validation, trying alternatives...")
    
    # If chosen result fails, try the first two results from filtered list
    for i, result in enumerate(filtered_results[:2]):
        print(f"Testing alternative {i+1}: '{result}'")
        if is_valid_match(title, result):
            print(f"  Alternative result '{result}' is valid")
            return result
    
    print("No valid results found")
    return None

#----------------------------------------------------------------------------------
def extract_plot(page) -> tuple[str, str]:
    """
    Extracts the plot section from a Wikipedia page.

    Args:
        page: Wikipedia page object.

    Returns:
        (header, plot_text):
            header (str): The actual header text of the plot section found.
            plot_text (str): Extracted plot text, truncated to 20,000 characters if necessary.
    """

    # Try realistic headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    session = requests.Session()
    session.headers.update(headers)

    # Add a delay before the HTML request (don't be too greedy!)
    time.sleep(0.5) # Additional delay before HTML fetch

    # Fetch the HTML content
    html_content = session.get(page.url).text

    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    #------------------------------------------
    list_plot_headers = [
        'plot', 
        'the plot', 
        'plot summary', 
        'plot synopsis', 
        'plot introduction',
        'plot outline', 
        'plot and storyline', 
        'synopsis', 
        'summary', 
        'novel plot',
        'setting and plot', 
        'setting and synopsis', 
        'story and significance',
        'story overview', 
        'premise and plot', 
        'narrative', 
        'storyline', 
        'story',
        'outline', 
        'content', 
        'overview', 
        'premise', 
        'introduction', 
        'description',
        'structure', 
        'fictional premise', 
        'characters and story', 
        'species of humans'
    ]
    
    # Find all section headings
    section_headings = soup.find_all(['h2'])
    # Locate the correct section, checking top-priority headers first
    plot_heading = None

    for header in list_plot_headers: # Original order
        for heading in section_headings:
            header_text = header.lower()
            heading_text = heading.get_text(strip=True).lower()
            if header_text == heading_text:
                plot_heading = heading
                break
        if plot_heading: # Exit outer loop once a match is found
            break

    if not plot_heading:
        print("No plot heading found.")
        return None, None

    #------------------------------------------
    # Collect paragraphs in the Plot section
    list_brackets = ['p', 'ol','li']
    plot_paragraphs = []
    for sibling in plot_heading.find_all_next():
        if sibling.name in ['h2']: # Stop at next section
            break
        if sibling.name in list_brackets: # Collect paragraphs or list of items
            plot_paragraphs.append(sibling.get_text().strip())

    #------------------------------------------
    if not plot_paragraphs:
        print("No plot paragraphs found.")
        return None, None

    # Combine paragraphs
    plot_text = "\n\n".join(plot_paragraphs)
    plot_text = clean_text(plot_text)

    #------------------------------------------
    if not plot_text:
        print("No plot text found.")
        return None, None
    
    return header, plot_text[:20000] # Truncate if very long

#----------------------------------------------------------------------------------
LOG_FILE = Path(FILTERED / "wikipedia_failed_novels.csv")

def log_failed_novel(result: dict, title: str, author: str, year: int):

    """
    Append failed cases to a CSV log file.
    
    Args:
        result (dict): Result dictionary from get_novel_summary function.
        title (str): Title of the novel.
        author (str): Name of the author as given.
        year (int): Year of publication.

    Returns:
        None
    """

    if result.get("success", True):
        return # only log failures

    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")

        if not file_exists: # write header once
            writer.writerow(["title", "author", "year", "error"])

        writer.writerow([title, author, year, result.get("error", "")])

#----------------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=20),
    retry=retry_if_exception_type((requests.exceptions.RequestException, 
                                   wikipedia.exceptions.WikipediaException))
)
def get_novel_summary(title: str, author: str, year: int) -> Dict[str, Any]:

    """
    Fetches novel plot/summary text from Wikipedia.

    Args:
        title (str): Title of the novel.
        author (str): Name of the author as given.
        year (int): Year of publication.

    Returns:
        result (dict): Dictionary containing 
            the plot text, 
            page title, 
            page URL, 
            counter of plots found, 
            success status, 
            and error message if any.
    """
    
    # Add a delay at the start of each function call (don't be too greedy!)
    time.sleep(3) # Wait 3 second between requests

    print(f'-- {title} ({year}) {author} --')

    # Before making the Wikipedia search, set the language
    wikipedia.set_lang("en")

    try:
        search_query = f'"{title}" {author} {year}'
        search_results = wikipedia.search(search_query, results=5)
        unwanted_terms = build_unwanted_terms(author)
        filtered_results = filter_results(search_results, unwanted_terms)
        print("Filtered results:", filtered_results)

        chosen_result = choose_result(title, filtered_results)
        print("Chosen result:", chosen_result)

        validated = validate_result(title, chosen_result, filtered_results)
        chosen_result = validated

        if not chosen_result: # Chosen result after validation may be None (not validated)
            return {
                'summary': "No plot available",
                'page_title': None,
                'page_url': None,
                'counter_plot': 0,
                'success': False,
                'error': "No suitable articles found."
            }

        page = wikipedia.page(chosen_result, auto_suggest=False)
        header, plot_text = extract_plot(page)

        print ("Chosen header:", header)
        print ("Plot text:", '"' + plot_text[:60] + "..." + '"' if plot_text else "None")

        if not plot_text:
            return {
                'summary': "No plot available",
                'page_title': page.title,
                'page_url': page.url,
                'counter_plot': 0,
                'success': False,
                'error': "Plot section not found or empty."
            }

        return {
            'summary': plot_text,
            'page_title': page.title,
            'page_url': page.url,
            'counter_plot': 1,
            'success': True
        }

    except Exception as e:
        return {
            'summary': "No plot available",
            'page_title': None,
            'page_url': None,
            'counter_plot': 0,
            'success': False,
            'error': str(e)
        }

#----------------------------------------------------------------------------------
def main():
    """
    Main execution function.
    Reads the CSV data files, calls the functions to scrape Wikipedia, saves the progress, and saves the final CSV files.
    """
    #------------------------------------------
    # Read the CSV files
    input_file_TEST = FILTERED / 'sci-fi_novels_TEST.csv'
    #input_file_TEST = FILTERED / 'sci-fi_novels_TEST_small.csv'

    df_TEST = pd.read_csv(input_file_TEST, sep = ';', encoding="utf-8-sig")
    df_TEST['plot'] = df_TEST['plot'].astype(object)
    df_TEST['url wikipedia'] = df_TEST['url wikipedia'].astype(object)

    #----------------------
    input_file = FILTERED / 'sci-fi_novels_TOP.csv'
    #input_file = input_file_TEST

    df_TOP = pd.read_csv(input_file, sep = ';', encoding="utf-8-sig")
    df_TOP['plot'] = df_TOP['plot'].astype(object)
    df_TOP['url wikipedia'] = df_TOP['url wikipedia'].astype(object)

    #----------------------
    output_file_TEST = FILTERED / 'sci-fi_novels_TEST_Wiki.csv'
    output_file = FILTERED / 'sci-fi_novels_TOP_Wiki.csv'

    #----------------------------------------
    # Load existing progress if the file exists
    if os.path.exists(output_file):
        df_processed = pd.read_csv(output_file, sep=';', encoding='utf-8-sig')
        processed_novels = set(df_processed['url goodreads'])
    else:
        df_processed = pd.DataFrame()
        processed_novels = set()

    #------------------------------------------
    counter_novels = 0
    counter_plots = 0

    # Add a new column for the plot text
    for _, novel in df_TOP.iterrows():

        # Skip already processed novels
        if novel['url goodreads'] in processed_novels:
            continue

        # Extract novel details
        title = novel['title']
        author = novel['author']
        year = int(novel['year'])
        decade = int(novel['decade'])
        rate = float(novel['rate'])
        ratings = int(novel['ratings'])
        series = novel['series']
        genres = novel['genres']
        synopsis = novel['synopsis']
        review = novel['review']
        url_g = novel['url goodreads']

        # Query Wikipedia
        returned_dict = get_novel_summary(title, author, year)

        if not returned_dict["success"]:
            log_failed_novel(returned_dict, title, author, year)

        counter_plot = returned_dict.get('counter_plot')
        returned_title = returned_dict.get('page_title')
        returned_text = returned_dict.get('summary')
        returned_url = returned_dict.get('page_url')
        
        #print(title, author)
        #print("Returned title:", returned_title)
        print("Returned url:", returned_url)
        #print(returned_text)
        print()

        #----------------------------------------
        # One-row dataframe to save the progress in the present novel/row
        df_progress = pd.DataFrame({
            'title': [title],
            'author': [author],
            'year': [year],
            'decade': [decade],
            'rate': [rate],
            'ratings': [ratings],
            'series': [series],
            'genres': [genres],
            'synopsis': [synopsis],
            'review': [review],
            'url goodreads': [url_g],
            'plot': [returned_text],
            'url wikipedia': [returned_url]
        })
        
        # Concatenate the one-row dataframe with the big dataframe with all anterior novels/rows
        df_processed = pd.concat([df_processed, df_progress], ignore_index=True)
        df_processed.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')

        counter_novels += 1
        counter_plots = counter_plots + counter_plot
    
    print(f"Analyzed {counter_novels} novels and added {counter_plots} plots to the file.")

    #----------------------------------------------------------------------------------
    # Add Wikipedia data to the test dataframe

    # Set new index for the dataframes
    df_TEST = df_TEST.set_index("url goodreads")
    df_processed = df_processed.set_index("url goodreads")

    # Update test with the values from processed (only existing columns in test are affected)
    df_TEST.update(df_processed[["plot", "url wikipedia"]])

    # Reset index
    df_TEST = df_TEST.reset_index()
    df_processed = df_processed.reset_index()
    
    #------------------------------------------
    # Order of the columns
    column_order = [
        'title', 
        'author', 
        'year',
        'decade', 
        'rate', 
        'ratings', 
        'series', 
        'genres', 
        'synopsis',
        'review',
        'url goodreads',
        'plot',
        'url wikipedia'
    ]

    # Reorder columns
    df_processed = df_processed.reindex(columns=column_order)
    df_processed = df_processed.sort_values(by=['year', 'author', 'title'], ascending=True)

    df_TEST = df_TEST.reindex(columns=column_order)
    df_TEST = df_TEST.sort_values(by=['year', 'author', 'title'], ascending=True)

    print(df_processed.info())
    print(df_TEST.info())

    #------------------------------------------
    # Save the CSV file
    df_processed.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
    df_TEST.to_csv(output_file_TEST, index=False, sep=';', encoding='utf-8-sig')

    print(f"Data saved to {output_file}")
    print(f"Data saved to {output_file_TEST}")

    #------------------------------------------
    # Final numbers
    total_novels = len(df_processed)
    total_Wiki_url = df_processed['url wikipedia'].apply(lambda x: 1 if pd.notna(x) else 0).sum() 
    total_plots = df_processed['plot'].apply(lambda x: 1 if pd.notna(x) and x != "No plot available" else 0).sum()

    print(f"\nTotal number of novels: {total_novels}")
    print(f"Total number of Wikipedia URLs: {total_Wiki_url}")
    print(f"Total number of plots: {total_plots}")

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