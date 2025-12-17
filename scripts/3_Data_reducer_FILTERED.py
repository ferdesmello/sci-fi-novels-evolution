"""
This script reads the data scraped from the Goodreads website and parses 
it (transform, reduce) to be used in the next phase.

Modules:
    - pandas
    - datetime
    - re
    - typing
    - pathlib
    - sys
"""

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
import pandas as pd
import datetime
import re
from typing import Union

from pathlib import Path
import sys

#----------------------------------------------------------------------------------
# Make relative path imports work
# Make project root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
# Import paths
from paths import BRUTE, FILTERED

#----------------------------------------------------------------------------------
def main():
    """
    Big main function with all the reduction, cleaning, and transformation of the data.
    """

    df_shelf = pd.read_csv(BRUTE / 'sci-fi_books_SHELF.csv', sep = ';', encoding='utf-8-sig')
    df_lists = pd.read_csv(BRUTE / 'sci-fi_books_LISTS.csv', sep = ';', encoding='utf-8-sig')

    frames = [df_shelf, df_lists]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(BRUTE / 'sci-fi_books_BRUTE.csv', index=False, sep=';', encoding='utf-8-sig')

    print("\nBRUTE Dataframe")
    print(df.info())

    #----------------------------------------------------------------------------------
    # Filtering out books without both synopses and reviews, books without publishing year, and without many ratings

    # Cleaning null synopsis and review fields
    #-------------------------------------------
    def clean_synopsis(row: Union[str, None]) -> str:
        """
        Replaces NaN values in synopses with a default message.

        Args:
            row (str or None): The synopsis text to analyze.

        Returns:
            (str): Either the original synopsis or a default message if NaN.
        """

        if pd.isna(row):
            return "No synopsis available"
        else:
            return row
    
    #-------------------------------------------
    def clean_review(row: Union[str, None]) -> str:
        """
        Replaces NaN values in reviews with a default message.

        Args:
            row (str or None): The review text to analyze.

        Returns:
            str: Either the original review or a default message if NaN.
        """

        if pd.isna(row):
            return "No review available"
        else:
            return row
    
    #-------------------------------------------
    # Apply the function to update the 'bracket_content' column
    df['synopsis'] = df['synopsis'].apply(clean_synopsis)
    df['review'] = df['review'].apply(clean_review)

    #-------------------------------------------
    # Minimum character length for synopses or reviews
    N_c = 100

    # Filter out rows where the length of synopses or reviews is shorter than N_c characters
    synopsis_mask = df['synopsis'].str.len().fillna(0) >= N_c
    review_mask = df['review'].str.len().fillna(0) >= N_c

    df = df[synopsis_mask | review_mask]

    #-----------------------------------------
    # Drop books without publishing year
    df = df.dropna(axis=0, subset=['year'])

    # Exclude books resulted of "time travel" (just use books with year until the current year)
    today = datetime.date.today()
    year = today.year

    mask_year = df['year'] <= year
    df = df[mask_year]

    #-----------------------------------------
    # Dropping books with fewer than N_r ratings

    # Minimum number of ratings
    N_r = 10

    # Filter out rows where the number of ratings is less than N_r
    ratings_mask = df['ratings'] >= N_r
    df = df[ratings_mask]

    #----------------------------------------------------------------------------------
    # Cleaning URLs
    df['url'] = df['url'].str.strip()

    #----------------------------------------------------------------------------------
    # Excluding series of books (eg. trilogies together in one volume)
    # (must be done before excluding parentheses, as this info is in the parentheses)

    # Regex pattern for exclusion ("#1-3" for books of a series together)
    pattern = r"#1-\d"

    # Exclude rows where the 'title' column matches the pattern
    mask_titles = df['title'].str.contains(pattern, regex=True, na=False)
    df = df[~mask_titles]

    # Regex pattern for exclusion ("#1-3" for books of a series together)
    pattern = r"#\d-\d"

    # Exclude rows where the 'series' column matches the pattern
    mask_series = df['series'].str.contains(pattern, regex=True, na=False)
    df = df[~mask_series]

    #----------------------------------------------------------------------------------
    # Deleting parentheses from titles
    # (must be done after excluding series together in a volume, as some info is in the parentheses)

    # Delete open and closed parentheses
    df.loc[:, "title"] = df["title"].str.replace(r' \(.*\)', '', regex=True)

    # Delete special case of digit error with only open parenthesis
    df.loc[:, "title"] = df["title"].str.replace(r' \(.*', '', regex=True)

    # Including series value in the case of brackes in title (2 cases)
    #-------------------------------------------
    def update_series(row: pd.DataFrame) -> Union[str, None]:
        """
        Adds the text inside of the field reserved to specify a series to a new column of data.

        Args:
            row (pandas.DataFrame): A single row of the DataFrame containing book-related information.

        Returns:
            (str or None): The determined string of data or None if not applicable.
        """

        # Check for existing content
        existing_value = row['series']
        
        # Search for the pattern and capture only the content inside the brackets
        match = re.search(r'\[(.*?)\]', row['title']) # Captures content inside brackets
        
        # Update only if a match is found and existing_value is None
        if match and pd.isna(existing_value):
            return match.group(1) # Return only the content inside the brackets
        else:
            return existing_value # Keep the existing value unchanged
        
    #-------------------------------------------
    # Apply the function to update the 'bracket_content' column
    df['series'] = df.apply(update_series, axis=1)

    # Deletes special case of brackets
    df.loc[:, "title"] = df["title"].str.replace(r' \[.*', '', regex=True)

    #----------------------------------------------------------------------------------
    # Excluding colections of books (eg. many books together in one volume)
    # Some collections have just "/" separating titles, but some titles use "/" right and I want to keep those
    # (must be done after excluding parentheses, as some series have " / " in the parentheses)

    def filter_bar(title: str) -> bool:
        """
        Excludes colections of books.
        Checks if the book title is in a desired format and returns True for that 
        or False if it is in an undesired format.

        Args:
            title (str): A single string, the book title.

        Returns:
            (bool): The determined desirability of the title format.
        """

        # Exceptions for exclusion (some books use "/" properly)
        # Use a set for faster lookups
        exceptions = {
            "11/22/63", 
            "The After/Life",
            "The Mighty Thor, Vol. 3: The Asgard/Shi'ar War",
            "The 7 1/2 Deaths of Evelyn Hardcastle"
        } 

        # Normalize title by stripping extra spaces
        title = title.strip()

        # Check if the title is in exceptions
        if title in exceptions:
            #print(f"Included as exception: {title}")
            return True

        # Check for unwanted slashes
        if re.search("/", title):
            #print(f"Excluded due to slash: {title}")
            return False
        
        # Check for unwanted anti-slashes
        if re.search(r'\\', title):
            #print(f"Excluded due to slash: {title}")
            return False

        # Include titles without slashes
        return True

    #-------------------------------------------
    # Apply the filter to exclude undesired titles
    mask = df['title'].apply(filter_bar)
    df = df[mask]

    #----------------------------------------------------------------------------------
    # Coding the series field
    df['series'] = df['series'].notna().map({True: 'yes', False: 'no'})

    #----------------------------------------------------------------------------------
    # Cleaning the pages field

    # Update the existing column to keep only the number part before the space
    df['pages'] = df['pages'].str.extract(r'(\d+)', expand=False)

    # Optionally, convert extracted values to integers (if all values are numeric)
    df['pages'] = df['pages'].fillna(0).astype(int)

    #----------------------------------------------------------------------------------
    # Fixing the many spaces in some author names

    def clean_whitespace(text: str) -> str:
        """
        Normalizes whitespace in the author names.

        Args:
            text (str): Author name.

        Returns:
            author_clean (str): Processed author name.
        """

        if isinstance(text, str):
            # Split the string into words and join with a single space
            author_clean = ' '.join(text.split())
            return author_clean # Returns the name cleaned
        return text # Or returns the orignal value if not a string

    # Apply the cleaning function to the 'author' column
    df['author'] = df['author'].apply(clean_whitespace)

    #----------------------------------------------------------------------------------
    # Excluding duplicates (some duplicates differ just by capitalization or apostrophe type: ’ or ')

    def normalize_apostrophe(title: str) -> str:
        """
        Normalizes titles by replacing typographic quotes with standard ones.

        Args:
            title (str): Book title.

        Returns:
            normalized_title (str): Processed book title.
        """
                
        # Replace right single quotation mark with a straight apostrophe
        normalized_title = title.replace('’', "'")
        return normalized_title

    # Apply the normalization function to the title column
    df['title'] = df['title'].apply(normalize_apostrophe)

    #-------------------------------------------
    # Deleting spaces using strip
    df['title'] = df['title'].apply(lambda x: x.strip())
    df['author'] = df['author'].apply(lambda x: x.strip())

    #-----------------------
    # Author corrections
    author_corrections = {
        "Anne; Lackey McCaffrey, Mercedes Lackey": "Anne McCaffrey, Mercedes Lackey",
        "Anne; Lackey McCaffrey": "Anne McCaffrey, Mercedes Lackey",
        "reynolds-alastair": "Alastair Reynolds",
        "Grant Naylor": "Rob Grant, Doug Naylor",
        "Pittacus Lore": "James Frey, Jobie Hughes",
        "J.W. Lynne": "Jenny Lynne",
        "Ilona Andrews": "Ilona Gordon, Andrew Gordon",
        "Edson McCann": "Frederik Pohl, Lester del Rey",
    }

    df['author'] = df['author'].replace(author_corrections)

    # Title corrections
    title_corrections = [
        # (wrong_title, correct_title, optional_author)
        ("if", "if [tribe] =: The Bridge Chronicles Book 3", "Gary Ballard"),
        ("残次品", "残次品 [Can Ci Pin]", "Priest"),
        ("Dr. Jekyll and Mr. Hyde", "Strange Case of Dr Jekyll and Mr Hyde", "Robert Louis Stevenson"),
        ("Thrawn", "Star Wars: Thrawn", "Timothy Zahn"),
        ("The Floating Island: The Pearl of the Pacific", "Propeller Island", "Jules Verne"),
        ("The Amphibian", "Amphibian Man", "Alexander Belyaev"),
        ("After Many a Summer Dies the Swan: A Novel", "After Many a Summer", "Aldous Huxley"),
        ("The Futurological Congress: From the Memoirs of Ijon Tichy", "The Futurological Congress", "Stanisław Lem"),
        ("Frankenstein: The 1818 Text", "Frankenstein", "Mary Wollstonecraft Shelley"),
        ("Round the Moon", "Around the Moon", "Jules Verne"),
        ("The Last Man", "Le Dernier Homme", "Jean-Baptiste Cousin de Grainville"),
        ("Tomorrow's Eve", "The Future Eve", "Auguste de Villiers de l'Isle-Adam"),
        ("The Coming Race", "Vril: The Power of the Coming Race", "Edward Bulwer-Lytton"),
        ("The Stars are Ours", "The Stars are Ours!", "Andre Norton"),
    ]

    for wrong, correct, author in title_corrections:
        mask = df["title"] == wrong

        if author:
            mask = mask & (df["author"] == author)

        df.loc[mask, "title"] = correct
    
    # Year correction
    mask_1 = df["title"] == "The Naked Sun"
    mask_2 = df["author"] == "Isaac Asimov"
    mask = mask_1 & mask_2
    df.loc[mask, "year"] = 1957

    #-----------------------
    # Create temporary lowercase columns for comparison
    df['title_lower'] = df['title'].str.lower()
    df['author_lower'] = df['author'].str.lower()

    # Drop duplicates based on the lowercase columns
    df = df.drop_duplicates(subset=['title_lower', 'author_lower'], keep='first')
    df = df.drop_duplicates(subset=['url'], keep='first')

    # I can't drop in function of equal reviews alone because some reviews are empty. I need to use more columns.
    df = df.drop_duplicates(subset=['review', 'author_lower', 'year'], keep='first')

    # Drop the temporary lowercase columns
    df = df.drop(columns=['title_lower', 'author_lower'])

    #----------------------------------------------------------------------------------
    # Deleting some left over duplicates, unwanted non-fiction, and collections.
    # Maybe some of these are not necessary anymore because of the drop of duplicates using the reviews.

    def delete_books(row: pd.DataFrame) -> bool:
        """
        Creates a boolean mask to be used in the dataframe.
        Checks if the book title and author are in the lists of undesirable books and authors.
        If yes, returns False for that.
        If no, returns True.

        Args:
            row (pandas.DataFrame): A single row of the DataFrame containing book-related information.

        Returns:
            (bool): The determined desirability of the book.
        """

        # Titles to be deleted
        titles_to_del = [
            "Feersum Endjinn",
            "Fiction 2000: Cyberpunk and the Future of Narrative",
            "From the Earth to the Moon and 'Round the Moon",
            "The Men From P.I.G. And R.O.B.O.T.",
            "H.G. Wells: Seven Novels",
            "Future Bright, Future Grimm: Transhumanist Tales for Mother Nature's Offspring",
            "Fahrenheit 451; The Illustrated Man; Dandelion Wine; The Golden Apples of the Sun; The Martian Chronicles",
            r"Vorkosigan's Game: The Vor Game \ Borders of Infinity",
            "Divergent Series Ultimate Four-Book Collection: Divergent; Insurgent; Allegiant; Four",
            "The Hitchhiker's Guide to the Galaxy: Tertiary Phase",
            "The Island of Dr. Moreau",
            "R.U.R.: Rossum's Universal Robots",	
            "Eternal Light",
            "Hard to Be a God",
            "Flatland / Sphereland",
            "Shadow Children Complete Set, Books 1-7: Among the Hidden, Among the Impostors, Among the Betrayed, Among the Barons, Among the Brave, Among the Enemy, and Among the Free",
            "Professor Jameson's Interstellar Adventures #1: The Jameson Satellite & Planet of the Double Sun",
            "The Zombie Survival Guide: Complete Protection from the Living Dead",
            "Three Science Fiction Novellas: From Prehistory to the End of Mankind",
            "Artemis Fowl",
            "John Carter of Mars",
            "Have Tech Will Travel: Star Trek S.c.e.",
            "The Reality Dysfunction 1: Emergence",
            "The Star Wars Trilogy",
            "Изобретения профессора Вагнера",
            "The Spacer's Blade and Other Stories",
            "SAMPLER ONLY: Catching Fire",
            "The Girl With All the Gifts: Extended Free Preview",
            "1984",
            "A Handful of Darkness",
            "Yesterday's Son",
            "Гиперболоид инженера Гарина. Аэлита",
            "Пикник на обочине. Отель «У погибшего альпиниста». Улитка на склоне",
        ]
        
        # Authors to be deleted
        authors_to_del = [
            "Iain M. Banks",
            "George Edgar Slusser",
            "Jules Verne",
            "Harry Harrison",
            "H.G. Wells",
            "D.J. MacLennan",
            "Ray Bradbury",
            "Lois McMaster Bujold",
            "Veronica Roth",
            "Douglas Adams",
            "Josef/Karel Capek",
            "Paul J. McAuley",
            "Arkadi Strugatski",
            "Edwin A. Abbott",
            "Margaret Peterson Haddix",
            "Neil R. Jones",
            "Max Brooks",
            "J.-H. Rosny aîné",
            "Eoin Colfer",
            "Edgar Rice Burroughs",
            "Keith R.A. DeCandido",
            "Peter F. Hamilton",
            "George Lucas",
            "Alexander Belyaev",
            "Annie Bellet",
            "Suzanne Collins",
            "M.R. Carey",
            "George Orwell",
            "Philip K. Dick",
            "Crispin",
            "Aleksey Nikolayevich Tolstoy",
            "Arkady Strugatsky",
        ]

        # Extract title and author from the row
        title = row['title']
        author = row['author']

        # Check if both title and author match those in the deletion lists
        if (title in titles_to_del) & (author in authors_to_del):
            #print(f"Excluded: {title} by {author}")
            return False
        else:
            return True

    # Apply the filter function to each row using axis=1 to access row data
    mask = df.apply(delete_books, axis=1)
    df = df[mask]

    #----------------------------------------------------------------------------------
    # Converting the genres' column to actual lists
    for index, row in df.iterrows():
        genres = row['genres']
        # Convert the string representation of lists into actual lists
        if isinstance(genres, str):
            df.at[index, 'genres'] = eval(genres)

    #----------------------------------------------------------------------------------
    # Grouping by decade

    # Create a 'decade' column
    # Floor division by 10, then multiply by 10 to get the start of the decade
    df['decade'] = (df['year'] // 10) * 10 

    # Group by decade and sort within each group by 'ratings' in descending order
    grouped = df.sort_values(by = ['decade', 'ratings'], ascending = [True, False])

    df = df.astype({'year': 'int64', 'decade': 'int64'})

    #----------------------------------------------------------------------------------
    # Filtering out genres
    #-----------------------------------------
    # Excluding books with fantasy as first genre

    # Initialize an empty list to hold indices of rows to keep
    indices_to_keep_1 = []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        genres = row['genres']
        # Check if the genres list is not empty and if the first genre is not 'fantasy'
        if genres and genres[0].lower() == 'fantasy':
            # Skip this row, i.e., don't add it to the indices_to_keep list
            continue
        # Add the index of the row to keep it
        indices_to_keep_1.append(index)

    # Create the filtered DataFrame using the indices of rows to keep
    #df = df.loc[indices_to_keep_1] # Uncomment this line to exclude books with 'fantasy' as first genre

    #-----------------------------------------
    # Define unwanted and required genres
    unwanted_genres = [
        'Graphic Novels',
        'Comics',
        'Graphic Novels Comics',
        'Comic Book',
        'Manga',
        'Short Stories',
        'Anthologies',
        'Nonfiction',
        'Art',
        'Reference',
        'Literary Criticism',
        'Essays',
        'Criticism',
        'High Fantasy',
        'Epic Fantasy',
        'Magic',
        'Angels',
        'Gaming',
        'Role Playing Games',
        'Games',
        'Poetry'
    ]
    unwanted_genres = [genre.lower() for genre in unwanted_genres]

    #required_genres = ['Short Stories', 'Anthologies']
    #required_genres = [genre.lower() for genre in required_genres]

    required_genre = 'science fiction'

    # Initialize an empty list to keep track of indices of rows to keep
    indices_to_keep_2 = []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        genres_line = row['genres']
        
        # Ensure genres are processed as lowercase and stripped of extra spaces
        processed_genres = [genre.lower().strip() for genre in genres_line]

        # Check if the row should be kept:
        # 1. The list of genres must not contain any unwanted genres
        # 2. The list must contain any of the required genres (if applicable)
        # 3. The list must contain the required genre
        has_unwanted_genres = any(genre in unwanted_genres for genre in processed_genres)
        #has_required_genres = any(genre in required_genres for genre in processed_genres)
        has_required_genre = required_genre in processed_genres

        # Add the index if the row does not have any unwanted genres and has the required genre
        #if not has_unwanted_genres and (has_required_genre & has_required_genres):
        if not has_unwanted_genres and has_required_genre:
            indices_to_keep_2.append(index)

    # Create the filtered DataFrame using the indices of rows to keep
    df_filtered = df.loc[indices_to_keep_2]

    #-----------------------------------------
    # Reordering the dataframe
    df_filtered = df_filtered.drop(labels="pages", axis=1)

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
        'url'
    ]

    df_filtered = df_filtered.reindex(columns=column_order)
    df_filtered = df_filtered.sort_values(by=['decade', 'year', 'author', 'title'], axis=0, ascending=True)

    #----------------------------------------------------------------------------------
    # Save the filtered DataFrame back to a CSV

    print("\nFILTERED Dataframe")
    print(df_filtered.info())

    df_filtered.to_csv(FILTERED / 'sci-fi_novels_FILTERED.csv', index=False, sep=';', encoding='utf-8-sig')
    print(f"\nData saved to FILTERED / sci-fi_novels_FILTERED.csv")

#----------------------------------------------------------------------------------
# Execution
if __name__ == "__main__":
    main()