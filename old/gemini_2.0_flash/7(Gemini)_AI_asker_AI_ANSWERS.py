"""
This script uses Gemini 2.0, via the Google API, to answer questions about the plot 
of the books scraped before, parses the answers and saves it.

Modules:
    - os
    - dotenv
    - google
    - pandas
    - tenacity
    - requests
    - logging
    - typing
"""

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, wait_fixed
from requests.exceptions import RequestException
import logging
from typing import List

#----------------------------------------------------------------------------------
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from the .env file
load_dotenv(dotenv_path='../../KEYs/My_GOOGLE_API_Key.env')

# Get the API key from the environment variable
api_key = os.environ.get("GOOGLE_API_KEY")

# Configure the genai library with the loaded API key
if api_key:
    genai.configure(api_key=api_key)
    print("Google Gemini API key loaded successfully.")
else:
    print("Error: GOOGLE_API_KEY not found in the environment variables. Make sure your .env file is correctly configured.")

# Define your default generation configuration
default_generation_config = {
    'max_output_tokens': 700,  # Set max tokens
    'temperature': 0.2         # Set temperature
}

# Initialize the Gemini model with the default configuration
#model = genai.GenerativeModel(model_name='gemini-2.5-pro', generation_config=default_generation_config)
model = genai.GenerativeModel(model_name='gemini-2.0-flash', generation_config=default_generation_config)

#----------------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type((RequestException, Exception)), # Retry on API errors or network issues
    wait=wait_exponential(multiplier=1, min=4, max=60), # Exponential backoff: starts at 4 seconds, max 60 seconds
    stop=stop_after_attempt(10) # Stop after 10 attempts
)
def analyze_book(title: str , author: str, year: int, synopsis: str, review: str, plot: str, genres: List[str]) -> str: 
    """
    Prompts Gemini 2.0 to analyse the data of each book and answers questions about it.

    Args:
        title (str): Book title.
        author (str): Book author.
        year (int): Book publishing year.
        synopsis (str): Goodreads synopsis for the book.
        review (str): Chosen Goodreads review for the book.
        plot (str): Wikipedia plot for the book.
        genres (List[str]): Goodreads list of genres.

    Returns:
        answer (str): Gemini 2.0 entire processed answer.
    """

    prompt = f"""
    You are a helpful assistant and scholar of comparative sci-fi literature who analyzes book plots based on your own knowledge and provided information.
    You received the task to carefully consider the plot of the book "{title}" by {author}, published in {year}, focusing on key elements that will help answer the following questions. 

    **Output Formatting Instructions**:
    Follow this exact format!

    Provide a concise paragraph summarizing the most iconic and well-known elements of the book, including:
    - Themes central to the story or that significantly impact the plot;
    - Memorable locations or time settings unique to the story;
    - The main character, if there is one, and secondary characters;
    - Important creatures (e.g., aliens, robots, AIs) central to the story;
    - Notable technologies that play a key role in the plot.

    After the summary, leave one blank line and provide one answer per line in the following format:
    Question number, followed by a period, then the selected answer from the alternatives given, followed by a colon, and then the detailed but short justification in a single sentence.
    Example:
    1. Hard: The story emphasizes hard sciences like physics and biology.
    2. Balanced: The story balances entertainment with deeper themes and ideas.
    3. Near future: The setting is within a few decades ahead of the publication date.

    Important Notes:
    Ensure there are no line breaks, extra spaces, or symbols between answers.
    If no clear summary is possible, provide a brief explanation before answering.

    **Questions**:
    1. Is the book considered more soft or hard sci-fi?
        Very soft: scientific accuracy is minimal or irrelevant, leaning towards fantasy or speculation;
        Soft: emphasis on social sciences such as psychology, sociology, or philosophy, with less focus on scientific plausibility;
        Mixed: balances elements of both soft and hard sci-fi equally;
        Hard: emphasis on natural or applied sciences like physics, astronomy, biology, or technology, with some focus on scientific plausibility;
        Very hard: scientific accuracy in many sciences is important and integrated into the narrative, leaning towards extrapolation from known science;
        Uncertain: not enough information to say.
    2. Is the book considered more of a light or heavy reading experience?
        Very light: easily accessible, fast read, minimal intellectual demands, focus on entertainment, humor, or adventure;
        Light: somewhat accessible, with some thoughtful elements and themes, but still focused on entertainment;
        Balanced: mix of light and heavy elements, moderately complex and deep;
        Heavy: intellectually or emotionally demanding, with complex ideas and deeper themes;
        Very heavy: challenging, slow read, dense in language, themes, or ideas, focus on philosophical or intricate scientific concepts;
        Uncertain: not enough information to say.
    3. When does most of the story take place in relation to the year the book was published?
        Distant past: millennia or more before; 
        Far past: centuries before; 
        Near past: within a few decades before; 
        Present: within a few years; 
        Near future: within a few decades ahead; 
        Far future: centuries ahead; 
        Distant future: millennia or more ahead; 
        Multiple timelines; 
        Uncertain: not enough information to say.
    4. What is the mood of the story?
        Very optimistic: overwhelmingly positive, uplifting, hopeful; 
        Optimistic: positive outlook but with moments of pessimism; 
        Balanced: mix of positive and negative moods without leaning towards one; 
        Pessimistic: negative outlook but with moments of optimism; 
        Very pessimistic: overwhelmingly negative, bleak, hopeless;
        Uncertain: not enough information to say.
    5. What is the social-political scenario depicted in the story?
        Utopic: ideal or perfect society;
        Leaning utopic: significant prosperity and desirable elements but with some flaws;
        Balanced: mix of both strengths and flaws elements, or an ordinary view of society;
        Leaning dystopic: significant problems and undesirable elements but with some strengths;
        Dystopic: bleak, deeply flawed, authoritarian, and oppressive;
        Uncertain: unclear, not a major focus of the story, or there is not enough information to say.
    6. Is most of the story set on Earth?
        Yes;
        No;
        Uncertain: not enough information to say.
    7. Is the story set in a post-apocalyptic world (after a civilization-collapsing event)?
        Yes: clear post-apocalyptic state;
        Somewhat: just some elements are present, or the collapse is partial or local;
        No: it's not set in a post-apocalyptic world;
        Uncertain: not enough information to say.
    8. Are there any depictions or mentions of non-terrestrial life forms (e.g., aliens, extraterrestrial organisms, creatures not originating from Earth, even if non-sentient) or alien technology in the story?
        Yes;
        No;
        Uncertain: not enough information to say.
    9. How are the non-terrestrial life forms generally depicted in the story?
        Not applicable: no extraterrestrial life forms present (answered No to the prior question);
        Good: friendly, virtuous, helpful, or heroic; 
        Leaning good: generally positive or benign but with flaws or minor conflicts; 
        Ambivalent: morally ambiguous, showing both positive and negative traits, or multifaceted; 
        Leaning bad: generally antagonistic or threatening but not entirely villainous; 
        Bad: hostile, villainous, antagonistic, or threatening; 
        Uncertain: not enough information to say, lack of (moral) characterization, neutral, or minimal plot relevance.
    10. Are there any depictions of robots or non-biological complex artificial intelligences in the story?
        Yes;
        No;
        Uncertain: not enough information to say.
    11. How are the robots or artificial intelligences generally depicted in the story?
        Not applicable: no robots or artificial intelligences present (answered No to the prior question);
        Good: friendly, virtuous, helpful, or heroic; 
        Leaning good: generally positive or benign but with flaws or minor conflicts; 
        Ambivalent: morally ambiguous, showing both positive and negative traits, or multifaceted; 
        Leaning bad: generally antagonistic or threatening but not entirely villainous; 
        Bad: hostile, villainous, antagonistic, or threatening; 
        Uncertain: not enough information to say, lack of (moral) characterization, neutral, or minimal plot relevance.
    12. Is there a single protagonist or main character?
        Yes; 
        No: multiple protagonists or main characters;
        Uncertain: not enough information to say.
    13. What is the gender of the single protagonist or main character?
        Male; 
        Female; 
        Other: the gender is ambiguous, fluid, or neither male nor female;
        Non-human: the central character is non-human (e.g., animal, AI, robot, alien, etc);
        Uncertain: not enough information to say;
        Not applicable: no clear single protagonist or main character (e.g., more than one main character, humanity, etc), answered No to the prior question.
    14. Are there any depictions of virtual reality or immersive digital environments (e.g., simulations, augmented reality) in the story?
        Yes: central or significant role in the plot;
        Somewhat: some form of it, minor or background role;
        No: not present;
        Uncertain: not enough information to say.
    15. How are technology and science depicted in the story?
        Good: optimistic and beneficial portrayal; 
        Leaning good: mostly positive but with some issues; 
        Ambivalent: balanced view with both positive and negative consequences; 
        Leaning bad: largely negative portrayal but with redeeming features; 
        Bad: pessimistic, harmful or destructive portrayal; 
        Uncertain: not enough information to say, lack of (moral) characterization, neutral, or minimal plot relevance.
    16. How central is the critique or reflection of specific social issues (e.g., inequality, war, discrimination, political oppression) to the story?
        Core: main driver of the plot or key theme;
        Major: significant role but secondary theme;
        Minor: subtle role or minimal theme;
        Absent: not present;
        Uncertain: not enough information to say.
    17. How central is an ecological or environmental message to the story?
        Core: main driver of the plot or key theme;
        Major: significant role but secondary theme;
        Minor: subtle role or minimal theme;
        Absent: not present;
        Uncertain: not enough information to say.

    When answering, consider how the following synopsis, review, plot summary, and genres may provide relevant context for each question.

    Book synopsis: {synopsis}
    Partial review: {review}
    Plot summary: {plot}
    Genres the book fits in: {genres}
    """
    
    response = model.generate_content(prompt)
    # Check if the response and text attribute exist
    if response and hasattr(response, 'text'):
        answer = response.text
    else:
        answer = None

    print(f'\n"{title}" by {author}, {year}')
    #print(prompt)
    print(answer)

    return answer

#----------------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def ask_to_AI(df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """
    Sends book data to the prompt function, parses the returned answers, and merges everything together.

    Args:
        df (pandas.DataFrame): Dataframe with all books information.
        output_file (str): Output file name to save the progress.

    Returns:
        df_processed (pandas.DataFrame): Dataframe with all original books information and 
            processed answers about them from GPT-4o.
    """
        
    # Lists to store the complete answer the AI give and its parts for each book

    # Complete AI answer
    complete_answer = []

    # Summarizing paragraph
    paragraph = []

    # Answers to the questions
    soft_hard = []
    light_heavy = []
    time = []
    mood = []
    social_political = []
    on_earth = []
    post_apocalyptic = []
    aliens = []
    aliens_are = []
    robots_ai = []
    robots_ai_are = []
    protagonist = []
    protagonist_is = []
    virtual = []
    tech_science = []
    social = []
    enviromental = []
    
    # Justifications to the answers given
    soft_hard_just = []
    light_heavy_just = []
    time_just = []
    mood_just = []
    social_political_just = []
    on_earth_just = []
    post_apocalyptic_just = []
    aliens_just = []
    aliens_are_just = []
    robots_ai_just = []
    robots_ai_are_just = []
    protagonist_just = []
    protagonist_is_just = []
    virtual_just = []
    tech_science_just = []
    social_just = []
    enviromental_just = []

    #----------------------------------------
    # Load existing progress if the file exists
    if os.path.exists(output_file):
        df_processed = pd.read_csv(output_file, sep=';', encoding='utf-8-sig')
        processed_books = set(df_processed['url goodreads'])
    else:
        df_processed = pd.DataFrame()
        processed_books = set()
    for _, book in df.iterrows():
        # Skip already processed books
        if book['url goodreads'] in processed_books:
            continue

        # Extract book details
        title = book['title']
        author = book['author']
        year = int(book['year'])
        decade = int(book['decade'])
        #pages = book['pages']
        rate = float(book['rate'])
        ratings = int(book['ratings'])
        series = book['series']
        genres = book['genres']
        synopsis = book['synopsis']
        review = book['review']
        url_g = book['url goodreads']
        plot = book['plot']
        url_w = book['url wikipedia']

        forbidden_titles = [
            "Songmaster",
            "The Hollow Lands",
            "A Clockwork Orange",
            "Under the Skin"
        ]

        if pd.isna(plot) or title in forbidden_titles:
            plot = 'No plot available.'
        #print(f"\nPlot: {plot}")

        if len(plot) > len(synopsis) and title not in forbidden_titles:
            synopsis = 'No synopsis available.'
            review = 'No review available.'
        #print(plot)
        #print(synopsis)
        #print(review)
        #----------------------------------------
        try:
            # Get the AI's answers for the book
            AI_answers = analyze_book(title, author, year, synopsis, review, plot, genres)
            #print("\n",AI_answers)

            # Split answers into a list of, hopefully, 19 items
            answers = []
            justifications = []
            lines = AI_answers.split('\n')
            if(len(lines) == 20):
                lines.pop(19) # Remove the last line that is empty
            print(f"Number of lines in the answer: {len(lines)}")

            # Just to garantee that it will produce a list with at least two items and not rase and index error
            if (len(lines) < 2) | (len(lines) > 19):
                lines = ["", ""]

            # Process the first line differently
            paragraph_text = lines[0]

            # Sometimes the AI output may or may not have a break line or even a paragraph, or the text may be all in one line
            # You can go along with just the paragraph and the answers with no break line between them
            # Problematic books/rows will be nulled an excluded later in the program
            # But you will need to rerun the program to try again to get (only) those books/rows
            #----------------------------------------
            # If there is a paragraph and NO break line separating it from the answers
            if (len(lines) == 18) & (len(lines[1].strip()) != 0):
                # Process each line
                for line in lines[1:]: # Start from the second line (first is paragraph)
                    # Split at the first occurrence of ". " to separate the number
                    parts_number_text = line.split('. ', 1)
                    
                    # Further split at the first occurrence of ": " to separate answer and justification
                    parts_answer_just = parts_number_text[1].split(': ', 1)
                    
                    # Check if the split was successful, i.e., there are exactly two parts
                    if len(parts_answer_just) == 2:
                        answers.append(parts_answer_just[0].strip()) # Append the word after the number and period
                        justifications.append(parts_answer_just[1].strip()) # Append the text after the colon

            #----------------------------------------
            # If there is a paragraph and a break line separating it from the answers
            elif (len(lines) == 19) & (len(lines[1].strip()) == 0):
                # Process each line
                for line in lines[2:]: # Start from the third line (first is paragraph and second is empty)
                    # Split at the first occurrence of ". " to separate the number
                    parts_number_text = line.split('. ', 1)
                    
                    # Further split at the first occurrence of ": " to separate answer and justification
                    parts_answer_just = parts_number_text[1].split(': ', 1)
                    
                    # Check if the split was successful, i.e., there are exactly two parts
                    if len(parts_answer_just) == 2:
                        answers.append(parts_answer_just[0].strip()) # Append the word after the number and period
                        justifications.append(parts_answer_just[1].strip()) # Append the text after the colon

            #----------------------------------------
            # Append answers to respective lists
            complete_answer.append(AI_answers)

            if (len(answers) == 17) & (len(justifications) == 17):
                
                paragraph.append(paragraph_text)

                soft_hard.append(answers[0])
                soft_hard_just.append(justifications[0])

                light_heavy.append(answers[1])
                light_heavy_just.append(justifications[1])

                time.append(answers[2])
                time_just.append(justifications[2])

                mood.append(answers[3])
                mood_just.append(justifications[3])

                social_political.append(answers[4])
                social_political_just.append(justifications[4])

                on_earth.append(answers[5])
                on_earth_just.append(justifications[5])

                post_apocalyptic.append(answers[6])
                post_apocalyptic_just.append(justifications[6])

                aliens.append(answers[7])
                aliens_just.append(justifications[7])

                aliens_are.append(answers[8])
                aliens_are_just.append(justifications[8])

                robots_ai.append(answers[9])
                robots_ai_just.append(justifications[9])

                robots_ai_are.append(answers[10])
                robots_ai_are_just.append(justifications[10])

                protagonist.append(answers[11])
                protagonist_just.append(justifications[11])

                protagonist_is.append(answers[12])
                protagonist_is_just.append(justifications[12])

                virtual.append(answers[13])
                virtual_just.append(justifications[13])

                tech_science.append(answers[14])
                tech_science_just.append(justifications[14])

                social.append(answers[15])
                social_just.append(justifications[15])

                enviromental.append(answers[16])
                enviromental_just.append(justifications[16])
            else:
                logging.warning(f"Unexpected number of answers for book: {title}\nFound {len(answers)}/17 answers and {len(justifications)}/17 justifications.")

                # Something went wrong and better fill a row of None in each list
                paragraph.append(paragraph_text)

                soft_hard.append(None)
                soft_hard_just.append(None)

                light_heavy.append(None)
                light_heavy_just.append(None)

                time.append(None)
                time_just.append(None)

                mood.append(None)
                mood_just.append(None)

                social_political.append(None)
                social_political_just.append(None)

                on_earth.append(None)
                on_earth_just.append(None)

                post_apocalyptic.append(None)
                post_apocalyptic_just.append(None)

                aliens.append(None)
                aliens_just.append(None)

                aliens_are.append(None)
                aliens_are_just.append(None)

                robots_ai.append(None)
                robots_ai_just.append(None)

                robots_ai_are.append(None)
                robots_ai_are_just.append(None)

                protagonist.append(None)
                protagonist_just.append(None)

                protagonist_is.append(None)
                protagonist_is_just.append(None)

                virtual.append(None)
                virtual_just.append(None)

                tech_science.append(None)
                tech_science_just.append(None)

                social.append(None)
                social_just.append(None)

                enviromental.append(None)
                enviromental_just.append(None)

            #----------------------------------------
            # One-row dataframe to save the progress in the present book/row
            df_progress = pd.DataFrame({
                'title': [title],
                'author': [author],
                'year': [year],

                'paragraph': [paragraph[-1]],

                '1 soft hard': [soft_hard[-1]],
                'justifying soft hard': [soft_hard_just[-1]],

                '2 light heavy': [light_heavy[-1]],
                'justifying light heavy': [light_heavy_just[-1]],

                '3 time': [time[-1]],
                'justifying time': [time_just[-1]],

                '4 mood': [mood[-1]],
                'justifying mood': [mood_just[-1]],

                '5 social political': [social_political[-1]],
                'justifying social political': [social_political_just[-1]],

                '6 on Earth': [on_earth[-1]],
                'justifying on Earth': [on_earth_just[-1]],

                '7 post apocalyptic': [post_apocalyptic[-1]],
                'justifying post apocalyptic': [post_apocalyptic_just[-1]],

                '8 aliens': [aliens[-1]],
                'justifying aliens': [aliens_just[-1]],

                '9 aliens are': [aliens_are[-1]],
                'justifying aliens are': [aliens_are_just[-1]],

                '10 robots and AI': [robots_ai[-1]],
                'justifying robots and AI': [robots_ai_just[-1]],

                '11 robots and AI are': [robots_ai_are[-1]],
                'justifying robots and AI are': [robots_ai_are_just[-1]],

                '12 protagonist': [protagonist[-1]],
                'justifying protagonist': [protagonist_just[-1]],

                '13 protagonist is': [protagonist_is[-1]],
                'justifying protagonist is': [protagonist_is_just[-1]],

                '14 virtual': [virtual[-1]],
                'justifying virtual': [virtual_just[-1]],

                '15 tech and science': [tech_science[-1]],
                'justifying tech and science': [tech_science_just[-1]],

                '16 social issues': [social[-1]],
                'justifying social issues': [social_just[-1]],

                '17 enviromental': [enviromental[-1]],
                'justifying enviromental': [enviromental_just[-1]],

                'complete answer': [complete_answer[-1]],

                'decade': [decade],
                #'pages': [pages],
                'rate': [rate],
                'ratings': [ratings],
                'series': [series],
                'genres': [genres],
                'synopsis': [synopsis],
                'review': [review],
                'url goodreads': [url_g],
                'plot': [plot],
                'url wikipedia': [url_w]
            })

            # Concatenate the one-row dataframe with the big dataframe with all anterior books/rows
            df_processed = pd.concat([df_processed, df_progress], ignore_index=True)
            df_processed.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
            logging.info(f"Progress saved for book: {title}")

        except Exception as e:
            logging.error(f"Failed to analyze book {title}. Error: {e}")
            raise  # Re-raise the exception to trigger a retry

    return df_processed

#----------------------------------------------------------------------------------
def main():
    """
    Main execution function for the script.
    Calls the AI asker function, orders the data, and saves it in a CSV file.
    """
        
    #------------------------------------------
    # Name of the input file
    #input_file = '../Data/Filtered/sci-fi_books_TEST_Wiki.csv'
    #input_file = '../Data/Filtered/sci-fi_books_TEST_Wiki_short.csv'
    input_file = '../Data/Filtered/sci-fi_books_TOP_Wiki.csv'

    # Name of the output file
    #output_file = 'sci-fi_books_AI_ANSWERS_TEST_Gemini_16.csv'
    #output_file = 'sci-fi_books_AI_ANSWERS_TEST_Gemini.csv'
    #output_file = 'sci-fi_books_AI_ANSWERS_TEST_short_Gemini.csv'
    output_file = 'sci-fi_books_AI_ANSWERS_Gemini.csv'

    #------------------------------------------
    # Load book data to send to the AI
    df = pd.read_csv(input_file, sep=';', encoding="utf-8-sig")

    # Ask the AI about ALL the books
    df_processed = ask_to_AI(df, output_file)

    # Retyping columns of the processed dataframe
    df_processed['year'] = df_processed['year'].astype(int)
    df_processed['decade'] = df_processed['decade'].astype(int)
    df_processed['rate'] = df_processed['rate'].astype(float)
    df_processed['ratings'] = df_processed['ratings'].astype(int)

    #------------------------------------------
    # Sometimes the AI output is not formatted right
    # This will exclude wrong rows (a null paragraph or null on some other column just to be sure)
    # You will need to rerun the program at least once to get all the books/rows but it will keep the progress until then
    df_processed = df_processed.dropna(axis=0, subset=['paragraph', 'justifying on Earth', '12 protagonist', 'justifying enviromental'], how = 'any', ignore_index=True)
    df_processed = df_processed.sort_values(by = ['decade', 'year', 'author', 'title'], ascending=True)

    #------------------------------------------
    print('\n',df_processed.info())
    #print(df_processed.head())

    size_in = df.shape[0] # Number of rows
    size_out = df_processed.shape[0] # Number of rows
    missing_books = size_in - size_out # Difference in number of rows

    print(f"Book(s) missing: {missing_books}. If that number is higher than 0, rerun this program until it is 0 AND there are no more WARNINGS.")

    #------------------------------------------
    df_processed.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
    print(f"Data saved to {output_file}")

#----------------------------------------------------------------------------------
# Execution
if __name__ == "__main__":
    main()