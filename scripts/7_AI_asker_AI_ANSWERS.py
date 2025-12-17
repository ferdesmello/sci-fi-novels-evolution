"""
This script uses GPT-5, via the OPENAI API, to answer questions about the plot 
of the novels scraped before, parses the answers, and saves it.

Modules:
    - os
    - pandas
    - openai
    - dotenv
    - tenacity
    - requests
    - logging
    - typing
    - datetime
    - winsound (only for Windows)
    - pathlib
    - sys
"""

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type, 
    wait_fixed
)
from requests.exceptions import RequestException
import logging
from typing import List
from datetime import datetime
import winsound

from pathlib import Path
import sys

#----------------------------------------------------------------------------------
# Make relative path imports work
# Make project root importable
sys.path.append(str(Path(__file__).resolve().parents[1]))
# Import paths
from paths import FILTERED, ANSWERS, KEYS, VARIABILITY_IN_ANSWERS

#----------------------------------------------------------------------------------
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from the .env file
load_dotenv(dotenv_path=KEYS / 'My_OPENAI_API_Key.env')

# Get the API key from the environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Configure the OpenAI client with the loaded API key
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
    print("OpenAI API key loaded successfully.")
else:
    print("Error: OPENAI_API_KEY not found in the environment variables. Make sure your .env file is correctly configured.")

#----------------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type((RequestException, Exception)), # Retry on API errors or network issues
    wait=wait_exponential(multiplier=1, min=4, max=60), # Exponential backoff: starts at 4 seconds, max 60 seconds
    stop=stop_after_attempt(10) # Stop after 10 attempts
)
def analyze_novel(title: str , author: str, year: int, synopsis: str, review: str, plot: str, genres: List[str]) -> str: 
    """
    Prompts GPT-5 to analyse the data of each novel and answers questions about it.

    Args:
        title (str): novel title.
        author (str): novel author.
        year (int): novel publishing year.
        synopsis (str): Goodreads synopsis for the novel.
        review (str): Chosen Goodreads review for the novel.
        plot (str): Wikipedia plot for the novel.
        genres (List[str]): Goodreads list of genres.

    Returns:
        answer (str): GPT-5 text answer.
    """

    prompt = f"""
    You are a helpful assistant and scholar of comparative sci-fi literature who analyzes novel plots based on your own knowledge and provided information.
    You received the task to carefully consider the plot of the novel "{title}" by {author}, published in {year}, focusing on key elements that will help you answer the following questions. 

    **Output Formatting Instructions**:
    Follow this exact format!

    Provide a concise paragraph summarizing the central and relevant elements of the novel, including:
    - Themes central to the story or that significantly impact the plot;
    - Memorable locations or time settings unique to the story;
    - Important events and conflicts in the story and its ending;
    - The main character, if there is one, and secondary characters;
    - Important creatures (aliens, robots, AIs, etc.) to the story;
    - Notable technologies that play a key role in the story.

    After the summary, leave one blank line and provide one answer per line in the following format:
    Question number, followed by a period, then the selected answer from the alternatives given, followed by a colon, and then the detailed but short justification in a single sentence.
    Example:
    1. Moderate: Scientific plausibility and speculation are equally important in the story.
    2. Mixed: The story balances elements of both soft and hard sciences.
    3. Heavy: The story is emotionally demanding, with complex ideas and scientific concepts.
    ...

    Important Notes:
    Ensure there are no line breaks, extra spaces, or symbols between answers.
    If no clear summary is possible, provide a brief explanation before answering.
    Answers must be one of the provided options.
    Answer in English.

    Important Clarifications:
    Extraterrestrials or alien life forms: They need not necessarily be intelligent, sapient, humanoid, or characters with moral agency. Any non-terrestrial organisms qualify (e.g., fauna, flora, microscopic life, alien ecologies, natural forces, native planetary or space life, etc.). Alien technology also qualifies, even without the aliens themselves.
    Robots and artificial intelligences: Include electronic, mechanical, or digital autonomous entities (e.g., robots, AIs, androids, automatons, etc.), conscious or not, characters with moral agency or not. Biological robots (e.g., engineered synthetic organisms framed as robots) are included. Exclude mere automated technologies and surveillance systems or humans acting like machines.
    Biotechnology and human biological alteration: Focus on intentional uses of biotechnology, genetic engineering, cloning, or biological modification. Exclude natural or accidental mutations unless explicitly framed as technological or deliberate.
    Transhumanism: Focus first on explicit depictions of tech-driven transcendence of human limitations (e.g., mind uploading, cybernetic enhancement, human or animal cognition uplift, radical life extension, merging humans with machines, etc.), and later, to a minor degree, on transcendence driven by other means.

    **Questions**:

    1. How important is scientific accuracy and plausibility in the story?
        Very high: Scientific accuracy is a central component, with the narrative driven by plausible scientific details.
        High: Scientific plausibility is important and generally consistent with known science.
        Moderate: A balanced mix, scientific plausibility is considered, but not always prioritized.
        Low: Some attention to plausibility, but scientific accuracy is secondary to other elements.
        Very low: Scientific accuracy is minimal or irrelevant, leaning heavily towards fantasy or speculation.
        Uncertain: Insufficient information.
    2. What is the main disciplinary focus of the story?
        Hard sciences: Physics, astronomy, biology, engineering, mathematics, etc., dominate the narrative.
        Leaning hard sciences: Hard topics dominate, with some soft sciences elements.
        Mixed: Balances elements of both soft and hard sciences somewhat equally.
        Leaning soft sciences: Soft topics dominate, with some hard sciences elements.
        Soft sciences: Psychology, sociology, politics, philosophy, anthropology, etc., dominate the narrative.
        Uncertain: Insufficient information.
    3. Is the story more of a light or heavy reading experience?
        Very heavy: Challenging, slow read, dense in language, themes, or ideas, focus on philosophical or intricate scientific concepts.
        Heavy: Intellectually or emotionally demanding, with complex ideas and deeper themes.
        Balanced: A mix of light and heavy elements, moderately complex and deep.
        Light: Somewhat accessible, with some thoughtful elements and themes, but still focused on entertainment.
        Very light: Easily accessible, fast read, minimal intellectual demands, focus on entertainment, humor, or adventure.
        Uncertain: Insufficient information.
    4. When does most of the story take place in relation to the year the novel was published?
        Distant future: Millennia or more ahead. 
        Far future: Centuries ahead. 
        Near future: Within a few decades ahead. 
        Present: Within a few years. 
        Near past: Within a few decades before. 
        Far past: Centuries before. 
        Distant past: Millennia or more before. 
        Multiple timelines: Distinct time periods without a single dominant timeframe. 
        Uncertain: Insufficient information.
    5. What is the mood of the story?
        Very optimistic: Overwhelmingly positive, uplifting, hopeful. 
        Optimistic: Positive outlook but with moments of pessimism. 
        Balanced: A mix of positive and negative moods without leaning towards one. 
        Pessimistic: Negative outlook but with moments of optimism. 
        Very pessimistic: Overwhelmingly negative, bleak, hopeless.
        Uncertain: Insufficient information.
    6. What is the overall mood and outcome of the story's ending?
        Very positive: Happy and optimistic, with major conflicts resolved favorably.
        Positive: Happy, but with some unresolved issues or bittersweet elements.
        Ambivalent: An equal mix of positive and negative outcomes.
        Negative: Sad or tragic, but with some elements of hope or redemption.
        Very negative: Tragic, bleak, and pessimistic, with major conflicts resolved unfavorably.
        Uncertain: Insufficient information.
    7. What is the social-political scenario depicted in the story?
        Utopic: Ideal or perfect society.
        Leaning utopic: Significant prosperity and desirable elements, but with some flaws.
        Balanced: A mix of both strengths and flaws elements, or an ordinary view of society.
        Leaning dystopic: Significant problems and undesirable elements, but with some strengths.
        Dystopic: Bleak, deeply flawed, authoritarian, and oppressive.
        Uncertain: Insufficient information.
    8. Is a unified, planetary-level or multi-planet state or government depicted in the story?
        Yes: A single, overarching political entity holds authority over an entire planet or a multi-planet civilization.
        Somewhat: A major political entity unifies most, but not all, of the setting, or its authority is heavily contested.
        No: Political authority is divided or fragmented among multiple, independent states or factions.
        Uncertain: Insufficient information.
    9. Is most of the story set on planet Earth?
        Yes: The story is mostly or entirely set on Earth.
        Somewhat: The story is equally divided between Earth and off-Earth locations.
        No: The story is mostly or entirely set off Earth, on other planets, in space, or other entirely off-Earth locations.
        Uncertain: Insufficient information.
    10. Is the setting of the story post-apocalyptic?
        Yes: Clear post-apocalyptic state, after a civilization-collapsing event.
        Somewhat: Just some elements of civilizational collapse are present, or the collapse is partial or local.
        No: The setting is not post-apocalyptic.
        Uncertain: Insufficient information.
    11. What is the dominant type of conflict in the story?
        Internal: Psychological, existential, or moral struggle.
        Interpersonal: Between individual characters.
        Societal: Society, societal norms, a large group, a political system, etc.
        Synthetic: Robots, artificial intelligences, androids, automatons, conscious artificial machines, etc.
        Technological: Machines and technology in general (excluding robots and AIs), science, engineering, inventions, automation, etc.
        Extraterrestrial: Aliens, extraterrestrial organisms (sapient or non-sapient), non-terrestrial beings, alien technology, etc.
        Natural: Environment, nature, natural forces, non-sentient life forms, etc.
        Mixed: Multiple conflicts are present without a clear dominant type.
        Uncertain: Insufficient information to determine.
    12. Does the story depict any type of extraterrestrial life forms?
        Yes: Clear depiction or mention of extraterrestrial life, extraterrestrial societies, or alien technology.
        Somewhat: Some form of extraterrestrial life or alien technology is present, but has a minor or background role.
        No: Not present in any form.
        Uncertain: Insufficient information.
    13. If the story depicts extraterrestrial life forms, how are they generally portrayed?
    (If you answered Somewhat to question 12, it still counts as depiction, not as Not applicable)
        Not applicable: Answered No to question 12, no extraterrestrial life forms or alien technology present.
        Good: Friendly, virtuous, helpful, or heroic. 
        Leaning good: Generally positive or benign but with flaws or minor conflicts. 
        Ambivalent: Morally ambiguous, showing both positive and negative traits, or multifaceted. 
        Leaning bad: Generally antagonistic or threatening but not entirely villainous. 
        Bad: Hostile, villainous, antagonistic, or threatening. 
        Non-moral: Amoral, non-sentient (e.g., animals, natural organisms, machines, forces, etc.), or acting solely based on instinct or programming.
        Uncertain: Insufficient information, or answered Uncertain to question 12.
    14. Does the story depict robots or artificial intelligences?
        Yes: Clear depiction or mention of robots, artificial intelligences, automatons, androids, conscious machines, etc.
        Somewhat: Some form of robot or artificial intelligence is present or discussed, but has a minor or background role.
        No: Not present in any form.
        Uncertain: Insufficient information.
    15. If the story depicts robots or artificial intelligences, how are they generally portrayed?
    (If you answered Somewhat to question 14, it still counts as depiction, not as Not applicable)
        Not applicable: Answered No to question 14, no robots or artificial intelligences present. 
        Good: Friendly, virtuous, helpful, or heroic. 
        Leaning good: Generally positive or benign but with flaws or minor conflicts. 
        Ambivalent: Morally ambiguous, showing both positive and negative traits, or multifaceted. 
        Leaning bad: Generally antagonistic or threatening but not entirely villainous. 
        Bad: Hostile, villainous, antagonistic, or threatening. 
        Non-moral: Amoral, non-sentient (e.g., instrumental machines, tools, etc.), or acting solely based on programming.
        Uncertain: Insufficient information, or answered Uncertain to question 14.
    16. Is there a single protagonist or main character in the story?
        Yes: One single character is the narrative's primary focus or center of the story.
        Somewhat: A main character is suggested but not strongly emphasized.
        No: No clear single protagonist, the story is an ensemble piece centered on multiple characters, a group, or a collective.
        Uncertain: Insufficient information.
    17. If the story has a single protagonist or main character, what is their nature?
    (If you answered Somewhat to question 16, it still counts as depiction, not as Not applicable)
        Not applicable: Answered No to question 16, no clear single protagonist or main character (e.g., multiple human or non-human main characters, a group or collective, etc.). 
        Human: The single protagonist is a human being (even if a cyborg or augmented).
        Non-human: The single protagonist is not a human being (e.g., animal, robot, AI, extraterrestrial, creature, post-human, geometric shape, abstract concept, human analog, etc.).
        Uncertain: Insufficient information, or answered Uncertain to question 16.
    18. If the story has a single protagonist or main character, what is their gender?
    (If you answered Somewhat to question 16, it still counts as depiction, not as Not applicable)
        Not applicable: Answered No to question 16, no clear single protagonist or main character to assign a gender to (there are multiple human or non-human main characters, a group or collective, etc.).
        Male: The single protagonist (human or non-human) is male. 
        Female: The single protagonist (human or non-human) is female. 
        Other: The single protagonist (human or non-human) is non-binary, gender ambiguous or fluid, or another gender identity that is not male nor female.
        Uncertain: Insufficient information, or answered Uncertain to question 16.
    19. If the story has a single protagonist or main character, how are they generally portrayed?
    (If you answered Somewhat to question 16, it still counts as depiction, not as Not applicable)
        Not applicable: Answered No to question 16, no clear single protagonist or main character to assign a gender to (e.g., multiple human or non-human main characters, a group or collective, etc.).
        Good: Friendly, virtuous, helpful, or heroic. 
        Leaning good: Generally positive or benign but with flaws or minor conflicts. 
        Ambivalent: Morally ambiguous, showing both positive and negative traits, or multifaceted. 
        Leaning bad: Generally antagonistic or threatening but not entirely villainous. 
        Bad: Hostile, villainous, antagonistic, or threatening. 
        Non-moral: Amoral (e.g., animals, natural organisms, instrumental machines, forces, etc.), or acting solely based on instinct or programming.
        Uncertain: Insufficient information, or answered Uncertain to question 16.
    20. Does the story depict virtual or augmented reality?
        Yes: Virtual reality, immersive digital environments, augmented reality, etc., have a central or significant role in the story.
        Somewhat: Some form of virtual reality or augmented reality is depicted or mentioned, but has a minor or background role, or is not framed as tech-driven.
        No: Not present in any form.
        Uncertain: Insufficient information.
    21. If the story depicts virtual or augmented reality, how are they generally depicted?
    (If you answered Somewhat to question 20, it still counts as depiction, not as Not applicable)
        Not applicable: Answered No to question 20, no virtual or augmented reality present.
        Good: Beneficial, constructive, and optimistic. 
        Leaning good: More positive and beneficial, but with some issues and harms. 
        Ambivalent: Mixed portrayal, with both significant benefits and harms.
        Leaning bad: More negative and harmful, but with redeeming features and benefits. 
        Bad: Harmful, destructive, dangerous, and pessimistic. 
        Instrumental: Neutral portrayal, used as neutral tools, without moral or ethical judgment.
        Uncertain: Insufficient information, or answered Uncertain to question 20.
    22. Does the story depict biotechnology, genetic engineering, or human biological alteration?
        Yes: Biotechnology, genetic engineering, cloning, or human biological modification, etc., have a central or significant role in the story.
        Somewhat: Some form of biotechnology or human biological alteration is depicted or mentioned in the story, but has a minor or background role, or is not framed as tech-driven.
        No: Not present in any form.
        Uncertain: Insufficient information.
    23. If the story depicts biotechnology, genetic engineering, or human biological alteration, how are they generally depicted?
    (If you answered Somewhat to question 22, it still counts as depiction, not as Not applicable)
        Not applicable: Answered No to question 22, no biotechnology, genetic engineering, or human biological alteration present.
        Good: Beneficial, constructive, useful, and optimistic. 
        Leaning good: More positive and beneficial, but with some issues and harms. 
        Ambivalent: Mixed portrayal, with both significant benefits and harms.
        Leaning bad: More negative and harmful, but with redeeming features and benefits. 
        Bad: Harmful, destructive, dangerous, and pessimistic. 
        Instrumental: Neutral portrayal, used as neutral tools, without moral or ethical judgment.
        Uncertain: Insufficient information, or answered Uncertain to question 22.
    24. Does the story depict transhumanism or the transcendence of human limitations?
        Yes: Transhumanist ideas (e.g., mind uploading, cybernetic enhancement, radical life extension, merging humans with machines, etc.) are central or significant to the story.
        Somewhat: Transhumanist elements appear, but in a minor or background role, or are not framed as tech-driven.
        No: Not present in any form.
        Uncertain: Insufficient information.
    25. If the story depicts transhumanism or the transcendence of human limitations, how are they generally depicted?
    (If you answered Somewhat to question 24, it still counts as depiction, not as Not applicable)
        Not applicable: Answered No to question 24, no transhumanism or the transcendence of human limitations present.
        Good: Beneficial, constructive, useful, and optimistic. 
        Leaning good: More positive and beneficial, but with some issues and harms. 
        Ambivalent: Mixed portrayal, with both significant benefits and harms.
        Leaning bad: More negative and harmful, but with redeeming features and benefits. 
        Bad: Harmful, destructive, dangerous, and pessimistic. 
        Instrumental: Neutral portrayal, used as neutral tools, without moral or ethical judgment.
        Uncertain: Insufficient information, or answered Uncertain to question 24.
    26. How are science and technology generally depicted in the story?
        Good: Beneficial, constructive, useful, and optimistic. 
        Leaning good: More positive and beneficial, but with some issues and harms. 
        Ambivalent: Mixed portrayal, with both significant benefits and harms.
        Leaning bad: More negative and harmful, but with redeeming features and benefits. 
        Bad: Harmful, destructive, dangerous, and pessimistic. 
        Instrumental: Neutral portrayal, used as neutral tools, without moral or ethical judgment.
        Uncertain: Insufficient information.
    27. How central is the critique of specific social issues in the story?
        Core: The critique of social issues (e.g., inequality, war, discrimination, political oppression, etc.) is the main driver of the story or a key theme.
        Major: Significant role but secondary theme.
        Minor: Subtle role or minimal theme.
        Absent: Not present.
        Uncertain: Insufficient information.
    28. How central are ecological or environmental themes in the story?
        Core: They (e.g., climate change, pollution, preservation, conservation, extinction, symbiosis, etc.) are the main driver of the story or a key theme.
        Major: Significant role but secondary theme.
        Minor: Subtle role or minimal theme.
        Absent: Not present.
        Uncertain: Insufficient information.

    When answering, consider how the following synopsis, review, plot summary, and genres may provide relevant data and context for each question.

    novel synopsis: {synopsis}
    Partial review: {review}
    Plot summary: {plot}
    Genres the novel fits in: {genres}
    """

    # Call the OpenAI API with the crafted prompt
    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            max_output_tokens=4000,
            reasoning={"effort": "medium"}, # can be "low", "medium", or "high"
            text={"verbosity":"low"} # can be "low", "medium", or "high"
        )

        # Extract and print the response
        answer = response.output_text
        print(f'\n"{title}" by {author}, {year}')
        print(answer)

    except Exception as e:
        print("Error:", e)

    return answer

#----------------------------------------------------------------------------------
@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
def ask_to_AI(df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """
    Sends novel data to the prompt function, parses the returned answers, and merges everything together.

    Args:
        df (pandas.DataFrame): Dataframe with all novels information.
        output_file (str): Output file name to save the progress.

    Returns:
        df_processed (pandas.DataFrame): Dataframe with all original novels information and 
            processed answers about them from GPT-5.
    """

    #----------------------------------------
    # Valid answers for each question
    valid_answers = [
        {"Very high", "High", "Moderate", "Low", "Very low", "Uncertain"},
        {"Hard sciences", "Leaning hard sciences", "Mixed", "Leaning soft sciences", "Soft sciences", "Uncertain"},
        {"Very heavy", "Heavy", "Balanced", "Light", "Very light", "Uncertain"},
        {"Distant future", "Far future", "Near future", "Present", "Near past", "Far past", "Distant past", "Multiple timelines", "Uncertain"},
        {"Very optimistic", "Optimistic", "Balanced", "Pessimistic", "Very pessimistic", "Uncertain"},
        {"Very positive", "Positive", "Ambivalent", "Negative", "Very negative", "Uncertain"},

        {"Utopic", "Leaning utopic", "Balanced", "Leaning dystopic", "Dystopic", "Uncertain"},
        {"Yes", "Somewhat", "No", "Uncertain"},
        {"Yes", "Somewhat", "No", "Uncertain"},
        {"Yes", "Somewhat", "No", "Uncertain"},
        {"Internal", "Interpersonal", "Societal", "Synthetic", "Technological", "Extraterrestrial", "Natural", "Mixed", "Uncertain"},

        {"Yes", "Somewhat", "No", "Uncertain"},
        {"Not applicable", "Good", "Leaning good", "Ambivalent", "Leaning bad", "Bad", "Non-moral", "Uncertain"},
        {"Yes", "Somewhat", "No", "Uncertain"},
        {"Not applicable", "Good", "Leaning good", "Ambivalent", "Leaning bad", "Bad", "Non-moral", "Uncertain"},

        {"Yes", "Somewhat", "No", "Uncertain"},
        {"Not applicable", "Human", "Non-human", "Uncertain"},
        {"Not applicable", "Male", "Female", "Other", "Uncertain"},
        {"Not applicable", "Good", "Leaning good", "Ambivalent", "Leaning bad", "Bad", "Non-moral", "Uncertain"},

        {"Yes", "Somewhat", "No", "Uncertain"},
        {"Not applicable", "Good", "Leaning good", "Ambivalent", "Leaning bad", "Bad", "Instrumental", "Uncertain"},
        {"Yes", "Somewhat", "No", "Uncertain"},
        {"Not applicable", "Good", "Leaning good", "Ambivalent", "Leaning bad", "Bad", "Instrumental", "Uncertain"},
        {"Yes", "Somewhat", "No", "Uncertain"},
        {"Not applicable", "Good", "Leaning good", "Ambivalent", "Leaning bad", "Bad", "Instrumental", "Uncertain"},
        {"Good", "Leaning good", "Ambivalent", "Leaning bad", "Bad", "Instrumental", "Uncertain"},

        {"Core", "Major", "Minor", "Absent", "Uncertain"},
        {"Core", "Major", "Minor", "Absent", "Uncertain"},
    ]

    question_columns = [
        '1 accuracy',
        '2 discipline',
        '3 light heavy',
        '4 time',
        '5 mood',
        '6 ending',

        '7 social political',
        '8 politically unified',
        '9 on Earth',
        '10 post apocalyptic',
        '11 conflict',

        '12 aliens',
        '13 aliens are',
        '14 robots and AI',
        '15 robots and AI are',

        '16 protagonist',
        '17 protagonist nature',
        '18 protagonist gender',
        '19 protagonist is',

        '20 virtual',
        '21 virtual is',
        '22 biotech',
        '23 biotech is',
        '24 transhuman',
        '25 transhuman is',
        '26 tech and science',

        '27 social issues',
        '28 enviromental',
    ]

    justification_columns = [
        'justifying accuracy',
        'justifying discipline',
        'justifying light heavy',
        'justifying time',
        'justifying mood',
        'justifying ending',

        'justifying social political',
        'justifying politically unified',
        'justifying on Earth',
        'justifying post apocalyptic',
        'justifying conflict',

        'justifying aliens',
        'justifying aliens are',
        'justifying robots and AI',
        'justifying robots and AI are',

        'justifying protagonist',
        'justifying protagonist nature',
        'justifying protagonist gender',
        'justifying protagonist is',

        'justifying virtual',
        'justifying virtual is',
        'justifying biotech',
        'justifying biotech is',
        'justifying transhuman',
        'justifying transhuman is',
        'justifying tech and science',

        'justifying social issues',
        'justifying enviromental',
    ]

    #----------------------------------------
    # Load existing progress if the file exists
    if os.path.exists(output_file):
        df_processed = pd.read_csv(output_file, sep=';', encoding='utf-8-sig')
        processed_novels = set(df_processed['url goodreads'])
    else:
        df_processed = pd.DataFrame()
        processed_novels = set()
    for _, novel in df.iterrows():
        # Skip already processed novels
        if novel['url goodreads'] in processed_novels:
            continue

        # Extract novel details
        title = novel['title']
        author = novel['author']
        year = int(novel['year'])
        decade = int(novel['decade'])
        #pages = novel['pages']
        rate = float(novel['rate'])
        ratings = int(novel['ratings'])
        series = novel['series']
        genres = novel['genres']
        synopsis = novel['synopsis']
        review = novel['review']
        url_g = novel['url goodreads']
        plot = novel['plot']
        url_w = novel['url wikipedia']

        forbidden_titles = [] # Empty for now. Add titles here if you want to force "No plot available."

        if pd.isna(plot) or len(plot) < 100 or title in forbidden_titles:
            plot = 'No plot available.'

        """if len(plot) > (len(synopsis) + len(review)): # and title not in forbidden_titles:
            synopsis = 'No synopsis available.'
            review = 'No review available.'"""
        
        #print(f"synopsis: {synopsis}\n")
        #print(f"review: {review}\n")
        #print(f"plot: {plot}\n")
        #print(f"genres: {genres}\n")

        #----------------------------------------
        number_of_lines = 30 # 1 paragraph + 1 blank + 28 answers
        number_of_answers = 28 # 28 answers

        try:
            # Get the AI's answers for the novel
            AI_answers = analyze_novel(title, author, year, synopsis, review, plot, genres)

            # Split answers into a list of, hopefully, 28 items
            answers = []
            justifications = []
            lines = [line.strip() for line in AI_answers.split('\n')]

            # Guarantee at least two lines (paragraph + at least one answer)
            if len(lines) < 2 or len(lines) > number_of_lines:
                lines = ["", ""]

            # First line is always the paragraph text
            paragraph = lines[0]

            # Detect where answers start (skip blank second line if present)
            start_idx = 2 if len(lines) == number_of_lines and lines[1] == "" else 1

            # Process each answer line
            for line in lines[start_idx:]:
                if not line:
                    continue

                # Split into number + rest
                parts_number_text = line.split('. ', 1)
                if len(parts_number_text) < 2:
                    logging.warning(f"Skipping malformed line (no '. '): {line}")
                    continue

                # Split into answer + justification
                parts_answer_just = parts_number_text[1].split(': ', 1)
                answer = parts_answer_just[0].strip()
                justification = parts_answer_just[1].strip() if len(parts_answer_just) == 2 else None

                answers.append(answer)
                justifications.append(justification)

            #----------------------------------------
            # Append answers to respective lists
            complete_answer = AI_answers

            # Check if we have exactly 28 answers and 28 justifications
            if (len(answers) == number_of_answers) & (len(justifications) == number_of_answers):

                # One pass: distribute each answer into the right result list
                for i in range(len(answers)):

                    # Check for valid answers
                    if answers[i] in valid_answers[i]:
                        continue
                    else:
                        answers[i] = None
                        justifications[i] = None
                        logging.warning(f"Invalid answer for question {i+1} in novel: {title}")
                
                # Test for coherence in follow-up answers
                # Aliens----------------------------------------
                if (answers[11] == "No") & ((answers[12] != "Not applicable") | (answers[10] == "Extraterrestrial")):
                    answers[12] = None
                    justifications[12] = None
                    logging.warning(f"Incoherent alien answer for novel: {title}\n")
                
                if (answers[11] == "Somewhat") & (answers[12] == "Not applicable"):
                    answers[12] = None
                    justifications[12] = None
                    logging.warning(f"Incoherent alien answer for novel: {title}\n")
                
                if (answers[11] == "Uncertain") & ((answers[12] != "Uncertain") | (answers[10] == "Extraterrestrial")) :
                    answers[12] = None
                    justifications[12] = None
                    logging.warning(f"Incoherent alien answer for novel: {title}\n")

                # Robots and AIs----------------------------------------
                if (answers[13] == "No") & ((answers[14] != "Not applicable") | (answers[10] == "Synthetic")) :
                    answers[14] = None
                    justifications[14] = None
                    logging.warning(f"Incoherent robot and AI answer for novel: {title}\n")
                                                
                if (answers[13] == "Somewhat") & (answers[14] == "Not applicable"):
                    answers[14] = None
                    justifications[14] = None
                    logging.warning(f"Incoherent robot and AI answer for novel: {title}\n")

                if (answers[13] == "Uncertain") & ((answers[14] != "Uncertain") | (answers[10] == "Synthetic")) :
                    answers[14] = None
                    justifications[14] = None
                    logging.warning(f"Incoherent robot and AI answer for novel: {title}\n")

                # Protagonist----------------------------------------
                if (answers[15] == "No") & ((answers[16] != "Not applicable") | (answers[17] != "Not applicable") | (answers[18] != "Not applicable")) :
                    answers[16] = None
                    answers[17] = None
                    answers[18] = None
                    justifications[16] = None
                    justifications[17] = None
                    justifications[18] = None
                    logging.warning(f"Incoherent protagonist answer for novel: {title}\n")

                if (answers[15] == "Somewhat") & ((answers[16] == "Not applicable") | (answers[17] == "Not applicable") | (answers[18] == "Not applicable")) :
                    answers[16] = None
                    answers[17] = None
                    answers[18] = None
                    justifications[16] = None
                    justifications[17] = None
                    justifications[18] = None
                    logging.warning(f"Incoherent protagonist answer for novel: {title}\n")

                if (answers[15] == "Uncertain") & ((answers[16] != "Uncertain") | (answers[17] != "Uncertain") | (answers[18] != "Uncertain")) :
                    answers[16] = None
                    answers[17] = None
                    answers[18] = None
                    justifications[16] = None
                    justifications[17] = None
                    justifications[18] = None
                    logging.warning(f"Incoherent protagonist answer for novel: {title}\n")

                # Virtual----------------------------------------
                if (answers[19] == "No") & (answers[20] != "Not applicable") :
                    answers[20] = None
                    justifications[20] = None
                    logging.warning(f"Incoherent virtual answer for novel: {title}\n")
                
                if (answers[19] == "Somewhat") & (answers[20] == "Not applicable") :
                    answers[20] = None
                    justifications[20] = None
                    logging.warning(f"Incoherent virtual answer for novel: {title}\n")

                if (answers[19] == "Uncertain") & (answers[20] != "Uncertain") :
                    answers[20] = None
                    justifications[20] = None
                    logging.warning(f"Incoherent virtual answer for novel: {title}\n")
                
                # Biotech----------------------------------------
                if (answers[21] == "No") & (answers[22] != "Not applicable") :
                    answers[22] = None
                    justifications[22] = None
                    logging.warning(f"Incoherent biotech answer for novel: {title}\n")

                if (answers[21] == "Somewhat") & (answers[22] == "Not applicable") :
                    answers[22] = None
                    justifications[22] = None
                    logging.warning(f"Incoherent biotech answer for novel: {title}\n")

                if (answers[21] == "Uncertain") & (answers[22] != "Uncertain") :
                    answers[22] = None
                    justifications[22] = None
                    logging.warning(f"Incoherent biotech answer for novel: {title}\n")

                # Transhuman----------------------------------------
                if (answers[23] == "No") & (answers[24] != "Not applicable") :
                    answers[24] = None
                    justifications[24] = None
                    logging.warning(f"Incoherent transhuman answer for novel: {title}\n")

                if (answers[23] == "Somewhat") & (answers[24] == "Not applicable") :
                    answers[24] = None
                    justifications[24] = None
                    logging.warning(f"Incoherent transhuman answer for novel: {title}\n")

                if (answers[23] == "Uncertain") & (answers[24] != "Uncertain") :
                    answers[24] = None
                    justifications[24] = None
                    logging.warning(f"Incoherent transhuman answer for novel: {title}\n")

            else:
                # Something went wrong and better fill a row of None in each list
                for i in range(len(answers)):
                    answers[i] = None
                    justifications[i] = None

                logging.warning(f"Unexpected number of answers for novel: {title}\nFound {len(answers)}/28 answers and {len(justifications)}/28 justifications.")

            #----------------------------------------
            # One-row dataframe to save the progress in the present novel/row
            data = {
                "title": [title],
                "author": [author],
                "year": [year],
                "paragraph": [paragraph],
                "complete answer": [complete_answer],
                "decade": [decade],
                # "pages": [pages],
                "rate": [rate],
                "ratings": [ratings],
                "series": [series],
                "genres": [genres],
                "synopsis": [synopsis],
                "review": [review],
                "url goodreads": [url_g],
                "plot": [plot],
                "url wikipedia": [url_w],
            }

            # Add the repeated questions/justifications columns
            if (len(answers) == number_of_answers):
                for i in range(number_of_answers):
                    data[question_columns[i]] = [answers[i]]
                    data[justification_columns[i]] = [justifications[i]]

            else:
                for i in range(number_of_answers):
                    data[question_columns[i]] = [None]
                    data[justification_columns[i]] = [None]
                logging.warning(f"Filling None for all answers and justifications for novel: {title}")

            # Build DataFrame
            df_progress = pd.DataFrame(data)

            #----------------------------------------
            column_order = [
                'title', 
                'author', 
                'year',

                'paragraph',

                '1 accuracy',
                'justifying accuracy',
                '2 discipline',
                'justifying discipline',
                '3 light heavy',
                'justifying light heavy',
                '4 time',
                'justifying time',
                '5 mood',
                'justifying mood',
                '6 ending',
                'justifying ending',

                '7 social political',
                'justifying social political',
                '8 politically unified',
                'justifying politically unified',
                '9 on Earth',
                'justifying on Earth',
                '10 post apocalyptic',
                'justifying post apocalyptic',
                '11 conflict',
                'justifying conflict',

                '12 aliens',
                'justifying aliens',
                '13 aliens are',
                'justifying aliens are',
                '14 robots and AI',
                'justifying robots and AI',
                '15 robots and AI are',
                'justifying robots and AI are',

                '16 protagonist',
                'justifying protagonist',
                '17 protagonist nature',
                'justifying protagonist nature',
                '18 protagonist gender',
                'justifying protagonist gender',
                '19 protagonist is',
                'justifying protagonist is',

                '20 virtual',
                'justifying virtual',
                '21 virtual is',
                'justifying virtual is',
                '22 biotech',
                'justifying biotech',
                '23 biotech is',
                'justifying biotech is',
                '24 transhuman',
                'justifying transhuman',
                '25 transhuman is',
                'justifying transhuman is',
                '26 tech and science',
                'justifying tech and science',

                '27 social issues',
                'justifying social issues',
                '28 enviromental',
                'justifying enviromental',

                'complete answer',

                'decade', 
                'rate', 
                'ratings', 
                'series', 
                'genres', 
                'synopsis',
                'review',
                'url goodreads',
                'plot',
                'url wikipedia',
            ]

            # Check for duplicate columns in column_order
            if len(column_order) != len(set(column_order)):
                raise ValueError("Duplicate columns in column_order")

            # Reorder the columns
            df_progress = df_progress.reindex(columns=column_order)

            #----------------------------------------
            # Concatenate the one-row dataframe with the big dataframe with all anterior novels/rows
            df_processed = pd.concat([df_processed, df_progress], ignore_index=True)
            df_processed.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
            logging.info(f"Progress saved for novel: {title}")

        except Exception as e:
            # Log the full AI response that caused the error for debugging
            logging.error(f"Failed to analyze novel {title}. Error: {e}")
            logging.error(f"Problematic AI response: {AI_answers}")
            raise # Re-raise the exception to trigger a retry

    return df_processed

#----------------------------------------------------------------------------------
def main():
    """
    Main execution function for the script.
    Calls the AI asker function, orders the data, and saves it in a CSV file.
    """

    #------------------------------------------
    # Name of the input file
    #input_file = FILTERED / 'sci-fi_novels_TEST_Wiki_small.csv'
    #input_file = FILTERED / 'sci-fi_novels_TEST_Wiki.csv'
    input_file = FILTERED / 'sci-fi_novels_TOP_Wiki.csv'

    # Name of the output file
    #output_file = ANSWERS / 'sci-fi_novels_AI_ANSWERS_TEST_small.csv'
    #output_file = ANSWERS / 'sci-fi_novels_AI_ANSWERS_TEST.csv'
    #output_file = VARIABILITY_IN_ANSWERS / 'sci-fi_novels_AI_ANSWERS_TEST_01.csv'
    output_file = ANSWERS / 'sci-fi_novels_AI_ANSWERS.csv'

    #------------------------------------------
    # Load novel data to send to the AI
    df = pd.read_csv(input_file, sep=';', encoding="utf-8-sig")

    # Ask the AI about ALL the novels
    df_processed = ask_to_AI(df, output_file)

    # Retyping columns of the processed dataframe
    df_processed['year'] = df_processed['year'].astype(int)
    df_processed['decade'] = df_processed['decade'].astype(int)
    df_processed['rate'] = df_processed['rate'].astype(float)
    df_processed['ratings'] = df_processed['ratings'].astype(int)

    #------------------------------------------
    # Sometimes the AI output is not formatted right or is null

    # Get all columns from "paragraph" through "complete answer"
    cols_to_check = df_processed.loc[:, "paragraph":"complete answer"].columns

    # Drop rows with any null in that slice
    df_processed = df_processed.dropna(axis=0, subset=cols_to_check, how="any", ignore_index=True)
    df_processed = df_processed.sort_values(by = ['decade', 'year', 'author', 'title'], ascending=True)

    #------------------------------------------
    size_in = df.shape[0] # Number of rows
    size_out = df_processed.shape[0] # Number of rows
    missing_novels = size_in - size_out # Difference in number of rows

    #------------------------------------------
    df_processed.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
    print(f"\nData saved to {output_file}")

    return missing_novels, df_processed # How many novels were not processed right and the DataFrame

#----------------------------------------------------------------------------------
# Execution
if __name__ == "__main__":

    # Record start time
    start = datetime.now()

    missing_novels = 1
    max_retries = 20
    attempt = 1

    while (missing_novels != 0) and (attempt <= max_retries):
        missing_novels, df_processed = main()
        print(f"novel(s) missing: {missing_novels}.\nAttempts made: {attempt}.\n")
        attempt += 1
    
    print("\n", df_processed.info())

    # Record end time
    end = datetime.now()

    # How long did it take?
    print(f"Script started at {start}")
    print(f"Script finished at {end}")
    print(f"Total runtime: {end - start}")

    winsound.Beep(800, 500) # Play a 800 Hz beep for 500 milliseconds
    winsound.Beep(500, 500) # Play a 500 Hz beep for 500 milliseconds
    winsound.Beep(300, 500) # Play a 300 Hz beep for 500 milliseconds