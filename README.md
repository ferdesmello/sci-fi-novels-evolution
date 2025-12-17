# sci-fi-novels-evolution

## Overview

Inspired by this project ([video](https://www.youtube.com/watch?v=nRQ2vMpw-n8), [page](https://pudding.cool/2024/07/scifi/)) about the evolution in the plot of sci-fi movies and series, and this [comment](https://www.youtube.com/watch?v=nRQ2vMpw-n8&lc=UgyRg89P8kRYQ2SdXrV4AaABAg), I decided to do a similar analysis but for sci-fi _novels_.

You can read in detail about the entire process of data analysis and the results [here](https://fdesmello.wordpress.com/2024/11/21/evolution-of-sci-fi-novels-using-data-and-ai/).

But, in short, I scraped data about thousands of sci-fi books from [Goodreads](https://www.goodreads.com/) lists and shelves, and the books' [Wikipedia](https://en.wikipedia.org/wiki/Main_Page) articles, cleaned and reduced the data, selected the top 200 novels per decade (or all the novels if fewer than 200 per decade), and fed that into GPT-5 (in the first tries, I used GPT-4o and Gemini 2.0 Flash) via the OpenAI API, asking about many plot-related things. Then, I aggregated the results in figures to see how things changed over time.

## Repository structure

<pre>
├── data/                                # Data to read, write, and modify
│   ├── answers/                         # AI answers data
│   ├── brute/                           # Raw scraped data (this folder will be created when the scraper scripts are run)
│   ├── filtered/                        # Filtered data to use with the AI
│   └── variability_in_answers/          # Data for the test in answers variability
├── figures/                             # All the figures
├── old_gemini_2.0_flash/                # Old code, answers, and figures using Gemini 2.0 flash
├── old_gpt-4o/                          # Old code, answers, and figures using GPT-4o
├── scripts/                             # All Python files to run the analysis
│   ├── 1_Goodreads_scraper_SHELF.py     # Scrap shelf data from Goodreads
│   ├── 2_Goodreads_scraper_LISTS.py     # Scrap lists data from Goodreads
│   ├── 3_Data_reducer_FILTERED.py       # Reduce and filter raw or brute data
│   ├── 4_Data_reducer_TOP.py            # Select top data for analysis
│   ├── 5_Data_fixer.py                  # Fix possible problems with data
│   ├── 6_Wikipedia_plot_scraper.py      # Scrap plots data from Wikipedia
│   ├── 7_AI_asker_AI_ANSWERS.py         # Main AI answers
│   ├── 8_AI_asker_AI_ANSWERS_GENDER.py  # AI answers for author's gender
│   ├── 9_Figure_maker_novels.py         # Figures for the general features in the novel samples
│   ├── 9_Figure_maker_questions.py      # Figures for the man AI answers
│   └── 9_Figure_maker_tests.py          # Figures for some statistical tests and more
├── .gitattributes
├── .gitignore
├── LICENSE
├── paths.py
└── README.md
</pre>

## What the code does

### 1. Scraping the data

I intended to scrape data directly from the [science fiction book shelf](https://www.goodreads.com/shelf/show/science-fiction) on Goodreads, but it didn't work, and for some error, it only shows books until page [25](https://www.goodreads.com/shelf/show/science-fiction?page=25). I got around that by downloading the pages and reading them locally.

**1_Goodreads_scraper_SHELF.py** reads the HTML files in the folder **Data/Saved_pages** and searches for links to the book pages and downloads data for book's _title_, _author_, _year_ published, number of _pages_, if it is part of a _series_, average _rate_, number of _ratings_, _genres_, _synopsis_ and the longest of the first two _reviews_, saving everything in the **sci-fi_books_SHELF.csv** file at the end.

You will need to download and manually place the HTML files in the Saved_pages folder. I didn't include them here because they are big and use too much space.

**2_Goodreads_scraper_LISTS.py** does the same, but from a list of URLs for the initial pages of thematic book lists, saving the progress in a JSON file and everything in the **sci-fi_books_LISTS.csv** file at the end. 

All data recovered and processed is stored in the **Data/Brute** folder.

#

### 2. Cleaning the data

**3_Data_reducer_FILTERED.py** merges the datafiles, creating **sci-fi_books_BRUTE.csv**, and cleans them (many duplicates, bad data, and books not of interest), creating the **sci-fi_books_FILTERED.csv** file.

**4_Data_reducer_TOP.py** selects the top 200 novels per decade (by the number of user ratings) and saves them as **sci-fi_books_TOP.csv**. It also creates the **sci-fi_books_TEST.csv** file, with just a small selection of the novels to test the AI performance. 

If this is your first time running everything, you can proceed to step 3. But if you have already run everything to the end and are just adding some books from the scraper (or deleting books via the filtering), that may change which books are in the top 200 per decade. Some rows from the AI answers' CSV may need to be excluded, and/or new ones may need to be processed. For this, run **5_Data_fixer.py** _now_.

Many of the novels in the top sample have a Wikipedia article with a plot section that gives much more details about the plot than Goodreads synopses and reviews, so they are preferable.

**6_Wikipedia_plot_scraper.py** searches the wikipedia after the novels listed in **sci-fi_books_TOP.csv** and creates the **sci-fi_books_TOP_Wiki.csv** and **sci-fi_books_TEST_Wiki.csv** files with the plot sections found in the articles.

All the files are stored in the **Data/Filtered** folder.

#

### 3. Prompting the LLM

**7_AI_asker_AI_ANSWERS.py** reads the **sci-fi_books_TOP_Wiki.csv** file (or **sci-fi_books_TEST_Wiki.csv**) and, for every novel in the file, sends the prompt with the novel's data to OpenAI's API for GPT-5 and receives a text answer, parses it, and saves it in the **sci-fi_books_AI_ANSWERS.csv** file in the **Data/Answers** folder.

In the **Data/Variability_in_Answers** folder, there are already 15 answer files to be used with the **Figure_maker.py** script to estimate the model change in answer for each novel and question.

**8_AI_asker_AI_ANSWERS_GENDER.py** reads the **sci-fi_books_TOP.csv** file and, for every different _author_ in the file, sends the prompt with the author's name to GPT-5 and receives the author's gender, and saves it in the **sci-fi_books_AI_ANSWERS_GENDER.csv** file.

You will need a working API key set in your environment and credits in your OpenAI account for them to work.

Older results and some code for GPT-4o and Gemini 2.0 Flash are stored in separate folders (**Old_GPT-4o** and **Old_Gemini_2.0_flash**).

#

### 4. Plotting the results

Figure making is divided into three parts: one for the sample (**9_Figure_maker_novel.py**), one for the answers to the questions (**9_Figure_maker_questions.py**), and one for some tests (**9_Figure_maker_tests.py**). They all read from **sci-fi_books_AI_ANSWERS.csv**,  **sci-fi_books_AI_ANSWERS_GENDER.csv**, and the files in the **Data/Variability_in_Answers** folder, and make figures from them, saving all of them in the **Figures** folder.

## Example of a figure

The figure below is one example of the output.

![A final plot as an example.](./figures/04%20time.png "When does most of the story take place in relation to the year the book was published? Distant past: millennia or more before; Far past: centuries before; Near past: within a few decades before; Present: within a few years; Near future: within a few decades ahead; Far future: centuries ahead; Distant future: millennia or more ahead; Multiple timelines: distinct time periods without a single dominant timeframe; Uncertain: Not enough information to say, unclear.")
