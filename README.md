# LegalDocNLP
Legal Document Summarizer Project for QMIND

# Jan 25, 2022
There exist two types of ways to create a text summary from an AI, which is expolained below by Jason Brownlee from machinelearningmastery.com:

"There are two main approaches to summarizing text documents; they are:

1. Extractive Methods.
2. Abstractive Methods.

"The different dimensions of text summarization can be generally categorized based on its input type (single or multi document), purpose (generic, domain specific, or query-based) and output type (extractive or abstractive)."

— A Review on Automatic Text Summarization Approaches, 2016.

Extractive text summarization involves the selection of phrases and sentences from the source document to make up the new summary. Techniques involve ranking the relevance of phrases in order to choose only those most relevant to the meaning of the source.

Abstractive text summarization involves generating entirely new phrases and sentences to capture the meaning of the source document. This is a more challenging approach, but is also the approach ultimately used by humans. Classical methods operate by selecting and compressing content from the source document."

We will likely be focusing on extractive simply because it'll be a less tedious and tough pill to swallow. As such, we'll need to gather up the HTML files from Canada Gazette and treat them accordingly. 

A really great starting point for this material is done on Github by user LaurentVeyssier. See his project here:

We'll be doing our first attempts for this project using his methodology and attempting a solution with TextRank. Should that not work, another common algorithm is seq2seq but this is a bit more technical and requires knowledge of RNN's so we'll save it as a backup for now.

To prepare for the implementation for our next meeting, please move about 10 Canada Gazette sections into a text file and dump them into the created folder "Gazette Extractions". Gazette sections aren't the whole thing, so no need to freak out. They're just the smaller pieces of into separated by headings. See the folder for some examples that I moved in for now. From there, we'll discuss next steps.

# Feb 2, 2022
We'll be converting the text files in the extractions folde rinto a readable CSV. CSV's are the base format for most, if not all, ML datasets. We'll be creating our CSV from Google Sheets, which is n the running meeting notes google doc. From here, we'll bigin the actual coding scrums, and have created either a Python Jupyter Notebook or raw Python file. Things are moving along!
