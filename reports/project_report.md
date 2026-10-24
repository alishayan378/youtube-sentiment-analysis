# YouTube Comments Sentiment Analysis

**Student Name:** [Your Name]  
**Course / Section:** [Course and Section]  
**Instructor:** [Instructor Name]  
**GitHub Repository:** [Paste your repository URL after uploading]

## 1. Introduction and Objective

YouTube comments contain opinions, reactions, and feedback about video content. Reading a large number of comments manually can be time-consuming. This project aims to analyze up to 2,000 comments from the selected YouTube video (https://youtu.be/kAgAQ24_Jso) and categorize each comment as **positive**, **neutral**, or **negative** using Natural Language Processing (NLP). The project also summarizes the sentiment distribution through charts.

## 2. Tools and Methodology

The system is implemented in Python and designed to run in Visual Studio Code. The YouTube Data API v3 is used to collect published top-level comments in pages of up to 100 items. The program follows the API's next-page token until 2,000 unique comments have been collected or no more comments are available. Empty comments and duplicate comment IDs are excluded.

For sentiment classification, the project uses the pretrained `lxyuan/distilbert-base-multilingual-cased-sentiments-student` transformer model through Hugging Face Transformers. Each comment receives a predicted sentiment label and a model confidence score. Pandas stores the results, while Matplotlib creates bar and pie charts. The outputs are saved as CSV and JSON files so the findings can be checked and reproduced.

## 3. Results

**Complete this section after running `python src/main.py`; do not submit estimates as measured results.**

- Comments collected and analyzed: **[copy from terminal / summary.json]**
- Positive: **[count] ([percentage]%)**
- Neutral: **[count] ([percentage]%)**
- Negative: **[count] ([percentage]%)**
- Most frequent sentiment: **[positive / neutral / negative, based on actual output]**

Insert `outputs/sentiment_bar_chart.png` and/or `outputs/sentiment_pie_chart.png` into the final report if space allows. Discuss the observed distribution in one or two sentences based on the generated charts.

## 4. Conclusion and Limitations

This project provides a reproducible workflow for collecting YouTube comments, estimating their sentiment, and visualizing the results. It can help summarize the general tone of the retrieved comments, but model predictions should not be treated as perfectly accurate. In particular, the model card lists 12 languages but does not list Urdu; therefore, Urdu and Roman Urdu, code-switching, sarcasm, slang, and emoji-based meanings may be misclassified. The API returns top-level comments only and the video may not have 2,000 accessible comments. Future work could manually label a sample of comments to estimate performance and compare alternative models.

## References

1. YouTube Data API v3, Comment Threads: List. https://developers.google.com/youtube/v3/docs/commentThreads/list  
2. Hugging Face model card. https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student
