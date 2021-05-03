from rake_nltk import Rake
rake_nltk_var = Rake()

text = """ I am a programmer from India, and I am here to guide you 
with Data Science, Machine Learning, Python, and C++ for free. 
I hope you will learn a lot in your journey towards Coding, 
Machine Learning and Artificial Intelligence with me."""

rake_nltk_var.extract_keywords_from_text(text)
keyword_extracted = rake_nltk_var.get_ranked_phrases()
print(keyword_extracted)


'''
Summary
The process of extracting keywords helps us identifying the importance of words in a text. This task can be also used for topic modelling. It is very useful to extract keywords for indexing the articles on the web so that people searching the keywords can get the best articles to read.

This technique is also used by various search engines. It is obvious that they don’t use any library but the process remains the same to extract keywords
'''