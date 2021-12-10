from rake_nltk import Rake
rake_nltk_var = Rake()
import pandas as pd

text = """ Profile 
Business-minded data scientist with a demonstrated ability to deliver valuable insights via data analytics and advanced data-driven methods,
With deep understanding of neural networks and the ability to explain technical topics,data, machine learning, and statistics.
Passionate about explaining data science to non-technical business audiences,
I have experience using stats and machine-learning to find useful insights in data.

Professional Experience
Analyzed and Predict Stock Price for next day:
Developed and trained Artificial Recurrent Neural network (RNN) on stock dataframe and tested it with loss matrix and accuracy to predict next day price.

Built Twitter and News Sentiment App: 
Create Algorithm with TensorFlow predict the polarity of textual data or sentiments like Positive, Neural, and negative.

Predict What Products Customers Likely Want
Boosted Machine learning to Predict what products customers likely want and then optimize a targeted campaign using advanced decision optimization in a Jupyter notebook. 

Analyzed Public Twitter Profile: 
Established Deeper personality insight on several dimensions, including traits , values , need and consumer preferences . Enhanced with chart plot.

Technical Skills 
Development Machine Learning: 
Classification, Regression, Clustering, Future Engineering, TensorFlow, Transfer Learning, Restful API, Terminal Bash

Programming Language:
Python (Scikit-Learn, NumPy, Pandas, TensorFlow, Matplotlib, SciPy), SQL, Microsoft, HTML, JavaScript, Dart, Excel, Flutter

Statics Methods:
Time Series, Regression Models, Hypothesis Testing.

Visualization:
Tableau, Oracle DVD.

Development Platform:
Web Browser, IOS & Android Apps .

Tools:
Anaconda, Jupyter Notebook, IDE Editor, Android Studio.

Selected Coursework:
Stochastic Gradient Decent, Liner Algebra, Theory, Probity and Statics, A/B Testing

Education 
•	Studying Master Degree in Data Scientist in University of Colorado Bolder
•	Faculty of Business Studies Accountant Diploma 2001-2004		CERTIFICATION 
	
Tensorflow Certified Developer: Googel.com

IBM Applied AI Specialization :

NLP Natural Language Processing with python :  Udemy
Building Ai Powered Chatbot : IBM
Browser Based Model with TensorFlow: deeplearning.io
Machine Learning with Python: Coursera.
Augmented Data Visualization
 
LANGUAGES
Arabic, English

ADDITIONAL INTEREST
Karate
Swimming
Travel 

"""

rake_nltk_var.extract_keywords_from_text(text)
keyword_extracted = rake_nltk_var.get_ranked_phrases()
#print(keyword_extracted)
#print(keyword_extracted[:7])
data = pd.DataFrame(keyword_extracted)
print(data.head())
print(data)



'''
Summary
The process of extracting keywords helps us identifying the importance of words in a text. This task can be also used for topic modelling. It is very useful to extract keywords for indexing the articles on the web so that people searching the keywords can get the best articles to read.

This technique is also used by various search engines. It is obvious that they don’t use any library but the process remains the same to extract keywords
'''