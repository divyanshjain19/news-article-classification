import streamlit as st
import joblib
from scipy.sparse import hstack
import numpy as np
import re

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


authors = joblib.load('authors.joblib')
ohe_year = joblib.load('ohe_year.joblib')
ohe_authors = joblib.load('ohe_authors.joblib')
vec_headline = joblib.load('vec_headline.joblib')
vec_desc = joblib.load('vec_desc.joblib')
model = joblib.load('model.joblib')
lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words('english'))

@st.cache
def clean_text(string):
    '''
    Removes all the special characters except A-Z and a-z
    Lowers the case
    Stripping the string off white spaces
    Lemmetizes the words
    Keeps only those words which are of length more than 1
    Removes stop words

    Input: raw string
    Output: cleaned string
    '''
    string = re.sub(' +', ' ', re.sub('[^A-Za-z ]+', ' ', string).lower()).strip()
    string = " ".join([lemmatizer.lemmatize(x) for x in string.split() if len(x) > 1 and x not in stops])
    return string

@st.cache
def clean_authors(author_name):
    '''
    Cleans the author name
    
    Input: author_name (raw)
    Output: author_name (cleaned)
    '''
    if author_name == 'None of the above':
        return "NA"
    init_author_name = author_name.lower().split(',')[0].split('and')[0].replace('_', ' ')\
    .replace('"', '').replace("'", '').replace('-', ' ').replace('.', ' ').replace('by ', '').strip()
    final_author_name = " ".join([s.capitalize() for s in init_author_name.split()]) 
    final_author_name = re.sub('\(.*\)', '', final_author_name).strip()
    if (len(final_author_name) <= 4) or \
       (re.search('([0-9]+)|(#)', final_author_name)) or \
       (not re.search('[a-z]', final_author_name)):
        return 'NA'
    return final_author_name

@st.cache
def predict(year, author, headline, desc):
    year_ohe = ohe_year.transform(np.array([str(year)]).reshape(-1, 1))
    author_ohe = ohe_authors.transform(np.array([clean_authors(author)]).reshape(-1, 1))
    headline_enc = vec_headline.transform([clean_text(headline)]) 
    desc_enc = vec_desc.transform([clean_text(desc)])
    datapoint = hstack((headline_enc, desc_enc, author_ohe, year_ohe))
    return model.predict(datapoint)
    
years = list(range(2022, 2000, -1))

st.title('News article classification')
st.header("What does this app do?")
st.write("Given some details (headline, description, year of publishing and author name(s)) about a news article, it predicts category of the article (Eg. Politics, Entertainment etc.)") 

st.header("Try it out")

with st.form("my_form"):
    # Every form must have a submit button.
    headline = st.text_input(
        'Article headline',
        placeholder="Enter news article headline")
    
    desc = st.text_area('Article description', placeholder="Enter news article description")
    
    year = st.selectbox(
        'Year of publishing',
        years)
    
    author = st.selectbox(
        'Author',
        authors,
        help='If author is not in the given list, please select "None of the above"')
    
    submitted = st.form_submit_button("Predict")

if submitted:
    if not headline:
        st.warning("Headline can't be empty")
        st.stop()
    if not desc:
        st.warning("Description can't be empty")
        st.stop()
    cat = predict(year, author, headline, desc)
    st.header("Predicted Category")
    st.write(cat[0])

st.markdown("<h5 style='text-align: right; color: white;'> Developed By </h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: right; color: white;'> Divyansh Jain </h5>", unsafe_allow_html=True)
