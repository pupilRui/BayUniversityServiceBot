import os
import openai
import sys
import requests
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['API_KEY']
goog_api_key    = os.environ['GOOG_API_KEY']


# We just discussed `Document Loading` and `Splitting`.


# In[ ]:


from langchain.document_loaders import PyPDFLoader


#############################################################
# 1. Load PDF
#
# References of different loading:
# - PDF
# - Youtube
# - URL
# - Notion DB
#############################################################

docs = []

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("data/pdf/2023Catalog.pdf")
docs.extend(loader.load())

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader


# In[ ]:


# ! pip install yt_dlp
# ! pip install pydub


# **Note**: This can take several minutes to complete.

# In[ ]:


def get_channel_videos(channel_id, api_key):
    url = f'https://www.googleapis.com/youtube/v3/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults=50'
    response = requests.get(url)
    data = response.json()
    videos = []

    for item in data['items']:
        if item['id']['kind'] == 'youtube#video':
            videos.append('https://www.youtube.com/watch?v=' + item['id']['videoId'])

    return videos


def search_videos(query, api_key):
    url = f'https://www.googleapis.com/youtube/v3/search?key={api_key}&q={query}&part=snippet,id&order=relevance&maxResults=50'
    response = requests.get(url)
    data = response.json()
    videos = []

    if 'items' in data:
        for item in data['items']:
            if item['id']['kind'] == 'youtube#video':
                videos.append('https://www.youtube.com/watch?v=' + item['id']['videoId'])

    return videos

channel_id = 'UCq476UNYNtbp-flsx-B6kLw'
videos1 = get_channel_videos(channel_id, goog_api_key)
search_query1 = 'SFBU DeepPiCar'
videos2 = search_videos(search_query1, goog_api_key)
search_query2 = 'SFBU'
videos3 = search_videos(search_query2, goog_api_key)

videos_special = ['https://www.youtube.com/watch?v=kuZNIvdwnMc', 'https://www.youtube.com/watch?v=1gJcCM5G32k', 'https://www.youtube.com/watch?v=hZE5fT7CVdo']

# Merge all the lists
all_videos = videos1 + videos2 + videos3 + videos_special

# Remove duplicates by converting the list to a set and back to a list
unique_videos = list(set(all_videos))


for url in unique_videos:
    save_dir="data/youtube/"
    print(f"Loading video: {url}")
    loader = GenericLoader(
        YoutubeAudioLoader([url],save_dir),
        OpenAIWhisperParser()
    )
    docs.extend(loader.load())

from langchain.document_loaders import WebBaseLoader

urls = ['https://www.sfbu.edu/admissions/student-health-insurance', 'https://www.sfbu.edu/about-us', 'https://www.sfbu.edu/admissions', 'https://www.sfbu.edu/academics', 'https://www.sfbu.edu/student-life', 'https://www.sfbu.edu/contact-us']

for url in urls:
    print(f"Loading url: {url}")
    loader = WebBaseLoader(url)
    docs.extend(loader.load())

# In[ ]:



#############################################################
# 2. Split the content to create chunks
#
# References
# - Document Splitting
#############################################################
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)


# In[ ]:


splits = text_splitter.split_documents(docs)


# In[ ]:


len(splits)


#############################################################
# 3. Create an index for each chunk by embeddings
# 
# Let's take our splits and embed them.
#############################################################

# In[ ]:


from langchain.embeddings.openai import OpenAIEmbeddings
# embedding = OpenAIEmbeddings()


# In[ ]:


# sentence1 = "i like dogs"
# sentence2 = "i like canines"
# sentence3 = "the weather is ugly outside"

sentence1 = "What are the application requirements for the MSCS program at SFBU?"
sentence2 = "Can you outline the core curriculum for the SFBU MSCS program?"
sentence3 = "What are the tuition and fees for the MSCS program at SFBU?"
sentence4 = "Are there any prerequisite courses or experience required for the MBA program at SFBU?"
sentence5 = "What specializations are available within the SFBU MBA program?"
sentence6 = "How do I contact the admissions office for the MBA program at SFBU?"
sentence7 = "What are the graduation requirements for the MSCS program at SFBU?"
sentence8 = "Can you provide information on the faculty for the MBA program at SFBU?"
sentence9 = "Are there part-time or online options for the MSCS or MBA programs at SFBU?"
sentence10 = "What career services are available for MSCS and MBA students at SFBU?"


# In[ ]:

embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)
embedding4 = embedding.embed_query(sentence4)
embedding5 = embedding.embed_query(sentence5)
embedding6 = embedding.embed_query(sentence6)
embedding7 = embedding.embed_query(sentence7)
embedding8 = embedding.embed_query(sentence8)
embedding9 = embedding.embed_query(sentence9)
embedding10 = embedding.embed_query(sentence10)


# In[ ]:


import numpy as np


# In[ ]:


# numpy.dot(vector_a, vector_b, out = None) 
# returns the dot product of vectors a and b.
np.dot(embedding1, embedding2)


# In[ ]:


np.dot(embedding1, embedding3)


# In[ ]:


np.dot(embedding2, embedding3)



#############################################################
# 4. Vectorstores
#############################################################


# In[ ]:


# ! pip install chromadb


# In[ ]:


from langchain.vectorstores import Chroma


# In[ ]:


persist_directory = 'docs/chroma/'


# In[ ]:


# remove old database files if any
if os.path.exists(persist_directory):
    import shutil
    shutil.rmtree(persist_directory)
    print("Removed old database files.")


# In[ ]:


vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)


# In[ ]:


print(vectordb._collection.count())



#############################################################
# 5. Similarity Search
#############################################################


# In[ ]:


question = "is there an email i can ask for help"


# In[ ]:


docs = vectordb.similarity_search(question,k=3)

print("Question: ", question)
print("Top 3 results: ")
print("==================================== answer 1\n")
print(docs[0].page_content)
print("==================================== answer 1 end\n")

print("==================================== answer 2\n")
print(docs[1].page_content)
print("==================================== answer 2 end\n")

print("==================================== answer 3\n")
print(docs[2].page_content)
print("==================================== answer 3 end\n")


# Let's save this so we can use it later!


# In[ ]:


vectordb.persist()



#############################################################
# 6. Edge Case - Failure modes
# 
# This seems great, and basic similarity 
# search will get you 80% of the way there 
# very easily. 
# 
# But there are some failure modes that can creep up. 
# 
# Here are some edge cases that can arise - we'll fix 
# them in the next class.
#############################################################


# In[ ]:


question = "what did they say about matlab?"


# In[ ]:


docs = vectordb.similarity_search(question,k=5)


print("Question: ", question)
print("Top 3 results: ")
print("==================================== answer 1\n")
print(docs[0].page_content)
print("==================================== answer 1 end\n")
print("==================================== answer 2\n")
print(docs[1].page_content)
print("==================================== answer 2 end\n")
print("==================================== answer 3\n")
print(docs[2].page_content)
print("==================================== answer 3 end\n")




#############################################################
# 6.2 Edge Case 2 - Failure modes: Specifity
#
# We can see a new failure mode.
# 
# The question below asks a question about 
# the third lecture, 
# but includes results from other lectures 
# as well.
#############################################################


# In[ ]:


question = "what did they say about regression \
  in the third lecture?"


# In[ ]:


docs = vectordb.similarity_search(question,k=5)


# In[ ]:


print("Question: ", question)
print("Top 3 results: ")
print("==================================== answer 1\n")
print(docs[0].page_content)
print("==================================== answer 1 end\n")
print("==================================== answer 2\n")
print(docs[1].page_content)
print("==================================== answer 2 end\n")
print("==================================== answer 3\n")
print(docs[2].page_content)
print("==================================== answer 3 end\n")