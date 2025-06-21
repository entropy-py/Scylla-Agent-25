import requests
import weaviate
import warnings
from langchain_cohere import CohereEmbeddings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq  
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv() 
url = "https://raw.githubusercontent.com/hwchase17/chroma-langchain/88bd99222d46f763957c2873e48bbba2f3e4b36a/state_of_the_union.txt"
res = requests.get(url)
with open("state_of_the_union.txt", "w") as f:
    f.write(res.text)

loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
embeddings = CohereEmbeddings(
    cohere_api_key=os.environ["COHERE_API_KEY"],
    model="embed-english-v3.0"
)
vectorstore = FAISS.from_documents(docs, embeddings)

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.5,
    max_tokens=200,
    groq_api_key=os.environ["GROQ_API_KEY"],
)
retriever=vectorstore.as_retriever()

template = """.You are a helpful assistant answering questions about the State of the Union address based on the provided context.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser()
)

query = input("Enter question- ")
terminate='exit'
while(query!=terminate):
  resp = rag_chain.invoke(query)
  print("\nAnswer:")
  print(resp)
  query = input("Enter question- ")
