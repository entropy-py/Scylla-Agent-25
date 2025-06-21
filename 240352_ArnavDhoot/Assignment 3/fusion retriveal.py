import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np
from langchain_groq import ChatGroq  
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.5,
    max_tokens=200,
    groq_api_key=os.environ["GROQ_API_KEY"],
)
embeddings = CohereEmbeddings(
    cohere_api_key=os.environ["COHERE_API_KEY"],
    model="embed-english-v3.0"
)  
# Function to encode the PDF to a vector store and return split documents
def encode_pdf_and_get_split_documents(chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader("AI research paper.pdf") 
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents) 
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore,texts

# Function to create BM25 index for keyword retrieval
def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)


# Function for fusion retrieval combining keyword-based (BM25) and vector-based search
def fusion_retrieval(vectorstore, bm25, query: str, k: int, alpha: float) -> List[Document]:
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
    bm25_scores = bm25.get_scores(query.split())
    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))

    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
    sorted_indices = np.argsort(combined_scores)[::-1]

    return [all_docs[i] for i in sorted_indices[:k]]


class FusionRetrievalRAG:
    def __init__(self,chunk_size: int , chunk_overlap: int):
        self.vectorstore, self.texts = encode_pdf_and_get_split_documents(chunk_size, chunk_overlap)
        self.bm25 = create_bm25_index(self.texts)

    def run(self, query: str, k: int , alpha:float):
        top_docs = fusion_retrieval(self.vectorstore, self.bm25, query, k, alpha)
        docs_content = [doc.page_content for doc in top_docs]
        return (docs_content)
    
    def generate_answer(self, query: str,doc_content: str) -> str:
        template="""
        Use the following context to answer the question.
        Context: {doc_content}
        Question: {query}
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        prompt = prompt.format(doc_content=doc_content, query=query)
        response = llm.invoke(prompt)
        return response.content

if __name__ == "__main__":
    chunk_size = 500
    chunk_overlap = 50
    k=5
    alpha=0.5
    retriever = FusionRetrievalRAG(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    while True:
        query = input("Enter your query: ")
        if query.lower() in {"exit", "terminate"}:
            print("Goodbye!")
            break
        docs_content=retriever.run(query=query,k=k,alpha=alpha)
        if isinstance(docs_content, list):
         doc_content = "\n".join(docs_content)
        print("\nGenerated Answer from LLM:")
        print(retriever.generate_answer(query,doc_content))
