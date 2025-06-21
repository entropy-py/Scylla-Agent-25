import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq  
from langchain.docstore.document import Document
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
embeddings = CohereEmbeddings(
    cohere_api_key=os.environ["COHERE_API_KEY"],
    model="embed-english-v3.0"
)  
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.5,
    max_tokens=1024,
    groq_api_key=os.environ["GROQ_API_KEY"],
)
# Function to split text into chunks with metadata of the chunk chronological index
def split_text_to_chunks_with_indices(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(Document(page_content=chunk, metadata={"index": len(chunks), "text": text}))
        start += chunk_size - chunk_overlap
    return chunks


# Function to retrieve a chunk from the vectorstore based on its index in the metadata
def get_chunk_by_index(vectorstore, target_index: int) -> Document:
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
    for doc in all_docs:
        if doc.metadata.get('index') == target_index:
            return doc
    return None


# Function that retrieves from the vectorstore based on semantic similarity and pads each retrieved chunk with its neighboring chunks
def retrieve_with_context_overlap(vectorstore, retriever, query: str, num_neighbors: int = 1, chunk_size: int = 200,
                                  chunk_overlap: int = 20) -> List[str]:
    relevant_chunks = retriever.invoke(query)
    result_sequences = []

    for chunk in relevant_chunks:
        current_index = chunk.metadata.get('index')
        if current_index is None:
            continue

        # Determine the range of chunks to retrieve
        start_index = max(0, current_index - num_neighbors)
        end_index = current_index + num_neighbors + 1

        # Retrieve all chunks in the range
        neighbor_chunks = []
        for i in range(start_index, end_index):
            neighbor_chunk = get_chunk_by_index(vectorstore, i)
            if neighbor_chunk:
                neighbor_chunks.append(neighbor_chunk)

        # Sort chunks by their index to ensure correct order
        neighbor_chunks.sort(key=lambda x: x.metadata.get('index', 0))

        # Concatenate chunks, accounting for overlap
        concatenated_text = neighbor_chunks[0].page_content
        for i in range(1, len(neighbor_chunks)):
            current_chunk = neighbor_chunks[i].page_content
            overlap_start = max(0, len(concatenated_text) - chunk_overlap)
            concatenated_text = concatenated_text[:overlap_start] + current_chunk

        result_sequences.append(concatenated_text)

    return result_sequences


# Main class that encapsulates the RAG method
class RAGMethod:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs = self._prepare_docs()
        self.vectorstore, self.retriever = self._prepare_retriever()

    def _prepare_docs(self) -> List[Document]:
         loader = PyPDFLoader("AI research paper.pdf") 
         pages = loader.load()
         content = "\n".join([page.page_content for page in pages])

         return split_text_to_chunks_with_indices(content, self.chunk_size, self.chunk_overlap)

    def _prepare_retriever(self):
        vectorstore = FAISS.from_documents(self.docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        return vectorstore, retriever

    def run(self, query:str, num_neighbors:int):
        baseline_chunks = self.retriever.invoke(query)
        enriched_chunks = retrieve_with_context_overlap(self.vectorstore, self.retriever, query, num_neighbors,
                                                        self.chunk_size, self.chunk_overlap)
        return baseline_chunks[0].page_content, enriched_chunks[0]
    def generate_answer(self, query: str, enriched_chunks: str) -> str:
        template="""
        Use the following context to answer the question.
        Context: {enriched_chunks}
        Question: {query}
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        prompt = prompt.format(enriched_chunks=enriched_chunks, query=query)
        response = llm.invoke(prompt)
        return response.content

# Main execution
if __name__ == "__main__":
    chunk_size = 500
    chunk_overlap = 50
    num_neighbors = 5
    rag_method = RAGMethod(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    while True:
        query = input("Enter your query: ")
        if query.lower() in {"exit", "terminate"}:
            print("Goodbye!")
            break
        baseline_chunks, enriched_chunks = rag_method.run(query, num_neighbors=num_neighbors)
        if isinstance(enriched_chunks, list):
         enriched_chunks = "\n".join(enriched_chunks)
        print("\nGenerated Answer from LLM:")
        print(rag_method.generate_answer(query, enriched_chunks))
        