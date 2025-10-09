from typing import List, Tuple

# Third-party library imports
import chromadb
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- Main RAG Pipeline Class ---

class RAGPipeline:
    """
    A comprehensive class to handle the entire RAG pipeline, including:
    1. Loading and chunking documents.
    2. Embedding chunks and storing them in a vector database.
    3. Retrieving relevant chunks for a query.
    4. Reranking the retrieved chunks for better context.
    5. Generating a final answer using a generative model.
    """

    def __init__(
        self,
        embedding_model_name: str = "shibing624/text2vec-base-chinese",
        cross_encoder_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        generative_model_name: str = "gemini-2.5-flash",
        collection_name: str = "default_collection"
    ):
        """
        Initializes the pipeline by loading all necessary models and setting up the vector database.
        This ensures that models are loaded only once.
        """
        print(" Initializing RAG Pipeline...")

        # 1. Load models (this happens only once)
        print(f"   - Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
        
        print(f"   - Loading cross-encoder model: {cross_encoder_model_name}")
        self.cross_encoder = CrossEncoder(cross_encoder_model_name, trust_remote_code=True)

        # 2. Configure the generative model client
        print(f"   - Configuring generative model: {generative_model_name}")
        self.generative_model_name = generative_model_name
        try:
            self.generative_model = genai.Client()
        except Exception as e:
            raise ValueError("Failed to configure Google GenAI. Ensure GEMINI_API_KEY is set in your .env file.") from e

        # 3. Setup vector database
        print("   - Setting up ChromaDB...")
        self.chromadb_client = chromadb.EphemeralClient()
        self.chroma_collection = self.chromadb_client.get_or_create_collection(name=collection_name)
        
        print(" Pipeline initialized successfully!")

    def _split_into_chunks(self, doc_file: str) -> List[str]:
        """Splits a document into chunks based on double newlines."""
        with open(doc_file, 'r', encoding='utf-8') as file:
            content = file.read()
        return [chunk for chunk in content.split("\n\n") if chunk.strip()]

    def add_document(self, doc_file: str) -> None:
        """Processes and stores a document in the vector database."""
        print(f"\n Processing and embedding document: {doc_file}")
        chunks = self._split_into_chunks(doc_file)
        
        embeddings = self.embedding_model.encode(
            chunks, 
            normalize_embeddings=True,
            show_progress_bar=True
        ).tolist()
        
        ids = [str(i) for i in range(len(chunks))]
        
        self.chroma_collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
        print(f"   - Added {len(chunks)} chunks to the database.")

    def _retrieve(self, query: str, top_k: int) -> List[str]:
        """Retrieves the top_k most relevant chunks from the database."""
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True).tolist()
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results['documents'][0]

    def _rerank(self, query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
        """Reranks the retrieved chunks using a cross-encoder for higher relevance."""
        pairs = [(query, chunk) for chunk in retrieved_chunks]
        scores = self.cross_encoder.predict(pairs)
        
        scored_chunks = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, _ in scored_chunks][:top_k]

    def _generate(self, query: str, context_chunks: List[str]) -> str:
        """Generates an answer based on the query and reranked context chunks."""
        prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。
    
        用户问题: {query}

        相关片段:
        {"\n\n---\n\n".join(context_chunks)}

        请基于上述内容作答，不要编造信息。"""

        print("\n Generating final answer with the following prompt:")
        print("--------------------")
        print(prompt)
        print("--------------------")

        response = self.generative_model.models.generate_content(model=self.generative_model_name,
                                                                 contents=prompt,
                                                                 config=types.GenerateContentConfig(
                                                                     thinking_config=types.ThinkingConfig(thinking_budget=0)
                                                                 ))
        return response.text

    def ask(self, query: str, retrieve_top_k: int = 5, rerank_top_k: int = 3) -> str:
        """
        The main method to ask a question to the RAG pipeline.
        It orchestrates the retrieve, rerank, and generate steps.
        """
        print(f"\n Received query: '{query}'")
        
        # 1. Retrieve relevant documents
        print(f"   - Retrieving top {retrieve_top_k} chunks...")
        retrieved_chunks = self._retrieve(query, top_k=retrieve_top_k)
        
        # 2. Rerank the retrieved documents
        print(f"   - Reranking to find the best {rerank_top_k} chunks...")
        reranked_chunks = self._rerank(query, retrieved_chunks, top_k=rerank_top_k)
        
        # 3. Generate the final answer
        answer = self._generate(query, reranked_chunks)
        
        return answer

# --- Main Execution Block ---

if __name__ == "__main__":

    load_dotenv(find_dotenv())
    
    # 1. Initialize the RAG pipeline
    # This will download/load the models into memory once.
    rag_pipeline = RAGPipeline()
    
    # 2. Add a document to the knowledge base
    # (Make sure 'doc.md' exists in the same directory)
    rag_pipeline.add_document("./doc.md")
    
    # 3. Ask a question
    query = "哆啦A梦使用的3个秘密道具分别是什么？"
    final_answer = rag_pipeline.ask(query)
    
    # 4. Print the final result
    print("\n\n====================")
    print(" Final Answer:")
    print("====================")
    print(final_answer)