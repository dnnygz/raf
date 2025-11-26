import os
from dotenv import load_dotenv
import uuid
import base64
import io
from typing import List, Tuple, Dict, Any
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

from groq import Groq
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from unstructured.partition.pdf import partition_pdf
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from transformers import CLIPProcessor, CLIPModel

# Load a pre-trained CLIP model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# CLIP embeddings wrapper class with token limit handling
class CLIPEmbeddingsWrapper:
    def __init__(self, processor, model, max_tokens=77):
        self.processor = processor
        self.model = model
        self.max_tokens = max_tokens

    def _truncate_text(self, text: str) -> str:
        # Always use truncation to prevent token limit errors
        truncated_inputs = self.processor(
            text=[text], 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_tokens
        )
        
        # Decode back to text to see what was kept
        truncated_tokens = truncated_inputs['input_ids'][0]
        # Remove special tokens (CLS, SEP, PAD)
        truncated_tokens = truncated_tokens[truncated_tokens != self.processor.tokenizer.pad_token_id]
        truncated_text = self.processor.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return truncated_text

    def _chunk_long_text(self, text: str, overlap_tokens: int = 10) -> List[str]:
        """
        Split long text into overlapping chunks that fit within token limit.
        """
        words = text.split()
        chunks = []
        
        # Estimate words per chunk (rough approximation: 1 token ≈ 0.75 words)
        words_per_chunk = int((self.max_tokens - 2) * 0.75)  # -2 for special tokens
        overlap_words = int(overlap_tokens * 0.75)
        
        start_idx = 0
        while start_idx < len(words):
            end_idx = min(start_idx + words_per_chunk, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            # Verify this chunk fits, if not, reduce it
            chunk_text = self._truncate_text(chunk_text)
            chunks.append(chunk_text)
            
            # Move start position with overlap
            start_idx = end_idx - overlap_words
            if start_idx >= len(words):
                break
                
        return chunks

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of document texts, with mandatory truncation for all texts.
        """
        all_embeddings = []
        
        for text in texts:
            # Always truncate first to prevent token limit errors
            truncated_text = self._truncate_text(text)
            
            # Now embed the truncated text safely
            inputs = self.processor(
                text=[truncated_text], 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.max_tokens
            )
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            all_embeddings.append(text_features[0].detach().numpy().tolist())
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query text, truncating if necessary.
        """
        # Truncate query if too long
        truncated_text = self._truncate_text(text)
        inputs = self.processor(text=[truncated_text], return_tensors="pt", padding=True, truncation=True, max_length=self.max_tokens)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features[0].detach().numpy().tolist()


class MultimodalRAG:
    def __init__(self, output_path: str = "./content/"):
        self.output_path = output_path
        self.texts = []
        self.tables = []
        self.images = []

    def process_pdf(self, file_path: str) -> str:
        """
        Partitions the PDF, separates texts, tables, and images, and stores them.
        """
        if not os.path.exists(file_path):
            return f"Error: The file {file_path} does not exist."
        
        try:
            chunks = partition_pdf(
                filename=file_path,
                infer_table_structure=True,
                strategy="hi_res",
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=10000,
                combine_text_under_n_chars=2000,
                new_after_n_chars=6000,
            )

            for chunk in chunks:
                if "Table" in str(type(chunk)):
                    self.tables.append(chunk)
                elif "CompositeElement" in str(type(chunk)):
                    self.texts.append(chunk)
                    chunk_els = chunk.metadata.orig_elements
                    for el in chunk_els:
                        if "Image" in str(type(el)):
                            self.images.append(el)
            
            return f"""Document processed successfully!
            
            📄 Text: {len(self.texts)} chunks found
            🖼️ Tables: {len(self.tables)} tables found
            🖼️ Images: {len(self.images)} images found
            
            The system is ready to answer questions."""
        except Exception as e:
            return f"Error processing the document: {str(e)}"
        
### 2. Multimodal Description Chain (now only for describing images)
def get_multimodal_description_chain():
    """Creates and returns the chain for describing images using Groq."""
    messages_multimodal = [
        (
            "user",
            [
                {"type": "text", "text": "Describe the image in detail. For context, the image is part of a technical document explaining an architecture."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]
    prompt_multimodal = ChatPromptTemplate.from_messages(messages_multimodal)
    model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct") 
    chain_multimodal = prompt_multimodal | model | StrOutputParser()
    return chain_multimodal

### 3. Loading and Linking Data with the Critical Improvement
def build_retriever(texts, tables, images):
    """
    Creates and loads data into the MultiVectorRetriever using CLIP.
    """
    # Instantiate the new CLIP embeddings wrapper class
    clip_embeddings = CLIPEmbeddingsWrapper(processor, clip_model)

    # Initialize the vector store with the wrapper instance
    vectorstore = Chroma(
        collection_name="multi_modal_rag", 
        embedding_function=clip_embeddings  # Pass the instance of the class
    )
    
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
        search_kwargs={"k": 3},  # Limit to top 3 most relevant documents
    )

    # Process text and tables (similar to the original code)
    if texts:
        doc_ids_text = [str(uuid.uuid4()) for _ in texts]
        docs_to_add = [
            Document(page_content=text.text, metadata={id_key: doc_ids_text[i]}) for i, text in enumerate(texts)
        ]
        retriever.vectorstore.add_documents(docs_to_add)
        retriever.docstore.mset(list(zip(doc_ids_text, [t.text for t in texts])))
        
    if tables:
        doc_ids_tables = [str(uuid.uuid4()) for _ in tables]
        docs_to_add = [
            Document(page_content=table.text_as_html, metadata={id_key: doc_ids_tables[i]}) for i, table in enumerate(tables)
        ]
        retriever.vectorstore.add_documents(docs_to_add)
        retriever.docstore.mset(list(zip(doc_ids_tables, [t.text for t in tables])))
        
    # Process images directly with CLIP
    if images:
        doc_ids_images = [str(uuid.uuid4()) for _ in images]
        
        # Use the multimodal description chain to get text summaries
        multimodal_description_chain = get_multimodal_description_chain()
        image_summaries = multimodal_description_chain.batch(
            [img.metadata.image_base64 for img in images], {"max_concurrency": 3}
        )

        # Create documents from these text summaries and add them to the vector store.
        # retrieval happens with text-based embeddings. 
        summary_docs = []
        for i, summary in enumerate(image_summaries):
            caption = getattr(images[i].metadata, 'caption', '')
            text_context = getattr(images[i].metadata, 'text_as_html', '')
            
            enriched_summary = f"{summary}\n\nDocument description: {caption}\n{text_context}"
            summary_docs.append(
                Document(page_content=enriched_summary, metadata={id_key: doc_ids_images[i]})
            )
        
        # Add the text summaries of the images to the vector store
        retriever.vectorstore.add_documents(summary_docs)

        # Store the original image data in the docstore
        retriever.docstore.mset(list(zip(doc_ids_images, [img.metadata.image_base64 for img in images])))
    
    return retriever

### 4. RAG Chain Construction
def parse_docs(retriever):
    """Returns a function that separates the retrieved content into images and text."""
    def _parse(docs):
        """Separates the retrieved content into images and text."""
        b64 = []
        text = []
        id_key = retriever.id_key
        
        for doc in docs:
            # Get the doc_id from metadata
            doc_id = None
            if hasattr(doc, 'metadata') and doc.metadata:
                doc_id = doc.metadata.get(id_key)
            
            if doc_id:
                # Look up the original content in the docstore
                try:
                    results = retriever.docstore.mget([doc_id])
                    original_content = results[0] if results else None
                    
                    if original_content:
                        # Check if it's base64 (image) or text
                        try:
                            # Try to decode as base64 (validate=True ensures proper base64)
                            if isinstance(original_content, str):
                                base64.b64decode(original_content, validate=True)
                                b64.append(original_content)
                            elif isinstance(original_content, bytes):
                                # Already bytes, treat as image
                                b64.append(base64.b64encode(original_content).decode('utf-8'))
                            else:
                                # Other types, try to convert to string for text
                                text.append(str(original_content))
                        except Exception:
                            # Not base64, treat as text
                            if isinstance(original_content, str):
                                text.append(original_content)
                            else:
                                text.append(str(original_content))
                        continue
                except Exception:
                    pass
            
            # Fallback: use the document's page_content
            if hasattr(doc, 'page_content'):
                text.append(doc.page_content)
            elif hasattr(doc, 'text'):
                text.append(doc.text)
            elif isinstance(doc, str):
                text.append(doc)
        
        return {"images": b64, "texts": text}
    return _parse

def build_prompt(kwargs):
    """Builds the prompt for the LLM with context and user question."""
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    # Limit text context to ~15000 characters to stay within token limits
    MAX_CONTEXT_LENGTH = 15000
    MAX_IMAGES = 2  # Limit number of images to reduce token usage
    
    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            if len(context_text) + len(text_element) > MAX_CONTEXT_LENGTH:
                remaining = MAX_CONTEXT_LENGTH - len(context_text)
                if remaining > 100:  # Only add if there's space
                    context_text += text_element[:remaining] + "... [truncated]"
                break
            context_text += text_element

    prompt_template = f"""
    Answer the question based solely on the following context, which may include text, tables, and the image shown below.

    Context: {context_text}

    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    # Limit the number of images to reduce token usage
    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"][:MAX_IMAGES]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )

def get_rag_chain(retriever):
    """Creates and returns the complete RAG pipeline."""
    parse_docs_func = parse_docs(retriever)
    
    # Limit retrieval to top 3 most relevant documents to reduce token usage
    def limit_docs(docs):
        """Limit the number of retrieved documents."""
        if isinstance(docs, list):
            return docs[:3]
        return docs
    
    chain = (
        {
            "context": retriever | RunnableLambda(limit_docs) | RunnableLambda(parse_docs_func),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
        | StrOutputParser()
    )
    return chain

### 5. System Execution
if __name__ == "__main__":
    pdf_path = "./content/rag-challenge.pdf"
    
    rag_system = MultimodalRAG()
    rag_system.process_pdf(pdf_path)
    
    retriever = build_retriever(rag_system.texts, rag_system.tables, rag_system.images)
    if retriever:
        rag_chain = get_rag_chain(retriever)
        # Example questions
        response = rag_chain.invoke("What is this document about?")
        print(response)