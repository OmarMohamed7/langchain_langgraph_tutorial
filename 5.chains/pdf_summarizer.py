from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama

load_dotenv()

# Initialize LLM
llm = ChatOllama(model="phi4-mini:latest")


# Step 1: Load PDF
def load_pdf(file_path: str):
    """Load PDF document"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


# Step 2: Tokenize/Split the document
def tokenize_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks (tokenization)"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# Step 3: Create summarization chain
def create_summarization_chain():
    """Create a chain to summarize text"""
    prompt = PromptTemplate(
        template="""Summarize the following text in a clear and concise manner. 
        Focus on the main points and key information.
        
        Text:
        {text}
        
        Summary:""",
        input_variables=["text"],
    )

    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain


# Main function to process PDF
def process_pdf(file_path: str):
    """Complete pipeline: Load PDF -> Tokenize -> Summarize"""

    print(f"Loading PDF: {file_path}")
    # Step 1: Load PDF
    documents = load_pdf(file_path)
    print(f"Loaded {len(documents)} pages")

    # Step 2: Tokenize
    print("Tokenizing document...")
    chunks = tokenize_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Step 3: Summarize each chunk
    print("Generating summaries...")
    summarization_chain = create_summarization_chain()

    summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Summarizing chunk {i}/{len(chunks)}...")
        summary = summarization_chain.invoke({"text": chunk.page_content})
        summaries.append(summary)

    # Step 4: Create final summary of all chunk summaries
    print("Creating final summary...")
    final_prompt = PromptTemplate(
        template="""Create a comprehensive summary by combining the following summaries.
        Provide a well-structured summary that covers all the main points.
        
        Summaries:
        {summaries}
        
        Comprehensive Summary:""",
        input_variables=["summaries"],
    )

    parser = StrOutputParser()
    final_chain = final_prompt | llm | parser

    combined_summaries = "\n\n".join(
        [f"Summary {i+1}: {s}" for i, s in enumerate(summaries)]
    )
    final_summary = final_chain.invoke({"summaries": combined_summaries})

    return final_summary


if __name__ == "__main__":
    # Example usage
    pdf_path = input("Enter the path to your PDF file: ").strip()

    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found!")
    else:
        try:
            summary = process_pdf(pdf_path)
            print("\n" + "=" * 50)
            print("FINAL SUMMARY")
            print("=" * 50)
            print(summary)
        except Exception as e:
            print(f"Error processing PDF: {e}")
