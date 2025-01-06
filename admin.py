import boto3
import streamlit as st
import os
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

## s3_client
s3_client = boto3.client("s3", region_name="us-east-1")
BUCKET_NAME = os.getenv("BUCKET_NAME")
if not BUCKET_NAME:
    raise ValueError("BUCKET_NAME environment variable is not set")

# Add function to create S3 bucket if it doesn't exist
def ensure_bucket_exists():
    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=BUCKET_NAME)
    except s3_client.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            # Bucket doesn't exist, create it
            try:
                s3_client.create_bucket(Bucket=BUCKET_NAME)
                st.success(f"Created new S3 bucket: {BUCKET_NAME}")
            except Exception as create_error:
                st.error(f"Failed to create S3 bucket: {str(create_error)}")
                return False
        else:
            st.error(f"Error checking S3 bucket: {str(e)}")
            return False
    return True

## Bedrock
from langchain_community.embeddings import BedrockEmbeddings

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

## import FAISS
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def get_unique_id():
    return str(uuid.uuid4())

## Split the pages / text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

## create vector store
def create_vector_store(style_key, documents):
    """
    Create and store a FAISS index for the given style_key (e.g., 'mail', 'normal', 'report', 'feedback').
    The index files will be saved with style_key suffix in /tmp and then uploaded to S3.
    """
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    folder_path = "/tmp/"
    
    # Decide on file names based on style_key
    faiss_index_name = f"my_faiss_{style_key}"
    faiss_index_file = faiss_index_name + ".faiss"
    faiss_pkl_file = faiss_index_name + ".pkl"
    
    # Save locally
    vectorstore_faiss.save_local(index_name=faiss_index_name, folder_path=folder_path)

    # Upload to S3
    try:
        s3_client.upload_file(
            Filename=os.path.join(folder_path, faiss_index_file), 
            Bucket=BUCKET_NAME,
            Key=faiss_index_file
        )
        s3_client.upload_file(
            Filename=os.path.join(folder_path, faiss_pkl_file), 
            Bucket=BUCKET_NAME,
            Key=faiss_pkl_file
        )
        return True
    except Exception as e:
        st.error(f"Error uploading to S3: {str(e)}")
        return False

def list_pdfs_in_s3(prefix):
    """
    List all PDF files in the specified prefix (subfolder) in S3.
    """
    try:
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=prefix
        )
        pdf_files = [obj['Key'] for obj in response.get('Contents', []) 
                    if obj['Key'].lower().endswith('.pdf')]
        return pdf_files
    except Exception as e:
        st.error(f"Error listing PDFs from S3 for prefix '{prefix}': {str(e)}")
        return []

def download_and_process_pdf(s3_key):
    """
    Download PDF from S3 and process it using PyPDFLoader.
    """
    try:
        local_file = f"/tmp/{os.path.basename(s3_key)}"
        s3_client.download_file(BUCKET_NAME, s3_key, local_file)
        
        loader = PyPDFLoader(local_file)
        pages = loader.load_and_split()
        return pages
    except Exception as e:
        st.error(f"Error processing {s3_key}: {str(e)}")
        return None

def main():
    st.title("Admin - Process PDFs into FAISS Index by Style")
    st.write("Process PDFs from specific S3 subfolders to create style-specific embeddings.")
    
    # Add bucket creation check at the start
    if not ensure_bucket_exists():
        st.error("Failed to ensure S3 bucket exists. Please check your AWS credentials and permissions.")
        return

    # Let user choose which style to process
    style_choice = st.radio(
        "Select which style you want to process:",
        ("Email Style", "Normal Style", "Report Style", "Feedback Style")
    )
    
    # Map the style_choice to the S3 prefix
    prefix_map = {
        "Email Style": "mail/",
        "Normal Style": "normal/",
        "Report Style": "report/",
        "Feedback Style": "feedback/"
    }
    
    # Also map the style_choice to a unique key used for saving FAISS indexes
    style_key_map = {
        "Email Style": "mail",
        "Normal Style": "normal",
        "Report Style": "report",
        "Feedback Style": "feedback"
    }
    
    selected_prefix = prefix_map[style_choice]
    style_key = style_key_map[style_choice]

    st.write(f"You selected: {style_choice} (S3 prefix: {selected_prefix})")
    pdf_files = list_pdfs_in_s3(selected_prefix)
    
    if not pdf_files:
        st.warning(f"No PDF files found in S3 bucket under prefix: {selected_prefix}")
        return

    st.write(f"Found {len(pdf_files)} PDF files in S3 for this style.")
    
    if st.button(f"Process PDFs for {style_choice}"):
        all_pages = []
        
        # Process each PDF
        for pdf_file in pdf_files:
            st.write(f"Processing {pdf_file}...")
            pages = download_and_process_pdf(pdf_file)
            if pages:
                all_pages.extend(pages)
                st.write(f"Added {len(pages)} pages from {pdf_file}")

        if all_pages:
            # Split Text
            splitted_docs = split_text(all_pages, chunk_size=1000, chunk_overlap=200)
            st.write(f"Total chunks created: {len(splitted_docs)}")

            # Show sample chunks
            if len(splitted_docs) > 0:
                with st.expander("View sample chunks"):
                    st.write(splitted_docs[0])
                    if len(splitted_docs) > 1:
                        st.write(splitted_docs[1])

            # Create the Vector Store
            st.write(f"Creating the Vector Store for {style_choice}...")
            result = create_vector_store(style_key, splitted_docs)

            if result:
                st.success(f"PDFs processed and FAISS index created successfully for {style_choice}!")
            else:
                st.error("Error creating vector store. Please check logs.")

if __name__ == "__main__":
    main()
