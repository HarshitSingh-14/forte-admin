# Style-admin

This project is a Streamlit-based web application that processes PDF documents stored in an S3 bucket, splits the text into chunks, and uses RAG (Retrieval-Augmented Generation) to generate different document styles.
You can also access the hosted application on [Streamlit Cloud](https://llm-style-main.streamlit.app).

## Project Structure

```
style-admin/
├── .env
├── Dockerfile
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── admin.py
    ├── s3_utils.py
    ├── text_processing.py
    └── vector_store.py
```

## Functionality

The application allows users to:

1. **Select a Document Style**: Choose from Email Style, Normal Style, Report Style, or Feedback Style.
2. **List PDF Files**: List all PDF files in the specified S3 prefix (subfolder).
3. **Process PDFs**: Download and process the selected PDFs, split the text into chunks, and create a FAISS index.
4. **View Results**: Display the results and any errors encountered during processing.

## Usage

### Prerequisites

- Docker installed on your machine.
- AWS credentials with access to the specified S3 bucket.

### Setup

1. **Create and Configure `.env` File**: Add the necessary environment variables to the `.env` file.

```env
BUCKET_NAME=your-s3-bucket-name
AWS_ACCESS_KEY_ID=your-aws-access-key-id
AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key
AWS_REGION=your-aws-region
```

2. **Build and Run the Docker Container**: Use the following commands to build and run the Docker container.

```sh
docker build -t style-admin .
docker run -p 8083:8083 --env-file .env style-admin
```

3. **Access the Application**: Open a web browser and navigate to `http://localhost:8083`. You should see the Streamlit application running.
