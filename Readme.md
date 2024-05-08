# Langchain Chat with your PDFs

## Package we will use
- **strimlit**: to build our user interface instead of html or jinja tempalte engine
- **pypdf2**: to extract/read pdf
- **langchain**: to interact with our language model
- **python-dotenv**: to load our secrets form `.env` file
- **faiss-cpu**: our vector database store
- **openai**: to generate response form openai
- **huggingface_hub**: free alternative of openai

> ALWAYS USE THIS VARIABLE NAME FOR HUGGING FACE AND OPEN AI API KEY
> OTHERWISE LANGCHAIN WILL NOT RECOGNIZE IT
> 
> **OPENAI_API_KEY**
> **HUGGINGFACEHUB_API_TOKEN**


## Step 1: Read the pdf
We will use pypdf2 to extract the text from pdfs

## Step 2:  Embedding Text
We have 2 option to embed any text.
- Paid: OpenAI ada model
- Free: `https://instructor-embedding.github.io/` which is free and can run locally. But as it will run on cpu the output will not as good as openai

But if you can run the `https://instructor-embedding.github.io/` on a server, then it is far more better than openai ada-02 model

For the simplicity we are using openai ada