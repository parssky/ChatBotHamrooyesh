from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import os
import tempfile


def readPdf(file):
    loader = PyMuPDFLoader(file)
    data = loader.load()
    db.from_documents(data, hf, es_url="http://localhost:9200", index_name="test")
    st.session_state.pdf = True


def setup_embeddings():
    print(">> setup_embeddings")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    return HuggingFaceEmbeddings(model_name=model_name, cache_folder="./transformer")


hf = setup_embeddings()

db = ElasticsearchStore(embedding=hf, es_url="http://localhost:9200", index_name="test")

uploaded_file = st.file_uploader("choose a file: ", accept_multiple_files=False)


def getFlanLarg():
    model_id = "google/flan-t5-large"
    print(f">> getFlanLarg {model_id}")
    tokenizer = AutoTokenizer.from_pretrained("./google/")
    model = AutoModelForSeq2SeqLM.from_pretrained("./google/", cache_dir="./cache")
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        truncation=True,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def make_llm_chain():
    prompt_sample = """
    I am a helpful AI that answers questions.
    When I don't know the answer I say I don't know. 
    I know context: {context}
    when asked: {question}
    my response using only information in the context is: 
    """
    prompt = PromptTemplate(
        template=prompt_sample, input_variables=["context", "question"]
    )
    llm = getFlanLarg()
    return LLMChain(prompt=prompt, llm=llm)


# llm_chain = make_llm_chain()
if "llm" not in st.session_state:
    llm_chain = make_llm_chain()
    st.session_state.llm = llm_chain
else:
    llm_chain = st.session_state.llm

if "pdf" not in st.session_state:
    st.session_state.pdf = False
st.title("Welcome to chatbot")
st.write("This is a chatbot with your own pdf")

if uploaded_file and not st.session_state.pdf:
    with st.spinner("Reading pdf ..."):
        temp_file_path = tempfile.mktemp(suffix=".pdf")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        readPdf(temp_file_path)
        os.remove(temp_file_path)
        st.success("PDF saved successfully")

user_question = st.text_area("Please enter your question:")


def ask(question):
    similar_doc = db.similarity_search(question)
    doc_context = similar_doc[0].page_content
    llm_response = llm_chain.invoke({"context": doc_context, "question": question})
    return llm_response


if user_question:
    response = ask(user_question)
    st.write("Your answer is :")
    st.write(response["text"])
