import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

## Streamlit App
st.set_page_config(page_title="LangChain: Summarize Text from YouTube or Website", page_icon="😊")
st.title("😊 LangChain: Summarize Text From YouTube or Website")
st.subheader('Summarize URL')

## Get the Groq API Key and url (YouTube or Website to be Summarized)
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

genric_url = st.text_input("URL", label_visibility="collapsed")

## Llama model using Groq API
llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Helper: Fetch Transcript

def fetch_youtube_transcript(url):
    try:
        # Extract video ID from YouTube URL
        if "v=" in url:
            video_id = url.split("v=")[-1].split("&")[0]
        elif "youtu.be" in url:
            video_id = url.split("/")[-1]
        else:
            raise ValueError("Invalid YouTube URL format.")

        # Fetch transcript using YouTubeTranscriptApi
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([t['text'] for t in transcript_data])

        # Return a Document object
        return [Document(page_content=text)]
    except Exception as e:
        st.error(f"Failed to fetch YouTube transcript: {e}")
        return []

# Button Action

if st.button("Summarize the Content from YouTube or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not genric_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(genric_url):
        st.error("Please enter a valid URL. It can may be a YouTube video url or website url")
    else:
        try:
            with st.spinner("Waiting..."):
                ## Load content
                if "youtube.com" in genric_url or "youtu.be" in genric_url:
                    docs = fetch_youtube_transcript(genric_url)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[genric_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                                          "Chrome/116.0.0.0 Safari/537.36"
                        })
                    docs = loader.load()

                ## Summarization
                if docs:
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)
                    st.success(output_summary)
                else:
                    st.error("No content found to summarize.")
        except Exception as e:
            st.exception(f"Exception: {e}")