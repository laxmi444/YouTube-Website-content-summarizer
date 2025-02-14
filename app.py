import validators
import streamlit as st 
from langchain_groq import ChatGroq # ChatGroq, an integration that allows using Groq’s AI models (such as Llama3) within LangChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain # this loads a prebuilt summarization pipeline from LangChain, which automates summarization workflows
from langchain_community.document_loaders import UnstructuredURLLoader # this is used to extract text content from URLs, even if the structure is unorganized
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document # this defines a standardized document format in LangChain, useful for processing and managing text
import trafilatura # a web scraping library optimized for extracting readable content from webpages while avoiding unnecessary elements like ads

# streamlit app
st.set_page_config(page_title="Summarize text from Youtube or Websites")
st.title("📜🤖 Smarter Summaries for YouTube & the Web! 🌟")
st.subheader("Summarize URL")
#st.secrets["GROQ_API_KEY"]

# get groq api key and url(YT or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Enter Groq Api key", value="", type="password")

# URL input
generic_url = st.text_input("URL", label_visibility="collapsed")



# prompt template
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def get_youtube_transcript(url):
    """Extract transcript from YouTube video"""
    try:
        # Extract video ID from URL
        if "v=" in url:     # checks if the URL contains "v=" (e.g., https://www.youtube.com/watch?v=XYZ123)                                
            video_id = url.split("v=")[1].split("&")[0]  # url.split("v=")[1] → Splits the URL at "v=" and takes the part after it ("XYZ123&something"). Splits again at "&" (if present) to get only the video ID ("XYZ123")
        elif "youtu.be" in url:
            video_id = url.split("youtu.be/")[1]
        else:
            raise ValueError("Invalid YouTube URL")
            
        # Get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript_list])
    except Exception as e:
        raise Exception(f"Failed to get YouTube transcript: {str(e)}")

if st.button("Summarize"):
    # validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the required information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or website")
    else:
        try:
            # Initialize LLM
            llm = ChatGroq(model="deepseek-r1-distill-llama-70b", groq_api_key=groq_api_key)
            with st.spinner("Loading content..."):
                # Handle YouTube videos and websites differently
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    # Get YouTube transcript
                    content = get_youtube_transcript(generic_url)
                    docs = [Document(page_content=content)]
                else:
                    # Try trafilatura first for website content
                    downloaded = trafilatura.fetch_url(generic_url)
                    content = trafilatura.extract(downloaded)
                    
                    if content:
                        docs = [Document(page_content=content)]
                    else:
                        # Fallback to UnstructuredURLLoader
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                            }
                        )
                        docs = loader.load()

                if not docs:
                    st.error("No content could be extracted from the URL. Please check if the URL is accessible.")
                    st.stop()

                with st.spinner("Generating summary..."):
                    # initialize chain for summarization
                    chain = load_summarize_chain(
                        llm,
                        chain_type="stuff",
                        prompt=prompt
                    )
                    
                    output_summary = chain.run(docs)
                    st.success(output_summary)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
