import fitz
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from apify_client import ApifyClient

load_dotenv()

GROQ_API_KEY= os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"]= GROQ_API_KEY

client= ChatGroq(api_key= GROQ_API_KEY,model= "openai/gpt-oss-120b", max_tokens=500)
apify_client= ApifyClient(os.getenv("APIFY_API_TOKEN"))


def extract_text_from_pdf(uploaded_file):
    doc= fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text=""
    for page in doc:
        text+=page.get_text()
    return text


def ask_groq(prompt):
    
    messages = [
        ("system", "You are a helpful assistant."),
        ("user", prompt)
    ]
    response = client.invoke(messages)
    return response.content


def fetch_linkedin_profile(search_query, location="india", rows=60):
    run_input={
        'title': search_query,
        'location': location,
        'rows': rows,
        'proxy': {
            'useApifyProxy': True,
            'apifyProxyGroups': ['Residential']
        }
    }
    run= apify_client.actor("BHzefUZlZRKWxkTck").call(run_input=run_input)
    jobs= list(apify_client.dataset(run['defaultDatasetId']).iterate_items())
    return jobs


