import streamlit as st
from src.helper import extract_text_from_pdf, ask_groq
from src.helper import fetch_linkedin_profile


st.set_page_config(page_title="Job Recommender", layout="wide")
st.title("AI Resume Analyzer")
st.markdown("Upload your resume and get job recommendations based on skills and expertise")

uploaded_file= st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from resume..."):
        resume_text= extract_text_from_pdf(uploaded_file)

    with st.spinner("Summarizing your resume..."):
        summary= ask_groq(f"Summarizing this resume highlighting skills, education: \n\n{resume_text}")


    with st.spinner("Finding skill gaps..."):
        gaps= ask_groq(f"Analyzing this resume and finding missing skills: \n\n{resume_text}")


    with st.spinner("Building roadmap..."):
        road_map= ask_groq(f"Based on this resume, suggesting future roadmap: \n\n{resume_text}")


    st.markdown("---")
    st.header("📑 Resume Summary")
    st.markdown(f"<div style='background-color: #000000; padding: 15px; border-radius: 10px; font-size:16px; color:white;'>{summary}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.header("🛠️ Skill Gaps & Missing Areas")
    st.markdown(f"<div style='background-color: #000000; padding: 15px; border-radius: 10px; font-size:16px; color:white;'>{gaps}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.header("🚀 Future Roadmap & Preparation Strategy")
    st.markdown(f"<div style='background-color: #000000; padding: 15px; border-radius: 10px; font-size:16px; color:white;'>{road_map}</div>", unsafe_allow_html=True)

    st.success("✅ Analysis Completed Successfully!")

        