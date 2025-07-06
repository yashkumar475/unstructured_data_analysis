import streamlit as st
from backend.file_processing import extract_text_from_pdf, extract_text_from_txt
from backend.qa_pipeline import process_text, build_vectorstore, answer_question, summarize_text, get_sentiment

def main():
    st.set_page_config(page_title='Ask your PDF/TXT', layout='centered')
    st.header('📄 Ask your PDF or TXT')

    uploaded_file = st.file_uploader('Upload your PDF or TXT file', type=['pdf', 'txt'])

    if uploaded_file is not None:
        if uploaded_file.type == 'application/pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == 'text/plain':
            text = extract_text_from_txt(uploaded_file)
        else:
            st.error('Unsupported file format')
            return

        # ------------------ TEXT STATISTICS -------------------
        word_count = len(text.split())
        reading_time = round(word_count / 200)
        st.subheader("📊 Document Statistics")
        st.write(f"**Word Count:** {word_count}")
        st.write(f"**Estimated Reading Time:** {reading_time} min")

        # ------------------ SENTIMENT ANALYSIS -------------------
        st.subheader("🩺 Sentiment Analysis")
        sentiment, polarity = get_sentiment(text)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Sentiment", value=sentiment)
        with col2:
            st.metric(label="Polarity Score", value=round(polarity, 3))

        # ------------------ GENERATE SUMMARY -------------------
        if st.button("🪄 Generate Summary"):
            with st.spinner("Generating summary using Groq LLM..."):
                summary = summarize_text(text)
            with st.container(border=True):
                st.subheader("📑 Summary")
                st.write(summary)

        # ------------------ Q&A SECTION -------------------
        st.subheader("❓ Ask a Question")
        chunks = process_text(text)
        knowledge_base = build_vectorstore(chunks)

        user_question = st.text_input('Ask a question about your document:')
        if user_question:
            with st.spinner("Fetching answer using Groq LLM..."):
                response = answer_question(knowledge_base, user_question)
            st.success(response)

if __name__ == '__main__':
    main()
