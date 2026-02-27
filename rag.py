import streamlit as st
from main import processurl, generate

st.title("Multi-Source Retrieval-Augmented Generation (RAG) System")

# Maintain state
if "processed" not in st.session_state:
    st.session_state.processed = False

# Sidebar Inputs
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

process_url_button = st.sidebar.button("Process URLs")

if process_url_button:
    urls = [url for url in (url1, url2, url3) if url.strip() != ""]

    if len(urls) == 0:
        st.warning("You must provide at least one URL.")
    else:
        try:
            processurl(urls)
            st.session_state.processed = True
            st.success("URLs processed successfully ✅")
        except Exception as e:
            st.error(f"Error while processing URLs: {e}")

st.divider()

# Question Section
query = st.text_input("Enter your question")
submit_question = st.button("Submit Question")

if submit_question:
    if not st.session_state.processed:
        st.warning("Please process URLs first.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            answer, sources = generate(query)

            st.header("Answer:")
            st.write(answer)

            if sources:
                st.subheader("Sources:")
                for source in sources:
                    st.write(source)

        except Exception as e:
            st.error(f"Error: {e}")
