import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Video Content Search", layout="wide")

@st.cache_data(show_spinner=False)
def load_embeddings(path="embeddings.joblib"):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": text_list},
    )
    return r.json()["embeddings"]

def search_in_videos(query, df, top_k=3):
    q_emb = create_embedding([query])[0]
    sims = cosine_similarity(np.vstack(df["embedding"]), [q_emb]).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    subset = df.loc[idxs]
    results = []
    for rank, (_, row) in enumerate(subset.iterrows(), 1):
        results.append({
            "Most Relevant": "Yes" if rank == 1 else "No",
            "Title": row["title"],
            "Start Time": f"{int(row['start'] // 60):02d}:{int(row['start'] % 60):02d}",
            "End Time": f"{int(row['end'] // 60):02d}:{int(row['end'] % 60):02d}",
            "Text": row["text"],
            "Raw_Start": row["start"],
            "Raw_End": row["end"]
        })
    return results

def call_llama_for_response(query, results):
    # Compose prompt from results
    context_chunks = [
        f'- {r["Title"]} ({r["Start Time"]}-{r["End Time"]}): {r["Text"]}'
        for r in results
    ]
    prompt = (
        "Here are video subtitle chunks containing video title, start time in seconds, end time in seconds, the text at that time:\n"
        + "\n".join(context_chunks)
        + "\n---------------------------------\n"
        f'"{query}"\n'
        "User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course"
    )
    # Send prompt to LLaMA API
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        if r.status_code == 200 and "response" in r.json():
            return r.json()["response"]
        else:
            return "Could not get a response from the AI model."
    except Exception as e:
        return f"Error: {e}"

df_embeddings = load_embeddings("embeddings.joblib")

st.title("🔍 Video Content Search")

query = st.text_input("Enter search query:", placeholder="e.g. CSS exercise or JS functions")

if query:
    with st.spinner("Searching videos..."):
        results = search_in_videos(query, df_embeddings, top_k=3)
    if results:
        # 1. Highlight the MOST MATCHED video
        most_matched = results[0]
        st.markdown("### 🎯 Most Relevant Video")
        st.markdown(
            f"**{most_matched['Title']}** &nbsp;&nbsp; "
            f"⏰ {most_matched['Start Time']} to {most_matched['End Time']}\n\n"
            f"> {most_matched['Text']}"
        )

        # 2. Show full results table
        st.markdown("### Top 3 Results")
        df = pd.DataFrame(results).drop(columns=['Raw_Start', 'Raw_End'])
        st.dataframe(df, use_container_width=True)

        # 3. Human answer from LLaMA
        st.markdown("### 🤖 Answer from AI Model")
        with st.spinner("Generating AI response..."):
            ai_answer = call_llama_for_response(query, results)
        st.info(ai_answer)
    else:
        st.warning("No results found. Try a different query.")
else:
    st.info("Please enter a query to search through video content.")

