import streamlit as st
import torch
from transformers import pipeline
from huggingface_hub import login
from github import Github  # Import PyGithub

HF_TOKEN = "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"
login(HF_TOKEN)

st.title("GitHub Chat App with LLaMA")

# Function to load the model with GPU support
@st.cache_resource  # Caches the model for efficiency
def load_model():
    # Ensure GPU is available
    if torch.cuda.is_available():
        device = 0  # Use the first GPU
        st.success("Using GPU for model inference.")
    else:
        device = -1  # Use CPU if no GPU is available
        st.warning("GPU not available. Using CPU for model inference.")
    
    return pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct",use_auth_token=HF_TOKEN, device=device)

# Load the model
model = load_model()

# GitHub credentials input
github_username = st.text_input("Enter your GitHub username")
github_pat = st.text_input("Enter your GitHub Personal Access Token", type="password")

# GitHub repository loader
repo_info = ""
if github_username and github_pat:
    try:
        # Initialize GitHub API client
        g = Github(github_pat)
        user = g.get_user(github_username)
        st.success(f"Connected to GitHub user: {user.login}")
        
        # Display repositories
        repos = [repo.name for repo in user.get_repos()]
        selected_repo = st.selectbox("Select a repository", repos)

        # Fetch repository details for selected repository
        if selected_repo:
            repo = user.get_repo(selected_repo)
            repo_info = f"Repository Name: {repo.name}\nDescription: {repo.description}\n" \
                        f"Stars: {repo.stargazers_count}\nForks: {repo.forks_count}\n" \
                        f"Language: {repo.language}\n"
            files = repo.get_contents("")
            file_names = ", ".join([file.name for file in files])
            repo_info += f"Files: {file_names}"

    except Exception as e:
        st.error(f"Failed to load GitHub repository data: {e}")

# Input question and generate answer using the model
if github_username and github_pat:
    question = st.text_input("Ask a question about your GitHub repository")
    if question and repo_info:
        try:
            # Combine the repository info and question as input for the model
            model_input = f"Here is information about the repository:\n{repo_info}\n\nQuestion: {question}"
            
            # Generate answer using GPU and specify max_new_tokens for response length
            answer = model(model_input, max_new_tokens=500, do_sample=True)
            st.write(answer[0]["generated_text"])
        except Exception as e:
            st.error(f"Error generating answer: {e}")
