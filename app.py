import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import pandas as pd
from streamlit_ace import st_ace
import time

# Set page configuration
st.set_page_config(page_title="SQLPro", page_icon="📊", layout="wide")

# Apply dark mode CSS
st.markdown("""
    <style>
    .main {background-color: #1e1e1e; color: #ffffff;}
    .stButton>button {border-radius: 8px; background-color: #1f77b4; color: white; font-weight: bold;}
    .stTextInput>input {border: 1px solid #1f77b4; border-radius: 5px;}
    .sidebar .sidebar-content {background-color: #2e2e2e;}
    </style>
""", unsafe_allow_html=True)

# Title and header
st.title("SQLPro 📊")
st.markdown("Query your databases with ease using natural language or raw SQL!")

# Sidebar for database selection
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"
radio_options = ["Use Default SQLite Database - StudentDB", "Connect to your own MySQL Database"]
selected_option = st.sidebar.radio("Choose a Database", options=radio_options, label_visibility="visible")

# Initialize MySQL variables
mysql_host, mysql_user, mysql_password, mysql_db = None, None, None, None
if radio_options.index(selected_option) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host", value="localhost")
    mysql_user = st.sidebar.text_input("MySQL User", value="root")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database Name")
else:
    db_uri = LOCALDB

# GROQ API key input
groq_api_key = st.sidebar.text_input("GROQ API Key", type="password", placeholder="Enter your GROQ API Key")

# Input validation
if not db_uri:
    st.error("Please provide a valid database URI.")
    st.stop()
if not groq_api_key:
    st.error("Please provide a GROQ API Key.")
    st.stop()

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it", streaming=True)

# Database configuration
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    try:
        if db_uri == LOCALDB:
            db_file_path = (Path(__file__).parent / "student.db").absolute()
            creator = lambda: sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)
            return SQLDatabase(create_engine("sqlite:///", creator=creator))
        elif db_uri == MYSQL:
            if not (mysql_host and mysql_user and mysql_password and mysql_db):
                st.error("Please provide all MySQL connection details.")
                st.stop()
            return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        st.stop()

# Configure database
db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)

# Initialize LangChain agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# Initialize session state
if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Ask a question about your database or write a SQL query."}]
if "query_history" not in st.session_state:
    st.session_state["query_history"] = []

# Display messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Query mode selection
query_mode = st.radio("Query Mode", ["Natural Language", "Raw SQL"], horizontal=True)

# Query input
if query_mode == "Natural Language":
    user_query = st.chat_input("Ask a question about your database! 📈")
else:
    user_query = st_ace(
        language="sql",
        placeholder="Enter your SQL query (e.g., SELECT * FROM students WHERE grade > 80)",
        height=150,
        theme="monokai"
    )

# Process query and display results directly
if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        with st.spinner("Running your query..."):
            streamlit_callback = StreamlitCallbackHandler(st.container())
            try:
                start_time = time.time()
                response = agent.run(user_query, callbacks=[streamlit_callback])
                execution_time = time.time() - start_time
                st.session_state["messages"].append({"role": "assistant", "content": response})
                st.session_state["query_time"] = f"{execution_time:.2f} seconds"
                
                # Store query and response in history (limit to last 10)
                st.session_state["query_history"].append({"query": user_query, "response": response})
                st.session_state["query_history"] = st.session_state["query_history"][-10:]
                
                # Display results
                if isinstance(response, str):
                    try:
                        response_df = pd.read_json(response)
                        st.dataframe(response_df, use_container_width=True)
                        st.download_button(
                            label="Download Results as CSV",
                            data=response_df.to_csv(index=False),
                            file_name="query_results.csv",
                            mime="text/csv"
                        )
                    except:
                        st.success(response)
                else:
                    st.dataframe(response, use_container_width=True)
                    st.download_button(
                        label="Download Results as CSV",
                        data=response.to_csv(index=False),
                        file_name="query_results.csv",
                        mime="text/csv"
                    )
                
                # Query execution feedback
                st.write(f"Query completed in {st.session_state['query_time']}")
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
                st.session_state["query_history"].append({"query": user_query, "response": f"Error: {str(e)}"})
                st.session_state["query_history"] = st.session_state["query_history"][-10:]

# Sidebar query history
with st.sidebar.expander("Query History (Last 10)"):
    if st.session_state["query_history"]:
        for i, entry in enumerate(st.session_state["query_history"][::-1], 1):
            st.write(f"**Query {i}:** {entry['query']}")
            st.write(f"**Response:** {entry['response']}")
            st.markdown("---")
    else:
        st.write("No queries executed yet.")

# Help section
st.sidebar.markdown("---")
st.sidebar.subheader("Help")
st.sidebar.info("Use natural language (e.g., 'Show all students with grade > 80') or write raw SQL queries.")