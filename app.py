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

st.set_page_config(page_title="Chat with your SQL Databases!", page_icon="📊")
st.title("Chat with your SQL Databases! 📊")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_options = ["Use Default SQLite Database - StudentDB", "Connect to your own MySQL Database"]
selected_option = st.sidebar.radio(label="Choose a Database to Chat With", options=radio_options) 

if radio_options.index(selected_option) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host", value="localhost")
    mysql_user = st.sidebar.text_input("MySQL User", value="root")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database Name")
else:
    db_uri = LOCALDB

groq_api_key = st.sidebar.text_input("GROQ API Key", type="password", placeholder="Enter your GROQ API Key")

if not db_uri:
    st.error("Please provide a valid database URI.")

if not groq_api_key:
    st.error("Please set the GROQ_API_KEY environment variable in your .env file.")
    st.stop()

llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it", streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        db_file_path = (Path(__file__).parent / "student.db").absolute()
        print("DB File Path:", db_file_path)
        creator = lambda: sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)


toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm =llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
    st.session_state["messages"] = [{"role" : "assistant", "content": "Hello! How can I assist you with your SQL database today?"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input("Ask a question about your database! 📈")
if user_query:
    st.session_state["messages"].append({"role" : "user", "content" : user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state["messages"].append({"role" : "assistant", "content" : response})
        st.write(response)