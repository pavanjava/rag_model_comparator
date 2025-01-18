import streamlit as st
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.tools import QueryEngineTool, QueryPlanTool
from llama_index.core import get_response_synthesizer
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import ReActAgent
from qdrant_client import QdrantClient
from dotenv import load_dotenv, find_dotenv
from langfuse.llama_index import LlamaIndexInstrumentor

load_dotenv(find_dotenv())

instrumentor = LlamaIndexInstrumentor()
# Automatically trace all LlamaIndex operations
instrumentor.start()


def initialize_models():

    Settings.chunk_size = 512
    Settings.chunk_overlap = 20

    # Initialize OpenAI
    openai_llm = OpenAI(model="gpt-4")

    # Initialize Ollama
    ollama_llm = Ollama(model="llama3.2:latest", temperature=0.2, base_url="http://localhost:11434/")

    # initialize Anthropic
    anthropic_llm = Anthropic(model="claude-3-5-sonnet-20240620")

    return openai_llm, ollama_llm, anthropic_llm


def load_documents():
    march_data = SimpleDirectoryReader(input_files=['./data/10q/uber_10q_march_2022.pdf']).load_data(show_progress=True)
    june_data = SimpleDirectoryReader(input_files=['./data/10q/uber_10q_june_2022.pdf']).load_data(show_progress=True)
    sept_data = SimpleDirectoryReader(input_files=['./data/10q/uber_10q_sept_2022.pdf']).load_data(show_progress=True)
    return march_data, june_data, sept_data


def setup_vector_store():
    qdrant_store_client = QdrantClient(url="http://localhost:6333/", api_key="th3s3cr3tk3y")
    vector_store = QdrantVectorStore(collection_name="financial_data", client=qdrant_store_client)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    return qdrant_store_client, vector_store, storage_ctx


def create_indices(qdrant_store_client, vector_store, storage_ctx, march_data, june_data, sept_data):
    if not qdrant_store_client.collection_exists(collection_name="financial_data"):
        march_index = VectorStoreIndex.from_documents(documents=march_data, storage_context=storage_ctx)
        june_index = VectorStoreIndex.from_documents(documents=june_data, storage_context=storage_ctx)
        sept_index = VectorStoreIndex.from_documents(documents=sept_data, storage_context=storage_ctx)
    else:
        march_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        june_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        sept_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return march_index, june_index, sept_index


def setup_query_engines(march_index, june_index, sept_index, openai_llm):
    march_engine = march_index.as_query_engine(similarity_top_k=3, llm=openai_llm)
    june_engine = june_index.as_query_engine(similarity_top_k=3, llm=openai_llm)
    sept_engine = sept_index.as_query_engine(similarity_top_k=3, llm=openai_llm)
    return march_engine, june_engine, sept_engine


def create_query_tools(march_engine, june_engine, sept_engine):
    query_tools = [
        QueryEngineTool.from_defaults(
            query_engine=sept_engine,
            name="sept_2022",
            description="Provides information about Uber quarterly financials ending September 2022"
        ),
        QueryEngineTool.from_defaults(
            query_engine=june_engine,
            name="june_2022",
            description="Provides information about Uber quarterly financials ending June 2022"
        ),
        QueryEngineTool.from_defaults(
            query_engine=march_engine,
            name="march_2022",
            description="Provides information about Uber quarterly financials ending March 2022"
        )
    ]
    return query_tools


def setup_agents(query_tools, openai_llm, ollama_llm, anthropic_llm):
    response_synthesizer = get_response_synthesizer()
    query_plan_tool = QueryPlanTool.from_defaults(
        query_engine_tools=query_tools,
        response_synthesizer=response_synthesizer
    )

    openai_agent = OpenAIAgent.from_tools(
        [query_plan_tool],
        max_function_calls=10,
        llm=openai_llm,
        verbose=True,
    )

    ollama_agent = ReActAgent.from_tools(
        query_tools,
        llm=ollama_llm,
        verbose=True
    )

    anthropic_agent = ReActAgent.from_tools(
        query_tools,
        llm=anthropic_llm,
        verbose=True
    )

    return openai_agent, ollama_agent, anthropic_agent


def main():
    st.set_page_config(layout="wide")
    st.title("Agentic Financial RAG - Model Comparison")

    # Initialize session state
    if 'agents_initialized' not in st.session_state:
        with st.spinner("Initializing models and loading documents..."):
            openai_llm, ollama_llm, anthropic_llm = initialize_models()
            march_data, june_data, sept_data = load_documents()
            qdrant_store_client, vector_store, storage_ctx = setup_vector_store()
            march_index, june_index, sept_index = create_indices(
                qdrant_store_client, vector_store, storage_ctx,
                march_data, june_data, sept_data
            )
            march_engine, june_engine, sept_engine = setup_query_engines(
                march_index, june_index, sept_index, openai_llm
            )
            query_tools = create_query_tools(march_engine, june_engine, sept_engine)
            openai_agent, ollama_agent, anthropic_agent = setup_agents(query_tools, openai_llm, ollama_llm,
                                                                       anthropic_llm)

            st.session_state.openai_agent = openai_agent
            st.session_state.ollama_agent = ollama_agent
            st.session_state.anthropic_agent = ollama_agent
            st.session_state.agents_initialized = True

    # Query input
    # Analyze Uber revenue growth in March, June, and September
    query = st.text_input("Enter your query:", "")

    if st.button("Analyze"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("OpenAI Response (GPT-4o)")
            with st.spinner("Getting OpenAI response..."):
                openai_response = st.session_state.openai_agent.query(query)
                st.write(openai_response.response)

        with col2:
            st.subheader("Ollama Response (Qwen2.5)")
            with st.spinner("Getting Ollama response..."):
                ollama_response = st.session_state.ollama_agent.query(query)
                st.write(ollama_response.response)

        with col3:
            st.subheader("Anthropic Response (claude-3-5-sonnet)")
            with st.spinner("Getting Anthropic Claud Response..."):
                anthropic_response = st.session_state.anthropic_agent.query(query)
                st.write(anthropic_response.response)


if __name__ == "__main__":
    main()
