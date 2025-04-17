import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType

# --- RAG Setup Function ---
def setup_rag(file_path):
    """Set up RAG with the uploaded medical report"""
    # Load data from the text file
    loader = TextLoader(file_path)
    documents = loader.load()
    st.success("📄 Loaded document")
    
    # Split data into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    st.success(f"🧩 Document split into {len(docs)} chunks")
    
    # Create vector store using Ollama embeddings and FAISS
    embeddings = OllamaEmbeddings(model="llama3")
    vector_store = FAISS.from_documents(docs, embeddings)
    st.success("✅ Embeddings stored in FAISS")
    
    # Initialize a retriever for querying the vector store
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Initialize the Ollama LLM for the RAG chain with temperature=0
    llm = ChatOllama(model="mistral-nemo:latest", temperature=0)
    
    # Create the Retrieval QA chain
    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    # Tool: Connect the document retriever to the agent
    retrieval_tool = Tool(
        name="Document Retrieval",
        func=lambda q: retrieval_qa_chain.invoke(q)["result"],
        description="Retrieves answers from the medical document database."
    )
    
    return retrieval_tool, llm

# --- Specialist Tools ---
@tool
def analyze_cardiologist(medical_report: str):
    """
    Analyze the patient's report from a cardiologist's perspective.
    """
    prompt = f"""
    Act like a cardiologist. You will receive a medical report of a patient.
    Task: Review the patient's cardiac workup, including ECG, blood tests, Holter monitor results, and echocardiogram.
    Focus: Determine if there are any subtle signs of cardiac issues that could explain the patient's symptoms. Rule out any underlying heart conditions.
    Recommendation: Provide guidance on any further cardiac testing or monitoring needed. Suggest potential management strategies.
    Medical Report: {medical_report}
    """
    llm = ChatOllama(model="mistral-nemo:latest", temperature=0)
    return llm.invoke(prompt).content

@tool
def analyze_psychologist(medical_report: str):
    """
    Analyze the patient's report from a psychologist's perspective.
    """
    prompt = f"""
    Act like a psychologist. You will receive a patient's report.
    Task: Review the report and provide a psychological assessment.
    Focus: Identify potential mental health issues such as anxiety or depression.
    Recommendation: Offer guidance on therapy or other interventions.
    Medical Report: {medical_report}
    """
    llm = ChatOllama(model="mistral-nemo:latest", temperature=0)
    return llm.invoke(prompt).content

@tool
def analyze_pulmonologist(medical_report: str):
    """
    Analyze the patient's report from a pulmonologist's perspective.
    """
    prompt = f"""
    Act like a pulmonologist. You will receive a patient's report.
    Task: Review the report and provide a pulmonary assessment.
    Focus: Identify potential respiratory issues such as asthma, COPD, or infections.
    Recommendation: Offer guidance on tests or treatments.
    Medical Report: {medical_report}
    """
    llm = ChatOllama(model="mistral-nemo:latest", temperature=0)
    return llm.invoke(prompt).content

@tool
def analyze_multidisciplinary(cardiologist_report: str, psychologist_report: str, pulmonologist_report: str):
    """
    Combine specialist reports to determine three possible health issues.
    """
    prompt = f"""
    You are a multidisciplinary team of healthcare professionals.
    Task: Review the cardiologist, psychologist, and pulmonologist reports.
    Output: List 3 possible health issues the patient may have, with reasons for each.

    Cardiologist Report: {cardiologist_report}
    Psychologist Report: {psychologist_report}
    Pulmonologist Report: {pulmonologist_report}
    """
    llm = ChatOllama(model="mistral-nemo:latest", temperature=0)
    return llm.invoke(prompt).content

# --- Streamlit App UI ---
st.set_page_config(page_title="Advanced Medical Report Analyzer", page_icon="🧠")
st.title("🧠 Advanced Medical Report Analyzer")

uploaded_file = st.file_uploader("📤 Upload a medical report (TXT)", type=["txt"])

if uploaded_file:
    # Save the uploaded file to a temporary location
    temp_file_path = "temp_medical_report.txt"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    medical_report = uploaded_file.getvalue().decode("utf-8")
    
    analysis_mode = st.radio(
        "Select Analysis Mode",
        options=["Specialist Analysis", "RAG-Powered Q&A"]
    )
    
    if analysis_mode == "Specialist Analysis":
        st.markdown("### 🧑‍⚕️ Choose a Specialist to Analyze the Report")
        specialist = st.selectbox(
            "Select Specialist",
            options=["Cardiologist", "Psychologist", "Pulmonologist", "All Specialists"]
        )

        if st.button("🔍 Analyze Report"):
            with st.spinner("Analyzing report..."):
                if specialist == "Cardiologist":
                    result = analyze_cardiologist.invoke({"medical_report": medical_report})
                    st.markdown("### 🫀 Cardiologist Report")
                    st.text(result)
                elif specialist == "Psychologist":
                    result = analyze_psychologist.invoke({"medical_report": medical_report})
                    st.markdown("### 🧠 Psychologist Report")
                    st.text(result)
                elif specialist == "Pulmonologist":
                    result = analyze_pulmonologist.invoke({"medical_report": medical_report})
                    st.markdown("### 🌬️ Pulmonologist Report")
                    st.text(result)
                elif specialist == "All Specialists":
                    st.markdown("### ⏳ Running all specialist tools...")
                    cardiologist_report = analyze_cardiologist.invoke({"medical_report": medical_report})
                    psychologist_report = analyze_psychologist.invoke({"medical_report": medical_report})
                    pulmonologist_report = analyze_pulmonologist.invoke({"medical_report": medical_report})

                    st.success("✅ All specialist reports generated!")

                    with st.expander("🫀 Cardiologist Report"):
                        st.text(cardiologist_report)
                    with st.expander("🧠 Psychologist Report"):
                        st.text(psychologist_report)
                    with st.expander("🌬️ Pulmonologist Report"):
                        st.text(pulmonologist_report)

                    # Final multidisciplinary analysis
                    with st.spinner("🔍 Combining results for final diagnosis..."):
                        final_diagnosis = analyze_multidisciplinary.invoke({
                            "cardiologist_report": cardiologist_report,
                            "psychologist_report": psychologist_report,
                            "pulmonologist_report": pulmonologist_report
                        })

                    st.markdown("### ✅ Final Diagnosis")
                    st.markdown(final_diagnosis)

                    # Save to file
                    os.makedirs("results", exist_ok=True)
                    output_path = "results/final_diagnosis.txt"
                    with open(output_path, "w") as f:
                        f.write(final_diagnosis)

                    st.success(f"📁 Final diagnosis saved to `{output_path}`.")
    
    else:  # RAG-Powered Q&A
        st.markdown("### 🔍 RAG-Powered Medical Report Q&A")
        
        if "rag_initialized" not in st.session_state:
            with st.spinner("Initializing RAG system..."):
                retrieval_tool, llm = setup_rag(temp_file_path)
                
                # Combine tools for the agent (only the retrieval tool)
                tools = [retrieval_tool]
                
                # Initialize the agent with tools
                agent = initialize_agent(
                    tools=tools,
                    llm=llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    handle_parsing_errors=True
                )
                
                st.session_state.agent = agent
                st.session_state.rag_initialized = True
                st.success("🤖 Agent initialized with RAG")
        
        # User query interface
        user_query = st.text_input("Ask a question about the medical report:")
        
        if user_query and st.button("Submit Query"):
            with st.spinner("Processing your query..."):
                try:
                    response = st.session_state.agent.run(user_query)
                    st.markdown("### 📝 Response")
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.markdown("Please try rephrasing your question or check if the system is properly initialized.")

# Cleanup temp file when app is done
try:
    if os.path.exists("temp_medical_report.txt"):
        os.remove("temp_medical_report.txt")
except:
    pass
