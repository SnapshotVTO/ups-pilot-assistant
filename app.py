import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from fatigue import UPSFatigueModel

# --- PAGE CONFIG ---
st.set_page_config(page_title="UPS Pilot Assistant", page_icon="✈️", layout="wide")

st.title("✈️ UPS Pilot Contract & Fatigue Manager")
st.markdown("### Maximize Efficiency. Minimize Fatigue.")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("🛠 Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.info("Enter API key to enable Chatbot/Contract search.")
    
    st.divider()
    st.header("📂 Contract Documents")
    uploaded_files = st.file_uploader("Upload Contract/Ref Guides", accept_multiple_files=True, type="pdf")
    
# --- TABS ---
tab1, tab2 = st.tabs(["📊 Fatigue Analysis", "💬 Contract Chatbot"])

# --- TAB 1: FATIGUE MODEL ---
with tab1:
    st.subheader("Trip Fatigue Analyzer")
    
    # Simple input for prototype (replace with parser later)
    col1, col2, col3 = st.columns(3)
    with col1:
        origin = st.text_input("Origin", "SDF")
    with col2:
        dest = st.text_input("Dest", "ANC")
    with col3:
        dept_time = st.time_input("Departure Time (Local)", datetime.strptime("02:30", "%H:%M").time())
    
    flight_date = st.date_input("Date", datetime.today())
    block_hours = st.number_input("Block Hours", min_value=0.0, value=7.5, step=0.1)
    
    if st.button("Run Analysis"):
        # Construct Flight Object
        dept_dt = datetime.combine(flight_date, dept_time)
        arr_dt = dept_dt + timedelta(hours=block_hours)
        
        flights = [{
            'origin': origin,
            'dest': dest,
            'dept': dept_dt,
            'arr': arr_dt
        }]
        
        # Run Model
        model = UPSFatigueModel()
        duty = model.calculate_duty(flights, is_in_domicile=True)
        
        # Display Results
        st.success(f"Duty Period: {duty['start'].strftime('%H:%M')} to {duty['end'].strftime('%H:%M')}")
        
        if duty['is_intl']:
            st.badge("International Rules Applied")
            crew = model.get_crew_complement(block_hours)
            st.info(f"Crew Requirement: {crew} Pilots")
        
        # Run Simulation
        df_results = model.run_simulation(duty['start'], duty['end'], flights)
        
        # Metrics
        min_eff = df_results['Effectiveness'].min()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Min Effectiveness", f"{min_eff}%")
        m2.metric("Duty Duration", f"{duty['duration']:.2f} hrs")
        
        # Show Data
        st.dataframe(df_results, use_container_width=True)
        
        # Simple Graph
        st.line_chart(df_results.set_index("Time")["Effectiveness"])

# --- TAB 2: CHATBOT (Placeholder for RAG) ---
with tab2:
    st.subheader("Contract Q&A")
    if not api_key:
        st.warning("Please enter an OpenAI API Key in the sidebar to use the Chatbot.")
    else:
        user_query = st.text_input("Ask a contract question...")
        if user_query:
            st.write("🤖 AI is thinking... (RAG Logic would go here)")
            # In the full version, we connect this to LangChain + VectorDB
            st.markdown(f"> **Answer:** Based on Article 13 of Contract 16...")
