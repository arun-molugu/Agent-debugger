import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Agent Debugger", page_icon="🔍")

st.title("🔍 Agent Debugger")
st.write("Paste any agent execution trace below to get an instant debugging report.")

api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
trace = st.text_area("Paste agent trace here", height=250, placeholder="""User: Process my refund
Agent: Looking up order...
Tool: order_lookup returned: error - not found
Agent: Your refund has been processed successfully.""")

if st.button("Analyze Trace", type="primary"):
    if not api_key or not trace:
        st.error("Please add your API key and paste a trace.")
    else:
        with st.spinner("Analyzing trace..."):
            client = OpenAI(api_key=api_key)
            prompt = f"""
You are a senior AI agent debugging engineer.
Analyze this agent trace and return a clear debugging report.

Return:
1. Timeline — step by step what happened
2. Failures detected — type, evidence, severity
3. Root cause — confirmed cause for each failure
4. Suggested fixes — quick fix and robust fix
5. Reliability score out of 100
   - Hallucination: -30
   - Tool misuse: -20
   - Logic error: -15
   - Missing context: -10
6. Confidence score 0 to 1

Rules:
- Use exact quotes from trace as evidence
- Never use unknown if contradiction is obvious
- Hallucination severity is always critical
- Be specific not generic

Trace:
{trace}
"""
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown(response.choices[0].message.content)

st.divider()
st.caption("Built by Arun | Agent observability and diagnosis layer")
