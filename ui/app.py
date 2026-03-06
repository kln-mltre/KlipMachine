"""
KlipMachine - Streamlit UI
Professional interface for AI-powered clip extraction.
V2.0 - State Machine Workflow
"""

import streamlit as st
#from PIL import Image

from ui.components import (
    init_session_state,
    render_step1_ingestion,
    render_step1_5_refine,
    render_step2_design,
    render_step3_export
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

#icon = Image.open("logo.png")
st.set_page_config(
    page_title="KlipMachine",
    #page_icon=icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CSS STYLING
# =============================================================================

# Resolve the active step early; needed to compute the CSS width before the markdown block.
current_step_css = st.session_state.get("step", 1)

# Step 1.5 (clip refinement) uses a wider canvas to accommodate side-by-side editing panels.
if current_step_css == 1.5:
    app_width = "1600px"
else:
    app_width = "1400px"  # Standard width for Ingestion, Design, and Export steps.

st.markdown(f"""
<style>
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: {app_width} !important;
    }}
    
    /* =========================================
       2. STEP INDICATOR (progress bar)
       ========================================= */
    .step-indicator {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
    }}
    
    .step-item {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        color: #9ca3af; /* Default inactive colour */
        transition: all 0.3s ease;
    }}
    
    /* Active step */
    .step-item.active {{
        color: white;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        box-shadow: 0 4px 10px rgba(99, 102, 241, 0.3);
    }}
    
    /* Completed step */
    .step-item.completed {{
        color: #c7d2fe;
        background: rgba(99, 102, 241, 0.2);
    }}

    /* =========================================
       3. BUTTONS
       ========================================= */

    /* Secondary (default) button style */
    .stButton > button {{
        width: 100%;
        height: 50px !important;            
        min-height: 50px !important;        
        display: flex !important;           
        align-items: center !important;     
        justify-content: center !important; 
        
        background-color: transparent !important;
        color: #e5e7eb !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0 1.5rem !important;
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        border-color: #6366f1 !important;
        color: #6366f1 !important;
        background-color: rgba(99, 102, 241, 0.1) !important;
    }}

    /* Primary (action) button style */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: 1px solid transparent !important; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }}

    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        color: white !important;
        border: 1px solid transparent !important;
    }}

    .stButton > button:disabled {{
        background-color: #374151 !important;
        background: #374151 !important;
        opacity: 0.5;
        border: 1px solid transparent !important;
        color: #9ca3af !important;
    }}
    
    /* Progress bar */
    .stProgress > div > div {{
        background-color: #6366f1;
    }}
    
    /* Spacing */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlock"] {{
        gap: 1.5rem;
    }}
    
    /* Radio buttons spacing */
    .stRadio > div {{
        gap: 0.75rem;
    }}
    
    /* Inputs */
    .stTextInput > div > div > input {{
        padding: 0.65rem 0.75rem;
    }}
    
    /* Dividers */
    hr {{
        margin: 1.5rem 0;
    }}
    
    /* Column titles */
    h3 {{
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 1.25rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #374151;
    }}
    
    h4 {{
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #9ca3af;
    }}
    
    /* Labels */
    .element-container:has(.stMarkdown) p strong {{
        font-weight: 500;
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    /* Column containers */
    [data-testid="column"] {{
        background-color: rgba(255, 255, 255, 0.02);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    
    /* Horizontal radios */
    .row-widget.stRadio > div {{
        flex-direction: row;
        gap: 1rem;
    }}
    
    /* Primary button */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        font-size: 1.05rem;
        padding: 1rem 1.5rem;
        margin-top: 0.5rem;
    }}
    
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    }}
    
    /* Footer */
    .caption {{
        opacity: 0.4;
        font-size: 0.75rem;
    }}
    
    /* Step indicator */
    .step-indicator {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
    }}
    
    .step-item {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        opacity: 0.5;
    }}
    
    .step-item.active {{
        opacity: 1;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    }}
    
    .step-item.completed {{
        opacity: 0.8;
        background: rgba(99, 102, 241, 0.2);
    }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

init_session_state()


# =============================================================================
# HEADER & STEP INDICATOR
# =============================================================================

st.title("KlipMachine")
st.caption("AI-Powered Clip Extraction • V2.0")

# Step indicator
current_step = st.session_state.step

step1_class = "active" if current_step in [1, 1.5] else ("completed" if current_step > 1.5 else "")
step2_class = "active" if current_step == 2 else ("completed" if current_step > 2 else "")
step3_class = "active" if current_step == 3 else ""

st.markdown(f"""
<div class="step-indicator">
    <div class="step-item {step1_class}">
        <span>Ingestion</span>
    </div>
    <span style="opacity: 0.3;">→</span>
    <div class="step-item {step2_class}">
        <span>Design</span>
    </div>
    <span style="opacity: 0.3;">→</span>
    <div class="step-item {step3_class}">
        <span>Export</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# =============================================================================
# WORKFLOW ROUTER
# =============================================================================

if current_step == 1:
    render_step1_ingestion()

elif current_step == 1.5:
    render_step1_5_refine()

elif current_step == 2:
    render_step2_design()

elif current_step == 3:
    render_step3_export()

else:
    st.error("Invalid workflow step")
    if st.button("Reset to Step 1"):
        st.session_state.step = 1
        st.rerun()


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("KlipMachine v2.0 • Made by KLN")