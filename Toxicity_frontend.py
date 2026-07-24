import streamlit as st
import requests
import io
import base64
from rdkit import Chem
from rdkit.Chem import Draw

# --- 1. Page Configuration & UI Settings ---
st.set_page_config(page_title="Stress Response Bio-Cascade UI ", layout="wide")

# --- CUSTOM CSS INJECTION ---
st.markdown("""
<style>
    /* Overall Light Blue Background */
    .stApp {
        background-color: #e6f3ff;
    }

    /* Enclosed box for Molecule Diagrams */
    .molecule-container {
        border: 2px solid #3d607e;
        border-radius: 8px;
        padding: 10px;
        background-color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        max-width: 250px;
        margin-bottom: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }

    /* General text coloring */
    .stMarkdown, .stText, h1, h3 {
        color: #1e3a5f !important;
    }

    /* Clean Custom HTML Table Styling */
    .custom-table {
        margin-left: auto;
        margin-right: auto;
        width: 100%;
        max-width: 450px;
        background-color: white;
        border: 1px solid #ddd;
        border-collapse: collapse;
        font-family: sans-serif;
    }
    
    .custom-table th {
        color: black !important;
        font-weight: bold !important;
        text-align: center;
        padding: 10px;
        border-bottom: 2px solid #ddd;
    }
    
    .custom-table td {
        color: #1e3a5f;
        padding: 10px;
        text-align: center;
        border-bottom: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)


# The updated title with the DNA helix
st.title("🧬 Cellular Stress Response Predictor")
st.markdown("Input chemical SMILES strings to run the QSAR predictive model.The model assesses the probability of stress response induction across the p53, ATAD5, ARE, and MMP pathways.")


# --- Helper Function: SMILES to Base64 Image string ---
def smiles_to_base64_image(smiles, size=(180, 180)):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=size)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        return None
    except Exception:
        return None


# --- 2. User Input Area ---
smiles_input = st.text_area(
    "Chemical Batch (One SMILES string per line):",
    "Oc1ccccc1\nCCO\nCC(=O)OC1=CC=CC=C1C(=O)O",
    height=150
)

# --- 3. The Execution Button ---
if st.button("Run Biological Cascade"):
    if not smiles_input.strip():
        st.warning("Please enter at least one chemical structure.")
    else:
        smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
        
        # 4. Talk to the FastAPI Backend
        api_url = "https://molecular-stress-response-api.onrender.com/diagnose"
        payload = {"smiles_list": smiles_list}
        
        with st.spinner("Analyzing structures and rendering UI..."):
            try:
                response = requests.post(api_url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    st.subheader("Molecular Report")
                    
                    # 5. Render Results 
                    for chem in data.get("processed_chemicals", []):
                        current_smiles = chem['smiles']
                        
                        col_image, col_table = st.columns([1.5, 3])
                        
                        # --- Column 1: Molecular Structure ---
                        with col_image:
                            st.write("") 
                            img_base64 = smiles_to_base64_image(current_smiles, size=(180, 180))
                            
                            if img_base64:
                                html_markup = f"""
                                <div class="molecule-container">
                                    <img src="data:image/png;base64,{img_base64}" alt="Chemical Structure">
                                </div>
                                """
                                st.markdown(html_markup, unsafe_allow_html=True)
                            else:
                                st.error("Image generation failed.")

                        # --- Column 2: Diagnostic Table (Flattened HTML) ---
                        with col_table:
                            st.markdown(f"**Molecule (SMILES):** `{current_smiles}`")
                            
                            diag = chem["diagnostics"]
                            
                            # Constructing the HTML completely flat to bypass Markdown's 4-space code block rule
                            table_html = '<table class="custom-table">'
                            table_html += '<thead><tr><th>Gene Target</th><th>Status</th><th>Probability</th></tr></thead><tbody>'
                            
                            for g in diag.keys():
                                raw_status = diag[g]["status"]
                                prob = f"{diag[g]['probability']:.4f}"
                                
                                # Vocabulary translation based on backend logic
                                display_status = "Activate" if raw_status == "CRITICAL" else "Safe"
                                color = "#d9534f" if display_status == "Activate" else "#5cb85c"
                                
                                # Flat string concatenation
                                table_html += f'<tr><td>{g}</td><td style="color: {color}; font-weight: bold;">{display_status}</td><td>{prob}</td></tr>'
                                
                            table_html += '</tbody></table>'
                            
                            st.markdown(table_html, unsafe_allow_html=True)
                        
                        st.divider() 
                        
                    if data.get("failed_chemicals"):
                        st.error(f"Failed to process: {', '.join(data['failed_chemicals'])}")
                        
                else:
                    st.error(f"Server Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Connection Failed. Is FastAPI server running on Port 8001?")