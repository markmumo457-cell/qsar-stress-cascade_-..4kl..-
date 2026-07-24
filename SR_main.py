from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

# 1. Initialize the Server
app = FastAPI(
    title="Toxicity Biological Cascade API",
    description="Multi-gene classifier chain predicting p53, ATAD5, ARE, and MMP disruptions."
)

import os
import joblib

# 1. Ask Python exactly where SR_main.py is physically located on the hard drive
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Attach the model filename to that exact folder location
model_path = os.path.join(BASE_DIR, "Tox21_stressresponse_cascade.joblib")

# 3. Load the model using that bulletproof path
#model = joblib.load(model_path)
# 2. Load the Masterpiece Globally
# We load the .joblib file OUTSIDE the endpoint. 
# This way, the server loads the heavy matrix into RAM exactly once when it starts, 
# rather than reloading it from the hard drive every single time a request comes in.

try:
    deployed_model = joblib.load(model_path)
    #deployed_model = joblib.load("Tox21_stressresponse_cascade.joblib")
except Exception as e:
    print(f"Failed to load model: {e}")
    deployed_model = None

# 3. Define the Input Schema
# This forces the API to strictly accept a JSON list of strings.
class ChemicalBatch(BaseModel):
    smiles_list: list[str]

targets = ['SR-p53', 'SR-ATAD5', 'SR-ARE', 'SR-MMP']
clinical_threshold = 0.15

# 4. The API Endpoint
@app.post("/diagnose")
def diagnose_chemicals(batch: ChemicalBatch):
    if deployed_model is None:
        raise HTTPException(status_code=500, detail="The model vault is offline.")
    
    batch_fps = []
    valid_smiles = []
    invalid_smiles = []
    
    # Step A: RDKit Translation
    for smiles in batch.smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles.append(smiles)
            continue
            
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_array = np.zeros((1, 1024))
        ConvertToNumpyArray(fp, fp_array[0])
        
        batch_fps.append(fp_array[0])
        valid_smiles.append(smiles)
        
    # If all provided SMILES were invalid, abort early
    if not batch_fps:
        return {"error": "No valid chemical structures could be rendered.", "invalid": invalid_smiles}
        
    # Step B: Batch Prediction
    fp_matrix = np.array(batch_fps)
    raw_probabilities = deployed_model.predict_proba(fp_matrix)
    hard_alarms = (raw_probabilities >= clinical_threshold).astype(int)
    
    # Step C: Format the JSON Response
    # Web APIs do not use print() statements; they return structured dictionaries.
    results = []
    for idx, smiles in enumerate(valid_smiles):
        chemical_report = {"smiles": smiles, "diagnostics": {}}
        for j, gene in enumerate(targets):
            prob = float(raw_probabilities[idx][j]) # Convert numpy float to standard Python float for JSON
            status = "CRITICAL" if hard_alarms[idx][j] == 1 else "SAFE"
            
            chemical_report["diagnostics"][gene] = {
                "status": status,
                "probability": prob
            }
        results.append(chemical_report)
        
    # Return the clean, machine-readable JSON data
    return {
        "processed_chemicals": results,
        "failed_chemicals": invalid_smiles
    }