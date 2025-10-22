import pandas as pd

def load_id_mappings(path: str):
    """Load ID mappings (admission_type, discharge_disposition, admission_source) from CSV"""
    mappings = {}
    df = pd.read_csv(path)

    #the mapping CSV may contain multiple sections separated by blank lines
    current_category = None
    for _, row in df.iterrows():
        if pd.isna(row[0] or row[0] == ""):
            continue
        key = str(row[0]).strip()
        val = str(row[1]).strip() if len(row) > 1 else ""
        if key.endswith("_id") and not key[0].isdigit():
            # header
            current_category = key
            mappings[current_category] = {}
        elif current_category:
            mappings[current_category][int(key)] = val
    return mappings

def map_icd_to_group(code: str):
    """Map an ICD code to a broader group (rough approximation if no explicit mapping provided)"""
    if not code or code == "?" or pd.isna(code):
        return None
    code_str = str(code)
    if code_str.startswith("V") or code_str.startswith("E"):
        # use first 3 characters for V/E codes
        group = code_str[:2]
    else:
        group = code_str[:3] if len(code_str) >=3 else code_str

    return group

def map_drug_to_class(drug: str):
    """Map drug name to a drug class (approximate mapping)"""
    drug = drug.lower()
    if "-" in drug:
        return "combination"
    classes = {
        "metformin": "biguanide",
        "repaglinide": "meglitinide", "nateglinide": "meglitinide",
        "chlorpropamide": "sulfonylurea", "glimepiride": "sulfonylurea",
        "acetohexamide": "sulfonylurea", "glipizide": "sulfonylurea",
        "glyburide": "sulfonylurea", "tolbutamide": "sulfonylurea", "tolazamide": "sulfonylurea",
        "pioglitazone": "thiazolidinedione", "rosiglitazone": "thiazolidinedione", "troglitazone": "thiazolidinedione",
        "acarbose": "alpha_glucosidase_inhibitor", "miglitol": "alpha_glucosidase_inhibitor",
        "sitagliptin": "dpp4_inhibitor",
        "insulin": "insulin",
        "examide": "other", "citoglipton": "other"
    }
    for name, cls in classes.items():
        if drug.startswith(name):
            return cls
    return "other"