import requests
import json
import pandas as pd
from tqdm import tqdm

def lookup_ensembl_coordinates(symbols, chunk_size=50):
    """
    Bulk-lookup chromosome, start, and end coordinates (GRCh38)
    for a list of gene symbols using Ensembl REST API.
    Automatically chunks requests to avoid the 50-symbol limit.
    
    Returns a pandas DataFrame with columns:
      ['gene_symbol', 'chromosome', 'start', 'end']
    """
    
    server = "https://rest.ensembl.org"
    endpoint = "/lookup/symbol/homo_sapiens"  # POST endpoint
    headers={ "Content-Type" : "application/json", "Accept" : "application/json"}
    
    all_results = []

    # Break the list of symbols into chunks
    for i in tqdm(range(0, len(symbols), chunk_size)):
        chunk = symbols[i : i + chunk_size]

        # Prepare the JSON body for POST
        # Use "symbols" key, with a list of {"symbol": "XYZ"} dicts
        data = {"symbols": chunk}

        # Make the POST request
        headers = {"Content-Type": "application/json"}
        response = requests.post(server + endpoint,
                                 headers=headers,
                                 data=json.dumps(data))
        
        # Raise an error if the request failed
        if not response.ok:
            response.raise_for_status()

        # Parse JSON response
        res_json = response.json()
        # and values = info about each gene or None if not found.
        for symbol in res_json.keys():
            gene_info = res_json[symbol]
            if gene_info is None:
                # Symbol not found by Ensembl
                all_results.append({
                    "gene_symbol": symbol,
                    "chromosome": None,
                    "start": None,
                    "end": None
                })
            else:
                # Extract the genomic coordinates
                ensembl_id = gene_info.get("id", None)
                chrom = gene_info.get("seq_region_name", None)
                start = gene_info.get("start", None)
                end   = gene_info.get("end", None)
        
                all_results.append({
                    "gene_symbol": symbol,
                    "ensembl_id": ensembl_id,
                    "chromosome": chrom,
                    "start": start,
                    "end": end
                })

    # Convert results to a DataFrame
    df = pd.DataFrame(all_results)
    return df
