#!/usr/bin/env python3
"""
Script to download MERS-CoV and SARS-CoV-2 main protease (Mpro) structures from RCSB PDB.
Includes retry logic and error handling to prevent API failures.
"""

import os
import requests
import gzip
import json
import argparse
import time
import random
from pathlib import Path
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Constants
CORONAVIRUS_TYPES = {
    "mers": {
        "name": "MERS-CoV",
        "search_terms": [
            "Middle East respiratory syndrome-related coronavirus", 
            "Middle East respiratory syndrome coronavirus"
        ],
        "dir_name": "mers_cov_mpro"
    },
    "sars2": {
        "name": "SARS-CoV-2",
        "search_terms": [
            "Severe acute respiratory syndrome coronavirus 2"
        ],
        "dir_name": "sars_cov2_mpro"
    }
}

PROTEASE_KEYWORDS = [
    "protease", 
    "main protease", 
    "3CLpro", 
    "Mpro"
]

# Method descriptions for filtering
EXPERIMENTAL_METHODS = {
    "x-ray": "X-RAY DIFFRACTION",
    "em": "ELECTRON MICROSCOPY",
    "nmr": "SOLUTION NMR"
}

# Create a session with retry mechanism
def create_requests_session(retries=5, backoff_factor=0.3, 
                           status_forcelist=(500, 502, 504)):
    """
    Create a requests session with retry mechanism.
    
    Args:
        retries (int): Number of retries
        backoff_factor (float): Backoff factor for retries
        status_forcelist (tuple): HTTP status codes to retry on
        
    Returns:
        requests.Session: Session with retry capability
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Global session for all requests
REQUEST_SESSION = create_requests_session()

def create_search_query(organism, keyword, resolution_cutoff=None, experimental_method=None):
    """
    Create a search query for RCSB PDB API.
    
    Args:
        organism (str): Organism name for taxonomy search
        keyword (str): Keyword to search in structure title
        resolution_cutoff (float, optional): Maximum resolution in Angstroms
        experimental_method (str, optional): Experimental method filter
        
    Returns:
        dict: Query dictionary for RCSB search API
    """
    # Start with organism and keyword nodes
    query_nodes = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entity_source_organism.taxonomy_lineage.name",
                "operator": "exact_match",
                "value": organism
            }
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "struct.title",
                "operator": "contains_words",
                "value": keyword
            }
        }
    ]
    
    # Add resolution filter if specified
    if resolution_cutoff is not None:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": resolution_cutoff
            }
        })
    
    # Add experimental method filter if specified
    if experimental_method is not None:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": EXPERIMENTAL_METHODS[experimental_method]
            }
        })
    
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": query_nodes
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
            "scoring_strategy": "combined",
            "return_all_hits": True
        }
    }

def create_similar_complex_query(resolution_cutoff=None, experimental_method=None):
    """
    Create a query for similar complex structures (protease-inhibitor complexes).
    
    Args:
        resolution_cutoff (float, optional): Maximum resolution in Angstroms
        experimental_method (str, optional): Experimental method filter
        
    Returns:
        dict: Query dictionary for RCSB search API
    """
    query_nodes = [
        # Match structures with "protease" and "inhibitor" in title
        {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "struct.title",
                        "operator": "contains_words",
                        "value": "protease"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "struct.title",
                        "operator": "contains_words",
                        "value": "inhibitor"
                    }
                }
            ]
        }
    ]
    
    # Add resolution filter if specified
    if resolution_cutoff is not None:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": resolution_cutoff
            }
        })
    
    # Add experimental method filter if specified
    if experimental_method is not None:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": EXPERIMENTAL_METHODS[experimental_method]
            }
        })
    
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": query_nodes
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
            "scoring_strategy": "combined",
            "return_all_hits": True
        }
    }

def search_pdb_structures(organisms, keywords, resolution_cutoff=None, experimental_method=None):
    """
    Search for PDB structures for given organisms and keywords with rate limiting.
    
    Args:
        organisms (list): List of organism names for taxonomy search
        keywords (list): List of keywords to search in structure title
        resolution_cutoff (float, optional): Maximum resolution in Angstroms
        experimental_method (str, optional): Experimental method filter
        
    Returns:
        list: List of PDB IDs
    """
    base_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    all_pdb_ids = set()
    
    print(f"Searching for structures with keywords: {', '.join(keywords)}")
    print(f"Resolution cutoff: {resolution_cutoff or 'None'}")
    print(f"Experimental method: {experimental_method or 'Any'}")
    
    for organism in organisms:
        for keyword in keywords:
            print(f"Searching for {organism} structures with keyword '{keyword}'...")
            query = create_search_query(organism, keyword, resolution_cutoff, experimental_method)
            
            try:
                # Add delay to prevent overwhelming the API
                time.sleep(random.uniform(0.5, 1.5))
                
                response = REQUEST_SESSION.post(base_url, json=query)
                response.raise_for_status()
                
                results = response.json()
                hits = results.get("result_set", [])
                
                # Extract PDB IDs
                pdb_ids = [hit["identifier"] for hit in hits]
                all_pdb_ids.update(pdb_ids)
                
                print(f"  Found {len(pdb_ids)} structures")
                
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                print(f"Error searching for {organism} structures with keyword '{keyword}': {e}")
                # Continue with next search despite errors
    
    return list(all_pdb_ids)

def search_similar_complexes(resolution_cutoff=None, experimental_method=None, max_results=1000):
    """
    Search for similar protease-inhibitor complex structures.
    
    Args:
        resolution_cutoff (float, optional): Maximum resolution in Angstroms
        experimental_method (str, optional): Experimental method filter
        max_results (int): Maximum number of results to return
        
    Returns:
        list: List of PDB IDs
    """
    base_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    print("\nSearching for similar protease-inhibitor complexes...")
    query = create_similar_complex_query(resolution_cutoff, experimental_method)
    
    try:
        response = REQUEST_SESSION.post(base_url, json=query)
        response.raise_for_status()
        
        results = response.json()
        hits = results.get("result_set", [])
        
        # Extract PDB IDs (limited to max_results to prevent overwhelming the system)
        pdb_ids = [hit["identifier"] for hit in hits[:max_results]]
        
        print(f"  Found {len(pdb_ids)} similar complex structures (out of {len(hits)} total)")
        return pdb_ids
        
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"Error searching for similar complexes: {e}")
        return []

def get_structure_info(pdb_id):
    """
    Get detailed information about a PDB structure with rate limiting.
    
    Args:
        pdb_id (str): PDB ID
        
    Returns:
        dict: Structure information or None if not found
    """
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    
    try:
        # Add delay to prevent overwhelming the API
        time.sleep(random.uniform(0.2, 0.5))
        
        response = REQUEST_SESSION.get(url)
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        return None

def download_pdb_file(pdb_id, output_dir, assembly_id=None, max_retries=3):
    """
    Download a PDB file with retry logic.
    
    Args:
        pdb_id (str): PDB ID
        output_dir (str): Directory to save the file
        assembly_id (str, optional): Assembly ID if downloading biological assembly
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        str: Path to downloaded file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if assembly_id:
        # Downloading biological assembly
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb{assembly_id}.gz"
        output_file = os.path.join(output_dir, f"{pdb_id}_assembly{assembly_id}.pdb")
    else:
        # Downloading asymmetric unit
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        output_file = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    # Check if file already exists
    if os.path.exists(output_file):
        print(f"File already exists: {output_file}")
        return output_file
    
    # Try downloading with retries
    for attempt in range(max_retries):
        try:
            # Add delay between retries
            if attempt > 0:
                time.sleep(random.uniform(1, 3))
                
            response = REQUEST_SESSION.get(url, stream=True)
            response.raise_for_status()
            
            # Get total file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Initialize progress bar
            progress_bar = tqdm(
                total=total_size, 
                unit='B', 
                unit_scale=True,
                desc=f"Downloading {pdb_id}" + (f" assembly {assembly_id}" if assembly_id else "")
            )
            
            # Download and decompress if needed
            if url.endswith('.gz'):
                # For gzipped files
                with open(output_file, 'wb') as f_out:
                    with gzip.GzipFile(fileobj=response.raw) as f_in:
                        for chunk in f_in:
                            f_out.write(chunk)
                            progress_bar.update(len(chunk))
            else:
                # For plain text files
                with open(output_file, 'wb') as f_out:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f_out.write(chunk)
                            progress_bar.update(len(chunk))
            
            progress_bar.close()
            return output_file
                
        except (requests.exceptions.RequestException, IOError) as e:
            if attempt < max_retries - 1:
                print(f"Error downloading {pdb_id}, attempt {attempt+1}/{max_retries}: {e}")
            else:
                print(f"Failed to download {pdb_id} after {max_retries} attempts: {e}")
                # Remove partially downloaded file if it exists
                if os.path.exists(output_file):
                    os.remove(output_file)
    
    return None

def get_biological_assemblies(pdb_id, max_retries=3):
    """
    Get biological assembly information for a PDB structure with retry logic.
    
    Args:
        pdb_id (str): PDB ID
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        list: List of assembly IDs
    """
    assembly_url = f"https://data.rcsb.org/rest/v1/core/assembly/{pdb_id}"
    
    for attempt in range(max_retries):
        try:
            # Add delay between retries
            if attempt > 0:
                time.sleep(random.uniform(1, 3))
                
            response = REQUEST_SESSION.get(assembly_url)
            response.raise_for_status()
            
            assemblies = response.json()
            # Filter out non-assembly keys
            assembly_ids = [key for key in assemblies.keys() if key != "rcsb_id"]
            return assembly_ids
            
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                # Don't log error for retries to reduce output noise
                pass
            else:
                print(f"Error getting assembly information for {pdb_id}: {e}")
    
    return []

def classify_coronavirus_type(pdb_id, structure_info):
    """
    Classify a structure into one of the coronavirus types with relaxed criteria.
    
    Args:
        pdb_id (str): PDB ID
        structure_info (dict): Structure information
        
    Returns:
        str: Coronavirus type key or None if not identified
    """
    # If structure_info is not available, return None
    if not structure_info:
        return None
    
    # Extract title and taxonomic information
    title = structure_info.get("struct", {}).get("title", "").lower()
    
    # Check entity sources
    entities = structure_info.get("rcsb_entity_source_organism", [])
    taxonomy_names = []
    
    for entity in entities:
        taxonomy = entity.get("taxonomy_lineage", [])
        for tax_entry in taxonomy:
            name = tax_entry.get("name", "")
            if name:
                taxonomy_names.append(name.lower())
    
    # MERS-CoV related keywords
    mers_keywords = [
        "middle east respiratory syndrome",
        "mers",
        "mers-cov"
    ]
    
    # SARS-CoV-2 related keywords
    sars2_keywords = [
        "sars-cov-2",
        "sars-cov2",
        "covid",
        "covid-19",
        "covid19",
        "2019-ncov"
    ]
    
    # Check for MERS-CoV
    for keyword in mers_keywords:
        if keyword in title:
            return "mers"
        
        for tax_name in taxonomy_names:
            if keyword in tax_name:
                return "mers"
    
    # Check for SARS-CoV-2
    for keyword in sars2_keywords:
        if keyword in title:
            return "sars2"
        
        for tax_name in taxonomy_names:
            if keyword in tax_name:
                return "sars2"
    
    # Check all coronavirus types from the defined dictionary (more strict match)
    for cov_type, info in CORONAVIRUS_TYPES.items():
        # Check all search terms for this type
        for term in info["search_terms"]:
            term_lower = term.lower()
            
            # Check title
            if term_lower in title:
                return cov_type
            
            # Check taxonomy
            for tax_name in taxonomy_names:
                if term_lower in tax_name:
                    return cov_type
    
    # Even more relaxed checks for protease structures that might be similar
    # Check if it's a coronavirus structure at all
    is_coronavirus = any("coronavirus" in name for name in taxonomy_names) or "coronavirus" in title
    
    # Check if it's any kind of SARS (could include SARS-CoV-1 or variants)
    is_sars = "sars" in title or any("sars" in name for name in taxonomy_names)
    
    # Check if it looks like a protease
    is_protease = ("protease" in title or 
                  "mpro" in title or 
                  "3clpro" in title or 
                  "main protease" in title)
    
    if is_coronavirus and is_protease:
        if is_sars:
            return "sars2"  # Classify any SARS-related coronavirus protease as SARS-CoV-2 for simplicity
        else:
            return "similar"  # Other coronavirus proteases go to similar
    
    # For any other protease that could be similar but not explicitly coronavirus
    if is_protease:
        return "similar"
    
    return "similar"  # Default to similar for all structures that made it through the search

def process_structures_in_batches(pdb_ids, batch_size=50):
    """
    Process structures in batches to reduce load and handle errors gracefully.
    
    Args:
        pdb_ids (list): List of PDB IDs
        batch_size (int): Number of structures to process in each batch
        
    Returns:
        dict: Dictionary mapping PDB IDs to structure info
    """
    pdb_info = {}
    total_batches = (len(pdb_ids) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(pdb_ids))
        batch = pdb_ids[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_num+1}/{total_batches} ({len(batch)} structures)")
        
        for pdb_id in tqdm(batch, desc="Fetching structure information"):
            try:
                info = get_structure_info(pdb_id)
                pdb_info[pdb_id] = info
            except Exception as e:
                print(f"Error processing {pdb_id}: {e}")
                # Continue with next structure despite errors
        
        # Save progress after each batch
        print(f"Completed batch {batch_num+1}/{total_batches}")
    
    return pdb_info

def download_coronavirus_proteases(output_base_dir, include_assemblies=False, 
                                   include_types=None, keywords=None, 
                                   resolution_cutoff=None, experimental_method=None,
                                   include_similar=True, max_similar=100,
                                   max_workers=4, batch_size=50):
    """
    Download coronavirus protease structures.
    
    Args:
        output_base_dir (str): Base directory to save the files
        include_assemblies (bool): Whether to download biological assemblies
        include_types (list): List of coronavirus types to include
        keywords (list): List of protease keywords to search for
        resolution_cutoff (float): Maximum resolution in Angstroms
        experimental_method (str): Experimental method filter
        include_similar (bool): Whether to include similar protease-inhibitor complexes
        max_similar (int): Maximum number of similar complexes to include
        max_workers (int): Maximum number of concurrent downloads
        batch_size (int): Number of structures to process in each batch
        
    Returns:
        dict: Dictionary mapping coronavirus types to lists of downloaded files
    """
    if include_types is None:
        include_types = ["mers", "sars2"]
    
    if keywords is None:
        keywords = PROTEASE_KEYWORDS
    
    # Add extra keywords to cast a wider net
    extra_keywords = ["3c like", "papain like", "replication", "viral", "virus"]
    search_keywords = keywords + extra_keywords
    
    # Create output directories for each type
    directories = {
        "mers": os.path.join(output_base_dir, CORONAVIRUS_TYPES["mers"]["dir_name"]),
        "sars2": os.path.join(output_base_dir, CORONAVIRUS_TYPES["sars2"]["dir_name"]),
        "similar": os.path.join(output_base_dir, "similar_protease_complexes")
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Initialize results
    all_files = {"mers": [], "sars2": [], "similar": []}
    all_pdb_ids = set()
    
    # Search for structures for each coronavirus type
    for cov_type in include_types:
        print(f"\n=== Searching for {CORONAVIRUS_TYPES[cov_type]['name']} structures ===")
        search_terms = CORONAVIRUS_TYPES[cov_type]["search_terms"]
        
        # Search for PDB IDs
        pdb_ids = search_pdb_structures(
            search_terms, 
            search_keywords,
            resolution_cutoff,
            experimental_method
        )
        all_pdb_ids.update(pdb_ids)
        
        print(f"Found {len(pdb_ids)} {CORONAVIRUS_TYPES[cov_type]['name']} structures")
    
    # Also search for general coronavirus structures
    print("\n=== Searching for general coronavirus protease structures ===")
    general_corona_terms = ["Coronavirus", "Human coronavirus", "Bat coronavirus", "SARS coronavirus"]
    general_pdb_ids = search_pdb_structures(
        general_corona_terms,
        search_keywords,
        resolution_cutoff,
        experimental_method
    )
    all_pdb_ids.update(general_pdb_ids)
    print(f"Found {len(general_pdb_ids)} general coronavirus structures")
    
    # Search for similar complexes if requested
    similar_pdb_ids = []
    if include_similar:
        similar_pdb_ids = search_similar_complexes(
            resolution_cutoff,
            experimental_method,
            max_similar
        )
        all_pdb_ids.update(similar_pdb_ids)
    
    total_structures = len(all_pdb_ids)
    print(f"\n=== Processing {total_structures} total unique structures ===")
    
    # Process structures in batches
    pdb_info = process_structures_in_batches(list(all_pdb_ids), batch_size)
    
    # Classify structures by type
    classified_ids = {"mers": [], "sars2": [], "similar": []}
    
    for pdb_id in all_pdb_ids:
        cov_type = classify_coronavirus_type(pdb_id, pdb_info.get(pdb_id))
        classified_ids[cov_type].append(pdb_id)
    
    # Output classification results
    print("\n=== Classification Results ===")
    print(f"Classified {len(classified_ids['mers'])} structures as MERS-CoV")
    print(f"Classified {len(classified_ids['sars2'])} structures as SARS-CoV-2")
    print(f"Classified {len(classified_ids['similar'])} structures as similar complexes")
    
    # Download classified structures
    download_types = ["mers", "sars2"]
    if include_similar:
        download_types.append("similar")
    
    # Set max structures to download per type to avoid overwhelming
    max_structures_per_type = 200
    for cov_type in download_types:
        pdb_ids = classified_ids[cov_type]
        
        # Limit number of structures if too many
        if len(pdb_ids) > max_structures_per_type:
            print(f"Limiting {cov_type} structures from {len(pdb_ids)} to {max_structures_per_type}")
            # Keeping a subset for efficiency
            pdb_ids = pdb_ids[:max_structures_per_type]
        
        if not pdb_ids:
            continue
            
        type_name = CORONAVIRUS_TYPES.get(cov_type, {"name": "Similar Complex"})["name"]
        print(f"\n=== Downloading {len(pdb_ids)} {type_name} structures ===")
        output_dir = directories[cov_type]
        
        # Download in smaller batches to avoid overwhelming the API
        batch_size = min(20, len(pdb_ids))
        for i in range(0, len(pdb_ids), batch_size):
            batch = pdb_ids[i:i+batch_size]
            print(f"Downloading batch {i//batch_size + 1}/{(len(pdb_ids) + batch_size - 1)//batch_size}")
            
            # Use ThreadPoolExecutor for parallel downloads within each batch
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit asymmetric unit downloads
                future_to_pdb = {
                    executor.submit(download_pdb_file, pdb_id, output_dir): pdb_id
                    for pdb_id in batch
                }
                
                # Process results as they complete
                for future in as_completed(future_to_pdb):
                    pdb_id = future_to_pdb[future]
                    try:
                        pdb_file = future.result()
                        if pdb_file:
                            all_files[cov_type].append(pdb_file)
                    except Exception as e:
                        print(f"Download failed for {pdb_id}: {e}")
                
                # If requested, download biological assemblies
                if include_assemblies:
                    # Get assembly IDs for each PDB
                    assembly_futures = {}
                    for pdb_id in batch:
                        # Submit assembly list retrieval
                        future = executor.submit(get_biological_assemblies, pdb_id)
                        assembly_futures[future] = pdb_id
                    
                    # Collect assembly IDs
                    pdb_to_assemblies = {}
                    for future in as_completed(assembly_futures):
                        pdb_id = assembly_futures[future]
                        try:
                            assembly_ids = future.result()
                            if assembly_ids:
                                pdb_to_assemblies[pdb_id] = assembly_ids
                        except Exception as e:
                            print(f"Failed to get assemblies for {pdb_id}: {e}")
                    
                    # Download assembly files
                    assembly_download_futures = {}
                    for pdb_id, assembly_ids in pdb_to_assemblies.items():
                        for assembly_id in assembly_ids:
                            future = executor.submit(download_pdb_file, pdb_id, output_dir, assembly_id)
                            assembly_download_futures[future] = (pdb_id, assembly_id)
                    
                    # Process assembly download results
                    for future in as_completed(assembly_download_futures):
                        pdb_id, assembly_id = assembly_download_futures[future]
                        try:
                            assembly_file = future.result()
                            if assembly_file:
                                all_files[cov_type].append(assembly_file)
                        except Exception as e:
                            print(f"Assembly download failed for {pdb_id} assembly {assembly_id}: {e}")
    
    return all_files

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Download coronavirus main protease (Mpro) structures from RCSB PDB."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/ist-nas/users/bunditb/boltz/scripts/examples/coronavirus_pdb",
        help="Base directory to save the structures"
    )
    parser.add_argument(
        "--include-assemblies", 
        action="store_true",
        help="Download biological assemblies in addition to asymmetric units"
    )
    parser.add_argument(
        "--resolution-cutoff",
        type=float,
        default=3.5,
        help="Maximum resolution in Angstroms (lower is better)"
    )
    parser.add_argument(
        "--experimental-method",
        type=str,
        choices=list(EXPERIMENTAL_METHODS.keys()),
        default=None,  # Changed to None to include all methods by default
        help="Experimental method filter (default: all methods)"
    )
    parser.add_argument(
        "--include-similar",
        action="store_true",
        default=True,  # Changed to True by default
        help="Include similar protease-inhibitor complexes"
    )
    parser.add_argument(
        "--max-similar",
        type=int,
        default=200,
        help="Maximum number of similar complexes to include"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of concurrent downloads"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of structures to process in each batch"
    )
    
    args = parser.parse_args()
    
    print(f"Output directory: {args.output_dir}")
    print(f"Including biological assemblies: {args.include_assemblies}")
    print(f"Resolution cutoff: {args.resolution_cutoff} Å")
    print(f"Experimental method: {args.experimental_method}")
    print(f"Include similar protease complexes: {args.include_similar}")
    print(f"Max workers: {args.max_workers}")
    print(f"Batch size: {args.batch_size}")
    
    try:
        # Download structures
        all_files = download_coronavirus_proteases(
            args.output_dir,
            args.include_assemblies,
            ["mers", "sars2"],
            PROTEASE_KEYWORDS,
            args.resolution_cutoff,
            args.experimental_method,
            args.include_similar,
            args.max_similar,
            args.max_workers,
            args.batch_size
        )
        
        # Print summary
        print("\n=== Download Summary ===")
        total_files = 0
        
        print(f"Downloaded {len(all_files['mers'])} MERS-CoV files")
        total_files += len(all_files['mers'])
        
        print(f"Downloaded {len(all_files['sars2'])} SARS-CoV-2 files")
        total_files += len(all_files['sars2'])
        
        if args.include_similar:
            print(f"Downloaded {len(all_files['similar'])} similar complex files")
            total_files += len(all_files['similar'])
        
        print(f"Total files: {total_files}")
        print(f"Files are saved in:")
        print(f"  MERS-CoV: {os.path.join(args.output_dir, CORONAVIRUS_TYPES['mers']['dir_name'])}")
        print(f"  SARS-CoV-2: {os.path.join(args.output_dir, CORONAVIRUS_TYPES['sars2']['dir_name'])}")
        
        if args.include_similar:
            print(f"  Similar complexes: {os.path.join(args.output_dir, 'similar_protease_complexes')}")
            
    except KeyboardInterrupt:
        print("\nDownload interrupted by user. Partial results may have been saved.")
    except Exception as e:
        print(f"\nError during download: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 