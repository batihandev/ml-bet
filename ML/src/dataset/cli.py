from .builder import load_raw_matches, normalize_columns, filter_matches, save_processed

def run_build_dataset_process():
    print("Loading raw matches...")
    df_raw = load_raw_matches()
    print(f"Raw shape: {df_raw.shape}")

    print("Normalizing columns and types...")
    df_clean = normalize_columns(df_raw)
    
    print("Filtering matches...")
    df_filtered = filter_matches(df_clean)
    print(f"Filtered shape: {df_filtered.shape}")

    print("Saving processed dataset...")
    save_processed(df_filtered)

if __name__ == "__main__":
    run_build_dataset_process()
