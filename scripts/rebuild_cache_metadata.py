#!/usr/bin/env python3
"""
Rebuild cache metadata to include date lists per chunk for lazy loading
"""

import pickle
from pathlib import Path

def rebuild_metadata():
    """Load each chunk and extract its date list"""

    models_dir = Path('models')
    old_metadata_path = models_dir / 'player_stats_cache_metadata.pkl'

    print("Loading existing metadata...")
    with open(old_metadata_path, 'rb') as f:
        old_metadata = pickle.load(f)

    print(f"Found {old_metadata['num_chunks']} chunks to process")

    # Build new metadata with date lists
    chunks_with_dates = []

    for i, chunk_file_rel in enumerate(old_metadata['chunk_files']):
        chunk_path = models_dir / chunk_file_rel
        print(f"Processing chunk {i+1}/{old_metadata['num_chunks']}: {chunk_file_rel}...", end=' ')

        # Load chunk and extract dates
        with open(chunk_path, 'rb') as f:
            chunk_data = pickle.load(f)

        dates = sorted(chunk_data.keys())
        print(f"✓ ({len(dates)} dates)")

        chunks_with_dates.append({
            'file': str(chunk_file_rel),
            'dates': dates,
            'num_dates': len(dates)
        })

        # Free memory
        del chunk_data

    # Create new metadata
    new_metadata = {
        **old_metadata,  # Keep all old fields
        'chunks': chunks_with_dates  # Add new chunks field with dates
    }

    # Save new metadata
    print("\nSaving updated metadata...")
    with open(old_metadata_path, 'wb') as f:
        pickle.dump(new_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✓ Metadata updated successfully!")
    print(f"  Total chunks: {len(chunks_with_dates)}")
    print(f"  Total dates: {sum(c['num_dates'] for c in chunks_with_dates)}")

    # Show sample
    print(f"\nSample chunk info:")
    print(f"  Chunk 0: {chunks_with_dates[0]['num_dates']} dates")
    print(f"    First date: {chunks_with_dates[0]['dates'][0]}")
    print(f"    Last date: {chunks_with_dates[0]['dates'][-1]}")

if __name__ == "__main__":
    rebuild_metadata()
