import json
import sys
from pathlib import Path

# --- Setup: Allow imports from parent directory ---
# This ensures the script can find sim_v1_2.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

# --- Imports from your actual project files ---
from sim_v1_2 import SimulationEngine, DummyModel, ResultAggregator, MatchResult
from sim_eval.loaders import TestMatchLoader

# --- Configuration: Tell the script which match to test ---
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# IMPORTANT: Change this path to a real JSON file from your test data directory
# Example: '../data/betting_test/1381220.json' or 'data/betting_test/1381220.json'
TEST_MATCH_JSON_PATH = "data/betting_test/1415709.json"
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


def run_test():
    """
    Loads a single real match, simulates it once, and tests the aggregator.
    """
    print("--- Starting Single Match Test ---")

    # 1. Initialize the Simulation Engine
    # We use DummyModel to avoid needing the trained XGBoost model for this test.
    # This isolates the problem to the simulation logic itself.
    print("Initializing simulation engine with DummyModel...")
    engine = SimulationEngine(model=DummyModel())
    
    # 2. Load the single test match data
    match_file = Path(TEST_MATCH_JSON_PATH)
    if not match_file.exists():
        print(f"❌ ERROR: Test match file not found at '{match_file}'")
        print("Please update the TEST_MATCH_JSON_PATH variable in this script.")
        return

    print(f"Loading match data from: {match_file.name}")
    try:
        with open(match_file, 'r') as f:
            match_data = json.load(f)
        
        loader = TestMatchLoader()
        # Use the loader's internal method to create the initial match state
        match_id, initial_state = loader._create_match_state(match_data)
        if not initial_state:
            print("❌ ERROR: Failed to create match state from JSON.")
            return
    except Exception as e:
        print(f"❌ ERROR: Could not load or parse the JSON file: {e}")
        return

    # 3. Run a SINGLE simulation
    print(f"\nRunning one simulation for match: {match_id}...")
    single_result: MatchResult = engine.simulate_match(initial_state, match_id)
    print("Simulation complete.")

    # 4. CRITICAL STEP: Inspect the data types of the result
    print("\n--- Inspecting Simulation Output ---")
    
    t1_score = single_result.team1_score
    t1_wickets = single_result.team1_wickets
    t2_score = single_result.team2_score
    t2_wickets = single_result.team2_wickets

    print(f"Team 1 Score:    {t1_score:<5} (Type: {type(t1_score).__name__})")
    print(f"Team 1 Wickets:  {t1_wickets:<5} (Type: {type(t1_wickets).__name__})")
    print(f"Team 2 Score:    {t2_score:<5} (Type: {type(t2_score).__name__})")
    print(f"Team 2 Wickets:  {t2_wickets:<5} (Type: {type(t2_wickets).__name__})")

    # 5. Test the Aggregator with the single, real result
    print("\n--- Testing the ResultAggregator ---")
    try:
        # We put the single result into a list, as the aggregator expects
        results_list = [single_result]
        ResultAggregator.aggregate(results_list)
        print("✅ SUCCESS: ResultAggregator.aggregate ran without errors.")
    except Exception as e:
        print(f"❌ FAILURE: ResultAggregator.aggregate failed!")
        print(f"   ERROR TYPE:    {type(e).__name__}")
        print(f"   ERROR MESSAGE: {e}")


if __name__ == "__main__":
    run_test()