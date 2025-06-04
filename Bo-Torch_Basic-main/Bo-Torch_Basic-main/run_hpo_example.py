#!/usr/bin/env python3
"""
Quick example to run TabPFN HPO study with reduced parameters for testing.
For the full study, modify HPO.py directly and run main().
"""

import sys
sys.path.append('.')

from HPO import TabPFNHPO, main
import time

def run_quick_hpo_example():
    """Run a quick HPO example with reduced parameters for testing"""
    
    print("🚀 Quick TabPFN HPO Example")
    print("=" * 50)
    
    # Create HPO instance
    hpo = TabPFNHPO()
    
    # Reduce parameters for quick testing
    print("\n📝 Reducing parameters for quick example:")
    original_ensembles = hpo.ensemble_sizes.copy()
    original_temps = hpo.temperatures.copy()
    
    # Test only a few configurations
    hpo.ensemble_sizes = [8, 16, 32]  # Reduced from [8, 16, 32, 64, 128]
    hpo.temperatures = [0.5, 1.0, 1.5]  # Reduced from full range
    
    print(f"  Ensemble sizes: {hpo.ensemble_sizes} (was {original_ensembles})")
    print(f"  Temperatures: {hpo.temperatures} (was {original_temps})")
    print(f"  Total configs: {len(hpo.ensemble_sizes) * len(hpo.temperatures)} (was {len(original_ensembles) * len(original_temps)})")
    
    # Test only a subset of functions for speed
    original_functions = hpo.bbob_functions.copy()
    hpo.bbob_functions = {
        'f1_sphere': hpo.bbob_functions['f1_sphere'],
        'f2_ellipsoid': hpo.bbob_functions['f2_ellipsoid'], 
        'f15_rastrigin': hpo.bbob_functions['f15_rastrigin']
    }
    print(f"  BBOB functions: {list(hpo.bbob_functions.keys())} (was {list(original_functions.keys())})")
    print(f"  BBOB IDs: {list(hpo.bbob_functions.values())}")
    
    # Run 2D grid search
    print(f"\n🔍 Starting reduced 2D grid search...")
    start_time = time.time()
    
    hpo.run_grid_search_2d()
    
    elapsed = time.time() - start_time
    print(f"\n✅ Quick example completed in {elapsed:.1f} seconds")
    
    # Show results
    hpo.print_summary()
    
    print(f"\n💡 To run the full study:")
    print(f"   python HPO.py")
    print(f"   Or call main() from HPO module")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Quick example (reduced parameters)")
    print("2. Full HPO study")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_quick_hpo_example()
    elif choice == "2":
        print("Running full HPO study...")
        main()
    else:
        print("Invalid choice. Running quick example...")
        run_quick_hpo_example() 