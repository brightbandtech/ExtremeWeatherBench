#!/usr/bin/env python3
"""
Clean version of GHCN-H streaming download script.
This is a redirect to the optimized streaming version.
"""

import sys
from pathlib import Path

# Add the current directory to the path to import the streaming version
sys.path.insert(0, str(Path(__file__).parent))

try:
    from generate_ghcnh_streaming import main as streaming_main

    def main():
        """Main function that uses the clean streaming implementation."""
        print("🔄 Redirecting to optimized streaming version...")
        print("📁 Using: generate_ghcnh_streaming.py")
        print()
        streaming_main()

    def main_station_list_approach():
        """Legacy compatibility function."""
        print("🔄 Redirecting to optimized streaming implementation...")
        streaming_main()

    # Re-export all important functions from streaming version
    from generate_ghcnh_streaming import (  # noqa: F401
        check_station_already_processed,
        clear_existing_data_cache,
        construct_station_download_url,
        download_all_stations_streaming,
        download_station_list,
    )

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"❌ Error importing streaming version: {e}")
    print("📄 Please use: python data_prep/generate_ghcnh_streaming.py")
    sys.exit(1)
