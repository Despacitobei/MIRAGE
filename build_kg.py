#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build knowledge graph from triples file"""

import argparse
import os
from src.kg_builder.builder import build_neo4j_kg_from_triples
from config import get_config

def main():
    """Main function to build Neo4j knowledge graph
    
    Usage:
        python build_kg.py                          # Use default config
        python build_kg.py --triples custom.txt     # Use custom triples file
    """
    parser = argparse.ArgumentParser(
        description='Build Neo4j knowledge graph from triples file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_kg.py                                    # Build with default settings
  python build_kg.py --triples data/custom.txt          # Use custom triples file
  python build_kg.py --keep-existing                    # Keep existing data
  python build_kg.py --start-from-line 1000             # Resume from line 1000
        """
    )
    parser.add_argument('--triples', type=str, default=None,
                       help='Triples file path (default: from config.py)')
    parser.add_argument('--uri', type=str, default=None,
                       help='Neo4j URI (default: from config.py)')
    parser.add_argument('--user', type=str, default=None,
                       help='Neo4j username (default: from config.py)')
    parser.add_argument('--password', type=str, default=None,
                       help='Neo4j password (default: from config.py)')
    parser.add_argument('--batch-size', type=int, default=50000,
                       help='Batch size (default: 50000)')
    parser.add_argument('--start-from-line', type=int, default=0,
                       help='Resume from line N (default: 0)')
    parser.add_argument('--keep-existing', action='store_true',
                       help='Keep existing data (do not clear database)')
    
    args = parser.parse_args()
    config = get_config()
    
    triples_file = args.triples or config.paths.triples_file
    uri = args.uri or config.neo4j.uri
    user = args.user or config.neo4j.user
    password = args.password or config.neo4j.password
    
    if not os.path.exists(triples_file):
        print(f"❌ Error: Triples file not found: {triples_file}")
        print(f"Please check the path in config.py or specify with --triples")
        return
    
    print(f"📁 Using triples file: {triples_file}")
    print(f"🔗 Neo4j URI: {uri}")
    print(f"👤 Neo4j User: {user}")
    print()
    
    build_neo4j_kg_from_triples(
        uri=uri,
        username=user,
        password=password,
        triples_file=triples_file,
        clear_existing=not args.keep_existing,
        batch_size=args.batch_size,
        start_from_line=args.start_from_line
    )

if __name__ == "__main__":
    main()

