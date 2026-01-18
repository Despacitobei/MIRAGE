import pandas as pd
from neo4j import GraphDatabase
import argparse
import os
import time

def build_neo4j_kg_from_triples(uri, username, password, triples_file, clear_existing=True, 
                                 batch_size=50000, start_from_line=0):
    """Build Neo4j knowledge graph from triples file (chatdoctor5k format)"""
    
    print("Building Neo4j knowledge graph...")
    print(f"Config: batch_size={batch_size}")
    if start_from_line > 0:
        print(f"🔄 Resume mode: starting from line {start_from_line}")
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    try:
        with driver.session() as session:
            if clear_existing and start_from_line == 0:
                print("Clearing existing database...")
                session.run("MATCH (n) DETACH DELETE n")
            elif start_from_line > 0:
                print("⚡ Resuming from previous progress, keeping existing data...")
            
            print(f"Loading triples from file: {triples_file}")
            
            df = pd.read_csv(triples_file, sep='\t', header=None, 
                           names=['entity1', 'relation', 'entity2'])
            
            if start_from_line > 0:
                original_len = len(df)
                df = df.iloc[start_from_line:].reset_index(drop=True)
                print(f"📊 Original file: {original_len:,} lines, starting from line {start_from_line}: {len(df):,} lines remaining")
            
            df = df.dropna(subset=['entity1', 'entity2', 'relation'])
            df = df[df['entity1'].astype(str).str.strip() != '']
            df = df[df['entity2'].astype(str).str.strip() != '']
            df = df[df['relation'].astype(str).str.strip() != '']
            
            print(f"Loaded {len(df):,} valid triples")
            
            print("Optimizing Neo4j settings...")
            try:
                session.run("CALL dbms.queryTimeout(0)")
                print("Query timeout disabled")
            except Exception as e:
                print(f"Warning: Failed to disable query timeout (non-critical): {str(e)}")
            
            print("Creating constraints and indexes...")
            try:
                session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
                print("Entity name constraint created")
            except Exception as e:
                print(f"Warning: Failed to create constraint (may already exist): {str(e)}")
                
            try:
                session.run("CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)")
                print("Entity name index created")
            except Exception as e:
                print(f"Warning: Failed to create index (may already exist): {str(e)}")
            
            print("Using batch import method...")
            _build_kg_directly(session, df, batch_size)
            
            progress_file = f"{triples_file}.progress"
            with open(progress_file, 'w') as f:
                f.write("COMPLETED")
            print(f"✅ Progress saved: {progress_file}")
            
            print("\nFinal database statistics:")
            result = session.run("MATCH (n:Entity) RETURN count(n) as node_count")
            node_count = result.single()["node_count"]
            print(f"Entity nodes: {node_count:,}")
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = result.single()["rel_count"]
            print(f"Relationship edges: {rel_count:,}")
            
            result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC LIMIT 10")
            print("Top 10 relationship types:")
            for record in result:
                print(f"  {record['rel_type']}: {record['count']:,}")
            
            print("Knowledge graph construction completed!")
            
    except Exception as e:
        print(f"Build failed: {e}")
        raise
    finally:
        driver.close()

def _build_kg_directly(session, df, batch_size):
    """Build knowledge graph directly in batches without temporary files"""
    
    print("Extracting entities...")
    all_entities = set()
    all_entities.update(df['entity1'].astype(str).str.strip())
    all_entities.update(df['entity2'].astype(str).str.strip())
    all_entities = {e for e in all_entities if e and e.strip()}
    
    print(f"Importing {len(all_entities):,} entities...")
    entities_list = list(all_entities)
    
    batch_size_entities = 10000
    for i in range(0, len(entities_list), batch_size_entities):
        batch = entities_list[i:i + batch_size_entities]
        query = """
        UNWIND $entities AS entity_name
        MERGE (e:Entity {name: entity_name})
        """
        session.run(query, entities=batch)
        print(f"Entity batch {i//batch_size_entities + 1}/{(len(entities_list)-1)//batch_size_entities + 1} completed")
    
    print(f"✅ Entity import completed: {len(all_entities):,} entities")
    
    print("Processing relationships by type...")
    relation_types = df['relation'].unique()
    
    for rel_type in relation_types:
        if not rel_type or str(rel_type).strip() == '':
            continue
            
        rel_df = df[df['relation'] == rel_type]
        relation_name = str(rel_type).strip().replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '').replace(')', '').replace(';', '_')
        
        print(f"Processing {len(rel_df):,} relationships of type '{rel_type}'...")
        
        rel_batch_size = 5000
        relationships = []
        for _, row in rel_df.iterrows():
            head = str(row['entity1']).strip()
            tail = str(row['entity2']).strip()
            if head and tail:
                relationships.append({'head': head, 'tail': tail})
        
        for i in range(0, len(relationships), rel_batch_size):
            batch = relationships[i:i + rel_batch_size]
            query = f"""
            UNWIND $relationships AS rel
            MATCH (h:Entity {{name: rel.head}})
            MATCH (t:Entity {{name: rel.tail}})
            MERGE (h)-[r:{relation_name}]->(t)
            """
            session.run(query, relationships=batch)
            if (i // rel_batch_size + 1) % 10 == 0:
                print(f"  Progress: {i + len(batch):,}/{len(relationships):,} relationships")
        
        print(f"✅ Relationship type '{rel_type}' import completed: {len(relationships):,} relationships")

def main():
    """Main function (for direct execution of builder.py)
    
    Note: It's recommended to use build_kg.py in the project root instead.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from config import get_config
    
    config = get_config()
    
    parser = argparse.ArgumentParser(description='Build Neo4j knowledge graph from triples file (chatdoctor5k format)')
    parser.add_argument('--triples_file', default=None, 
                       help='Path to triples file (default: from config.py)')
    parser.add_argument('--batch_size', type=int, default=50000,
                       help='Batch size (default: 50000)')
    parser.add_argument('--no_clear', action='store_true', 
                       help='Do not clear existing database')
    parser.add_argument('--start_from_line', type=int, default=0,
                       help='Start from specific line (for resume)')
    parser.add_argument('--force_restart', action='store_true',
                       help='Force restart from beginning, ignore progress file')
    
    args = parser.parse_args()
    
    triples_file = args.triples_file or config.paths.triples_file
    
    if not os.path.exists(triples_file):
        print(f"Triples file not found: {triples_file}")
        return
    
    progress_file = f"{triples_file}.progress"
    start_line = args.start_from_line
    
    if not args.force_restart and os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress_content = f.read().strip()
        
        if progress_content == "COMPLETED":
            print(f"✅ Knowledge graph construction already completed: {triples_file}")
            print("Use --force_restart to rebuild from scratch")
            return
        elif progress_content.isdigit():
            last_line = int(progress_content)
            if start_line == 0:
                start_line = last_line
                print(f"🔄 Found progress file: resuming from line {start_line}")
            else:
                print(f"📋 Progress file shows line {last_line}, but using user-specified line {start_line}")
    
    start_time = time.time()
    
    build_neo4j_kg_from_triples(
        uri=config.neo4j.uri,
        username=config.neo4j.user, 
        password=config.neo4j.password,
        triples_file=triples_file,
        clear_existing=not args.no_clear and start_line == 0,
        batch_size=args.batch_size,
        start_from_line=start_line
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()

