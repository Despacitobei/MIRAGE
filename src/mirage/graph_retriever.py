# GRAG.py - Graph Retrieval for AI Guidance
import os
import json
import re
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import time
import torch
from sklearn.metrics.pairwise import cosine_similarity
import atexit
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_config

_retriever_instance = None

class GraphRetriever:
    """
    A class for retrieving information from Neo4j knowledge graph.
    Supports path queries between entities and neighbor queries for single entities.
    """
    
    @classmethod
    def get_instance(cls, uri, username, password):
        """
        Get singleton instance of GraphRetriever to avoid reloading models and embeddings
        
        Args:
            uri (str): Neo4j database URI
            username (str): Username
            password (str): Password
            
        Returns:
            GraphRetriever: Singleton instance
        """
        global _retriever_instance
        if _retriever_instance is None:
            _retriever_instance = cls(uri, username, password)
        return _retriever_instance
    
    def __init__(self, uri, username, password):
        """Initialize GraphRetriever class."""
        config = get_config()
        
        self.uri = uri
        self.username = username
        self.password = password
        
        self.driver = GraphDatabase.driver(
            uri, 
            auth=(username, password)
        )
        

        self.model_loaded = False
        self.embeddings_loaded = False
        
        self.model = None
        self.entity_data = None
        self.entity_names = None
        self.entity_embeddings = None
        
        self.model_path = config.paths.sentence_transformer_path
        self.entity_embeddings_path = config.paths.entity_embeddings_path
    
    def _safe_session(self):
        """
        Get a safe Neo4j session. Automatically reconnects if driver is invalid.
        
        Returns:
            Session: A valid Neo4j session
        """
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            print(f"[Warning] Neo4j connection lost or stale: {e}")
            print("[Action] Reconnecting to Neo4j...")
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                    connection_timeout=10,
                    max_connection_lifetime=300,
                    max_connection_pool_size=50,
                    keep_alive=True
                )
                print("[Success] Reconnected to Neo4j.")
            except Exception as re:
                print(f"[Error] Failed to reconnect: {re}")
                raise re
        
        return self.driver.session()
    
    def _load_model(self):
        """Lazy load Sentence Transformer model"""
        if not self.model_loaded:
            print("Loading Sentence Transformer model...")
            self.model = SentenceTransformer(self.model_path)
            self.model_loaded = True
            print("Model loaded.")
    
    def _load_embeddings(self):
        """Lazy load entity embeddings"""
        if not self.embeddings_loaded:
            print("Loading entity embeddings...")
            with open(self.entity_embeddings_path, 'rb') as f:
                self.entity_data = pickle.load(f)
            
            # Convert to numpy arrays for efficient matching
            self.entity_names = self.entity_data["entities"]
            self.entity_embeddings = np.array(self.entity_data["embeddings"])
            self.embeddings_loaded = True
            print("Entity embeddings loaded.")
        
    def close(self):
        """Close Neo4j connection and release resources"""
        if self.driver:
            self.driver.close()
            self.driver = None
        
        self.model = None
        self.entity_data = None
        self.entity_embeddings = None
        self.entity_names = None
        self.model_loaded = False
        self.embeddings_loaded = False
        
        global _retriever_instance
        _retriever_instance = None
        
    def match_entity(self, query_entity: str, top_k: int = 3, threshold: float = None) -> List[Dict]:
        """
        Match a query entity to the most similar entities in the knowledge graph.
        
        Args:
            query_entity (str): The entity name to match
            top_k (int): Number of top matches to return
            threshold (float): Minimum similarity threshold (0.0-1.0), if None use config value
            
        Returns:
            List[Dict]: List of matched entities with similarity scores
        """
        if threshold is None:
            config = get_config()
            threshold = config.search.entity_match_threshold
        
        self._load_model()
        self._load_embeddings()
        
        # Generate embedding for the query entity
        query_embedding = self.model.encode([query_entity])[0]
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.entity_embeddings)[0]
        
        # Get top-k matches
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        matches = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                matches.append({
                    "entity": self.entity_names[idx],
                    "similarity": float(similarities[idx]),
                    "entity_id": idx
                })
            
        return matches
        
    def graph_search_path(self, entity1: str, entity2: str, max_hops: int = 3, limit: int = 5, threshold: float = None) -> List[Dict]:
        """
        Find all shortest paths between two entities.
        
        Args:
            entity1 (str): Source entity name
            entity2 (str): Target entity name
            max_hops (int): Maximum hop limit (currently hardcoded in query)
            limit (int): Maximum number of paths to return
            threshold (float): Minimum similarity threshold for entity matching, if None use config value
            
        Returns:
            List[Dict]: List of dictionaries containing path information
        """
        if threshold is None:
            config = get_config()
            threshold = config.search.entity_match_threshold
        
        with self._safe_session() as session:
            entity1_matches = self.match_entity(entity1, threshold=threshold)
            entity2_matches = self.match_entity(entity2, threshold=threshold)
            
            if not entity1_matches:
                print(f"No matches found for '{entity1}' that meet the similarity threshold ({threshold})")
                return []
            
            if not entity2_matches:
                print(f"No matches found for '{entity2}' that meet the similarity threshold ({threshold})")
                return []
                
            matched_entity1 = self._clean_entity_name(entity1_matches[0]["entity"])
            matched_entity2 = self._clean_entity_name(entity2_matches[0]["entity"])
            
            query = (
                f"MATCH path = shortestPath((start:Entity {{name: $entity1}})-[*1..3]->(end:Entity {{name: $entity2}})) "
                f"RETURN path "
                f"LIMIT $limit"
            )
            
            # Try query
            try:
                result = session.run(
                    query,
                    entity1=matched_entity1,
                    entity2=matched_entity2,
                    limit=limit
                )
                
                # Process results
                paths = []
                for record in result:
                    path = record["path"]
                    path_data = self._extract_path_data(path)
                    paths.append(path_data)
                
                if not paths:
                    fallback_query = (
                        f"MATCH path = allShortestPaths((start:Entity {{name: $entity1}})-[*1..3]->(end:Entity {{name: $entity2}})) "
                        f"RETURN path "
                        f"LIMIT $limit"
                    )
                    try:
                        result = session.run(
                            fallback_query,
                            entity1=matched_entity1,
                            entity2=matched_entity2,
                            limit=limit
                        )
                        for record in result:
                            path = record["path"]
                            path_data = self._extract_path_data(path)
                            paths.append(path_data)
                    except Exception as e:
                        # If allShortestPaths fails, ignore
                        pass
                
                return paths
                
            except Exception as e:
                print(f"Error in graph_search_path: {str(e)}")
                return []
    
    def graph_search_neighbors(self, entity: str, 
                            relation_filter: Optional[str] = None, 
                            limit: int = 10,
                            threshold: float = None) -> Dict[str, List[Dict]]:
        """
        Find all neighbor relationships of an entity, grouped by relationship type.
        
        Args:
            entity (str): Entity name
            relation_filter (str, optional): Relationship type filter
            limit (int): Maximum number of neighbors to return per relationship type
            threshold (float): Minimum similarity threshold for entity matching, if None use config value
            
        Returns:
            Dict[str, List[Dict]]: Neighbors grouped by relationship type
        """
        if threshold is None:
            config = get_config()
            threshold = config.search.entity_match_threshold
        
        with self._safe_session() as session:
            entity_matches = self.match_entity(entity, threshold=threshold)
            
            if not entity_matches:
                print(f"No matches found for '{entity}' that meet the similarity threshold ({threshold})")
                return {}
                
            # Use the best match
            matched_entity = self._clean_entity_name(entity_matches[0]["entity"])
            
            # Build Cypher query
            relation_clause = ""
            if relation_filter:
                relation_clause = f"WHERE type(r) = '{relation_filter}' "
                
            query = (
                f"MATCH (e:Entity {{name: $entity}})-[r]->(n) "
                f"{relation_clause}"
                f"RETURN type(r) AS relation, collect(n.name) AS neighbors "
                f"ORDER BY relation"
            )
            
            try:
                result = session.run(query, entity=matched_entity)
                
                grouped_neighbors = {}
                
                for record in result:
                    relation = record["relation"].replace("_", " ")
                    neighbors = record["neighbors"]
                    
                    if len(neighbors) > limit:
                        neighbors = neighbors[:limit]
                    
                    neighbor_data = []
                    for neighbor in neighbors:
                        display_name = neighbor.replace("_", " ")
                        neighbor_data.append({
                            "name": neighbor,
                            "display_name": display_name
                        })
                    
                    if neighbor_data:
                        grouped_neighbors[relation] = neighbor_data
                
                return grouped_neighbors
                
            except Exception as e:
                print(f"Error in graph_search_neighbors: {str(e)}")
                return {}
    
    def search_graph(self, query: str) -> Dict:
        """
        Execute knowledge graph search, supporting entity pairs or single entity queries.
        
        Args:
            query (str): Query string in format "entity1||entity2" or "entity1"
            
        Returns:
            Dict: Query results
        """
        start_time = time.time()
        
        # Parse query
        entity1, entity2 = self._parse_query(query)
        
        # Prepare result dictionary
        result = {
            "query": query,
            "parsed": {
                "entity1": entity1,
                "entity2": entity2
            },
            "search_type": "path" if entity2 else "neighbors",
            "results": None,
            "time_taken": None,
            "error": None,
            "matched_entities": [],
            "low_similarity_match": False
        }
        
        try:
            config = get_config()
            threshold = config.search.entity_match_threshold
            
            entity1_matches = self.match_entity(entity1, threshold=threshold)
            result["matched_entities"].append({
                "query": entity1,
                "matches": entity1_matches
            })
            
            if not entity1_matches:
                result["low_similarity_match"] = True
                result["error"] = f"No matches found for '{entity1}' that meet the similarity threshold ({threshold}). Please try a different entity."
                result["time_taken"] = time.time() - start_time
                return result
            
            if entity2:
                entity2_matches = self.match_entity(entity2, threshold=threshold)
                result["matched_entities"].append({
                    "query": entity2,
                    "matches": entity2_matches
                })
                
                if not entity2_matches:
                    result["low_similarity_match"] = True
                    result["error"] = f"No matches found for '{entity2}' that meet the similarity threshold ({threshold}). Please try a different entity."
                    result["time_taken"] = time.time() - start_time
                    return result
            
            # Execute appropriate search based on query type
            if entity2:
                # Path query
                paths = self.graph_search_path(entity1, entity2, threshold=threshold)
                result["results"] = {
                    "paths": paths,
                    "count": len(paths)
                }
            else:
                # Neighbor query
                neighbors = self.graph_search_neighbors(entity1, threshold=threshold)
                result["results"] = {
                    "neighbors": neighbors,
                    "relation_count": len(neighbors),
                    "total_neighbors": sum(len(v) for v in neighbors.values())
                }
                
        except Exception as e:
            result["error"] = str(e)
            
        # Record execution time
        result["time_taken"] = time.time() - start_time
        
        return result

    def format_graph_results(self, result: Dict) -> str:
        """
        Format graph query results into user-friendly text.
        
        Args:
            result (Dict): Graph query results
            
        Returns:
            str: Formatted text results
        """
        if result.get("low_similarity_match"):
            return result['error']
        
        if result.get("error"):
            return f"Query Error: {result['error']}"
            
        query_type = result["search_type"]
        
        orig_entity1 = result["parsed"]["entity1"].replace("_", " ")
        orig_entity2 = result["parsed"]["entity2"].replace("_", " ") if result["parsed"]["entity2"] else None
        
        matched_entity1 = orig_entity1
        matched_entity2 = orig_entity2
        
        if result.get("matched_entities") and len(result["matched_entities"]) > 0:
            if result["matched_entities"][0].get("matches") and len(result["matched_entities"][0]["matches"]) > 0:
                matched_entity1 = result["matched_entities"][0]["matches"][0]["entity"]
            
            if len(result["matched_entities"]) > 1 and result["matched_entities"][1].get("matches") and len(result["matched_entities"][1]["matches"]) > 0:
                matched_entity2 = result["matched_entities"][1]["matches"][0]["entity"]
        
        if result.get("matched_entities"):
            for match_result in result["matched_entities"]:
                if not match_result["matches"]:
                    return f"No matches found for '{match_result['query']}'"
        
        if query_type == "path":
            return self._format_path_results(matched_entity1, matched_entity2, result["results"]["paths"])
        else:
            return self._format_neighbor_results(matched_entity1, result["results"]["neighbors"])
    
    def _format_path_results(self, entity1: str, entity2: str, paths: List[Dict]) -> str:
        """Format path query results into natural language"""
        if not paths:
            return f"No paths found from '{entity1}' to '{entity2}'."
            
        formatted = ""
        
        for path in paths:
            nodes = path.get("entities", [])
            edges = path.get("relations", [])
            
            if not nodes or not edges:
                continue
                
            current_entity = nodes[0]["name"]
            for i, edge in enumerate(edges):
                next_entity = nodes[i+1]["name"]
                relation = edge["type"]
                
                normalized_relation = relation.replace(" ", "_")
                
                if normalized_relation == "has_symptom":
                    formatted += f"{current_entity} has symptom {next_entity}.\n"
                elif normalized_relation == "need_medical_test":
                    formatted += f"For {current_entity}, {next_entity} is required to confirm diagnosis.\n"
                elif normalized_relation == "need_medication":
                    formatted += f"{current_entity} needs {next_entity}.\n"
                elif normalized_relation == "possible_cure_disease":
                    formatted += f"{current_entity} may be cured by {next_entity}.\n"
                elif normalized_relation == "can_check_disease":
                    formatted += f"{current_entity} can be checked by {next_entity}.\n"
                elif normalized_relation == "possible_disease":
                    formatted += f"{next_entity} may indicate {current_entity}.\n"
                
                current_entity = next_entity
                
        return formatted if formatted else f"No valid paths found from '{entity1}' to '{entity2}'."
    
    def _format_neighbor_results(self, entity: str, neighbors: Dict[str, List[Dict]]) -> str:
        """Format neighbor query results into natural language"""
        if not neighbors:
            return f"No neighbors found for '{entity}'."
            
        formatted = ""
        
        for relation_type, neighbor_list in neighbors.items():
            neighbor_names = [n["display_name"] for n in neighbor_list]
            
            normalized_relation = relation_type.replace(" ", "_")
            
            if normalized_relation == "has_symptom":
                formatted += f"{entity} has symptoms {', '.join(neighbor_names)}.\n"
            elif normalized_relation == "need_medical_test":
                formatted += f"For {entity}, {', '.join(neighbor_names)} are required to confirm diagnosis.\n"
            elif normalized_relation == "need_medication":
                formatted += f"{entity} needs {', '.join(neighbor_names)}.\n"
            elif normalized_relation == "possible_cure_disease":
                formatted += f"{entity} may be cured by {', '.join(neighbor_names)}.\n"
            elif normalized_relation == "can_check_disease":
                formatted += f"{entity} can be checked by {', '.join(neighbor_names)}.\n"
            elif normalized_relation == "possible_disease":
                formatted += f"{', '.join(neighbor_names)} may indicate {entity}.\n"
                
        return formatted if formatted else f"No valid neighbors found for '{entity}'."
    
    def _extract_path_data(self, path) -> Dict:
        """Extract path data from Neo4j path object"""
        entities = []
        relations = []
        
        for i, node in enumerate(path.nodes):
            entity_name = node["name"]
            entity_display = entity_name.replace("_", " ")
            entities.append({
                "name": entity_name,
                "display_name": entity_display
            })
            
            if i < len(path.relationships):
                rel = path.relationships[i]
                rel_type = rel.type
                rel_display = rel_type.replace("_", " ")
                
                start_node_id = rel.start_node.id
                current_node_id = node.id
                direction = "outgoing" if start_node_id == current_node_id else "incoming"
                
                relations.append({
                    "type": rel_type,
                    "display_type": rel_display,
                    "direction": direction
                })
        
        return {
            "entities": entities,
            "relations": relations,
            "length": len(relations)
        }
    
    def _parse_query(self, query: str) -> Tuple[str, Optional[str]]:
        """Parse query string, extract entity pair"""
        if "||" in query:
            parts = query.split("||", 1)
            return parts[0].strip(), parts[1].strip() if parts[1].strip() else None
        elif "|" in query:
            parts = query.split("|", 1)
            return parts[0].strip(), parts[1].strip() if parts[1].strip() else None
        else:
            return query.strip(), None
    
    def _clean_entity_name(self, name: str) -> str:
        """Clean entity name, replace spaces with underscores etc."""
        if not name:
            return ""
        # Replace spaces with underscores to conform to Neo4j naming conventions
        return name.replace(" ", "_")


def graph_search(query, neo4j_uri, neo4j_user, neo4j_password):
    """Execute knowledge graph query and return formatted results."""
    retriever = GraphRetriever.get_instance(neo4j_uri, neo4j_user, neo4j_password)
    try:
        result = retriever.search_graph(query)
        return result
    except Exception as e:
        print(f"Serious error in graph_search: {str(e)}")
        retriever.close()
        raise e

def format_graph_results(result):
    """
    Format knowledge graph query results.
    
    Args:
        result (Dict): Knowledge graph query results
        
    Returns:
        str: Formatted text
    """
    if _retriever_instance:
        return _retriever_instance.format_graph_results(result)
    else:
        retriever = GraphRetriever("", "", "")
        return retriever.format_graph_results(result)

def cleanup():
    """Clean up resources on program exit"""
    global _retriever_instance
    if _retriever_instance:
        print("Cleaning up GRAG resources...")
        _retriever_instance.close()
        _retriever_instance = None
        print("GRAG resources cleaned up.")

atexit.register(cleanup)


if __name__ == "__main__":
    config = get_config()
    
    uri = config.neo4j.uri
    username = config.neo4j.user
    password = config.neo4j.password
    
    test_queries = [
        "diabetes||insulin",
        "headache",
        "COVID-19||fever"
    ]
    
    for query in test_queries:
        print(f"\n>>> Query: {query}")
        result = graph_search(query, uri, username, password)
        
        print(f"Query type: {result['search_type']}")
        print(f"Time taken: {result['time_taken']:.3f} seconds")
        
        formatted = format_graph_results(result)
        print("\n" + formatted)
