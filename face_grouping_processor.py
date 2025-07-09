import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import gc
import multiprocessing as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_similarities(args):
    """Compute similarities for a face against cluster representatives."""
    face_embedding, cluster_embeddings, cluster_ids = args
    similarities = cosine_similarity([face_embedding], cluster_embeddings)[0]
    return similarities, cluster_ids


class StreamingFaceGrouper:
    def __init__(
        self,
        results_file: str = "estonian_song_festival_faces.json",
        output_db: str = "face_groups.db",
        similarity_threshold: float = 0.6,
        batch_size: int = 1000,  # Smaller batches for more frequent progress
        max_memory_mb: float = 8000,
        num_workers: int = None,
    ):
        self.results_file = Path(results_file)
        self.output_db = output_db
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.num_workers = num_workers or min(mp.cpu_count(), 16)  # Use up to 16 cores

        # In-memory storage for current processing
        self.face_clusters = {}
        self.face_to_cluster = {}
        self.cluster_representatives = {}
        self.next_cluster_id = 0

        self.setup_database()

    def setup_database(self):
        """Create database tables for storing face groups."""
        conn = sqlite3.connect(self.output_db)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_clusters (
                cluster_id INTEGER PRIMARY KEY,
                representative_embedding BLOB,
                face_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_assignments (
                face_id TEXT PRIMARY KEY,
                cluster_id INTEGER,
                frame_number INTEGER,
                chunk_number INTEGER,
                timestamp TEXT,
                confidence REAL,
                thumbnail_path TEXT,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                FOREIGN KEY (cluster_id) REFERENCES face_clusters (cluster_id)
            )
        """)

        conn.commit()
        conn.close()

    def check_memory_usage(self) -> float:
        """Check current memory usage in MB."""
        import resource

        memory_mb = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )  # On Linux, ru_maxrss is in KB
        return memory_mb

    def parse_faces_manually(self, file_path: Path):
        """Manually parse JSON file to extract faces without loading everything."""
        logger.info(f"Starting manual parsing of {file_path}")

        with open(file_path, "r") as f:
            # Skip to the faces array
            content = f.read(1024)
            while '"faces": [' not in content:
                chunk = f.read(1024)
                if not chunk:
                    raise ValueError("Could not find faces array in JSON")
                content += chunk

            # Find the start of the faces array
            faces_start = content.find('"faces": [') + len('"faces": [')
            f.seek(faces_start)

            # Now read face objects one by one
            brace_count = 0
            current_face = ""
            in_face = False
            face_count = 0

            while True:
                char = f.read(1)
                if not char:
                    break

                if char == "{":
                    brace_count += 1
                    in_face = True
                    current_face += char
                elif char == "}":
                    brace_count -= 1
                    current_face += char

                    if brace_count == 0 and in_face:
                        # Complete face object
                        try:
                            face_data = json.loads(current_face)
                            if face_data.get("embedding"):
                                face_count += 1
                                if face_count % 1000 == 0:
                                    memory_mb = self.check_memory_usage()
                                    logger.info(
                                        f"Processed {face_count} faces, Memory: {memory_mb:.1f}MB"
                                    )
                                yield face_data
                        except json.JSONDecodeError:
                            pass

                        current_face = ""
                        in_face = False
                elif in_face:
                    current_face += char

    def extract_face_data(self, face: Dict[str, Any]) -> Dict[str, Any]:
        """Extract essential face data for processing."""
        return {
            "face_id": f"f_{face['frame_number']}_{face['face_id']}",
            "embedding": np.array(face["embedding"], dtype=np.float32),
            "frame_number": face["frame_number"],
            "chunk_number": face["chunk_number"],
            "timestamp": face["timestamp"],
            "confidence": face["confidence"],
            "thumbnail_path": face.get("thumbnail_path"),
            "bbox": face["bbox"],
        }

    def find_matching_cluster_batch(self, face_embeddings):
        if not self.cluster_representatives:
            return [None] * len(face_embeddings)

        # Use numpy for faster operations
        cluster_matrix = np.array(list(self.cluster_representatives.values()))
        face_matrix = np.array(face_embeddings)

        # Vectorized cosine similarity
        similarities = np.dot(face_matrix, cluster_matrix.T) / (
            np.linalg.norm(face_matrix, axis=1, keepdims=True)
            * np.linalg.norm(cluster_matrix, axis=1)
        )

        matches = []
        for sim_row in similarities:
            best_idx = np.argmax(sim_row)
            if sim_row[best_idx] >= self.similarity_threshold:
                cluster_ids = list(self.cluster_representatives.keys())
                matches.append(cluster_ids[best_idx])
            else:
                matches.append(None)

        return matches

    def create_new_cluster(self, face_data: Dict[str, Any]) -> int:
        """Create a new cluster for a face."""
        cluster_id = self.next_cluster_id
        self.next_cluster_id += 1

        self.face_clusters[cluster_id] = []
        self.cluster_representatives[cluster_id] = face_data["embedding"].copy()

        return cluster_id

    def add_face_to_cluster(self, face_data: Dict[str, Any], cluster_id: int):
        """Add a face to an existing cluster."""
        face_id = face_data["face_id"]

        self.face_clusters[cluster_id].append(face_data)
        self.face_to_cluster[face_id] = cluster_id

        # Update cluster representative (simple averaging)
        current_faces = self.face_clusters[cluster_id]
        if len(current_faces) > 1:
            embeddings = [f["embedding"] for f in current_faces]
            self.cluster_representatives[cluster_id] = np.mean(embeddings, axis=0)

    def process_face_batch(self, faces: List[Dict[str, Any]], cursor):
        """Process a batch of faces using vectorized operations."""
        logger.info(f"Processing batch of {len(faces)} faces...")

        # Extract embeddings for batch processing
        embeddings = [f["embedding"] for f in faces]

        # Find matches using vectorized operations
        matches = self.find_matching_cluster_batch(embeddings)

        # Process each face
        for face_data, matching_cluster in zip(faces, matches):
            if matching_cluster is not None:
                self.add_face_to_cluster(face_data, matching_cluster)
            else:
                cluster_id = self.create_new_cluster(face_data)
                self.add_face_to_cluster(face_data, cluster_id)

        self.save_batch_to_db(faces, cursor)
        logger.info(f"Batch complete. Total clusters: {len(self.face_clusters)}")

    def save_batch_to_db(self, faces: List[Dict[str, Any]], cursor):
        """Save batch results to database."""
        for face_data in faces:
            cluster_id = self.face_to_cluster[face_data["face_id"]]

            cursor.execute(
                """
                INSERT OR REPLACE INTO face_assignments 
                (face_id, cluster_id, frame_number, chunk_number, timestamp, 
                 confidence, thumbnail_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    face_data["face_id"],
                    cluster_id,
                    face_data["frame_number"],
                    face_data["chunk_number"],
                    face_data["timestamp"],
                    face_data["confidence"],
                    face_data["thumbnail_path"],
                    face_data["bbox"]["x1"],
                    face_data["bbox"]["y1"],
                    face_data["bbox"]["x2"],
                    face_data["bbox"]["y2"],
                ),
            )

        # Update cluster info
        for cluster_id, faces_in_cluster in self.face_clusters.items():
            cursor.execute(
                """
                INSERT OR REPLACE INTO face_clusters 
                (cluster_id, representative_embedding, face_count)
                VALUES (?, ?, ?)
            """,
                (
                    cluster_id,
                    self.cluster_representatives[cluster_id].tobytes(),
                    len(faces_in_cluster),
                ),
            )

    def process_streaming(self):
        """Process the massive JSON file in a streaming fashion."""
        logger.info("Starting streaming face grouping process...")

        conn = sqlite3.connect(self.output_db)
        cursor = conn.cursor()

        face_buffer = []
        total_faces_processed = 0

        try:
            for face in self.parse_faces_manually(self.results_file):
                face_data = self.extract_face_data(face)
                face_buffer.append(face_data)

                if len(face_buffer) >= self.batch_size:
                    self.process_face_batch(face_buffer, cursor)
                    total_faces_processed += len(face_buffer)
                    face_buffer.clear()

                    conn.commit()
                    gc.collect()

                    memory_mb = self.check_memory_usage()
                    logger.info(
                        f"Processed {total_faces_processed} faces, {len(self.face_clusters)} clusters, Memory: {memory_mb:.1f}MB"
                    )

                    if memory_mb > self.max_memory_mb:
                        logger.warning("Memory usage high, forcing cleanup")
                        # Keep only cluster representatives
                        for cluster_id in list(self.face_clusters.keys()):
                            self.face_clusters[cluster_id] = self.face_clusters[
                                cluster_id
                            ][:1]
                        gc.collect()

            # Process final batch
            if face_buffer:
                self.process_face_batch(face_buffer, cursor)
                total_faces_processed += len(face_buffer)
                conn.commit()

            logger.info("Streaming processing complete!")
            logger.info(f"Total faces processed: {total_faces_processed}")
            logger.info(f"Total clusters created: {len(self.face_clusters)}")

        finally:
            conn.close()


def main():
    """Main function to run the streaming face grouper."""
    grouper = StreamingFaceGrouper(
        results_file="estonian_song_festival_faces.json",
        similarity_threshold=0.7,  # Higher threshold = fewer clusters = faster
        batch_size=1000,  # Smaller batches for more frequent progress
        max_memory_mb=16000,  # Use more of your 32GB RAM
        num_workers=16,  # Use more cores
    )

    grouper.process_streaming()


if __name__ == "__main__":
    main()
