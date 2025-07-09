import sqlite3
import numpy as np
import logging
import time
from typing import List, Tuple
from datetime import datetime
import gc

try:
    import cupy as cp
    from cupy import cuda

    GPU_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ GPU (CuPy) available")
except ImportError:
    GPU_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️  GPU not available, falling back to CPU")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("small_cluster_merger.log"), logging.StreamHandler()],
)


class SmallClusterMerger:
    def __init__(
        self,
        db_path: str = "face_groups.db",
        similarity_threshold: float = 0.8,
        batch_size: int = 1000,
    ):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.total_merged = 0
        self.total_faces_merged = 0

    def get_smallest_clusters(self, n: int) -> List[Tuple[int, np.ndarray, int]]:
        """Get the N smallest clusters, prioritizing unprocessed ones."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # First try to get unprocessed clusters
        cursor.execute(
            """
                       SELECT cluster_id, representative_embedding, face_count
                       FROM face_clusters
                       WHERE representative_embedding IS NOT NULL
                         AND face_count >= 1
                         AND merge_attempted IS NULL
                       ORDER BY face_count ASC, cluster_id ASC
                       LIMIT ?
                       """,
            (n,),
        )

        clusters = []
        for cluster_id, embedding_bytes, face_count in cursor.fetchall():
            if embedding_bytes:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                clusters.append((cluster_id, embedding, face_count))

        # If we don't have enough unprocessed clusters, fill with oldest processed ones
        if len(clusters) < n:
            remaining = n - len(clusters)
            processed_cluster_ids = [str(c[0]) for c in clusters] if clusters else []

            exclude_clause = ""
            if processed_cluster_ids:
                exclude_clause = (
                    f"AND cluster_id NOT IN ({','.join(processed_cluster_ids)})"
                )

            cursor.execute(
                f"""
                SELECT cluster_id, representative_embedding, face_count 
                FROM face_clusters 
                WHERE representative_embedding IS NOT NULL 
                AND face_count >= 1
                AND merge_attempted IS NOT NULL
                {exclude_clause}
                ORDER BY merge_attempted ASC, face_count ASC, cluster_id ASC
                LIMIT ?
            """,
                (remaining,),
            )

            for cluster_id, embedding_bytes, face_count in cursor.fetchall():
                if embedding_bytes:
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    clusters.append((cluster_id, embedding, face_count))

        conn.close()

        unprocessed_count = sum(1 for c in clusters if self._is_unprocessed(c[0]))
        logger.info(
            f"📊 Selected {len(clusters)} clusters ({unprocessed_count} unprocessed, {len(clusters) - unprocessed_count} previously processed)"
        )

        return clusters

    def _is_unprocessed(self, cluster_id: int) -> bool:
        """Check if cluster has been processed before."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT merge_attempted FROM face_clusters WHERE cluster_id = ?",
            (cluster_id,),
        )
        result = cursor.fetchone()
        conn.close()
        return result and result[0] is None

    def mark_clusters_as_processed(self, cluster_ids: List[int]):
        """Mark clusters as having been processed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        current_time = datetime.now().isoformat()

        for cluster_id in cluster_ids:
            cursor.execute(
                """
                           UPDATE face_clusters
                           SET merge_attempted = ?
                           WHERE cluster_id = ?
                           """,
                (current_time, cluster_id),
            )

        conn.commit()
        conn.close()

        logger.info(f"📝 Marked {len(cluster_ids)} clusters as processed")

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute full similarity matrix using GPU if available."""
        logger.info(
            f"🧮 Computing {len(embeddings)}x{len(embeddings)} similarity matrix"
        )

        if GPU_AVAILABLE:
            try:
                # GPU computation
                gpu_embeddings = cp.array(embeddings)

                # Compute dot product matrix
                gpu_similarities = cp.dot(gpu_embeddings, gpu_embeddings.T)

                # Normalize to get cosine similarity
                norms = cp.linalg.norm(gpu_embeddings, axis=1)
                gpu_similarities = gpu_similarities / (
                    norms[:, cp.newaxis] * norms[cp.newaxis, :]
                )

                # Transfer back to CPU
                similarities = cp.asnumpy(gpu_similarities)

                # Clean up GPU memory
                del gpu_embeddings, gpu_similarities
                cuda.Device().synchronize()

                logger.info("✅ GPU similarity computation complete")
                return similarities

            except Exception as e:
                logger.warning(f"⚠️  GPU computation failed: {e}, falling back to CPU")

        # CPU computation
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(embeddings)
        logger.info("✅ CPU similarity computation complete")
        return similarities

    def find_merge_groups(
        self, similarity_matrix: np.ndarray, cluster_ids: List[int]
    ) -> List[List[int]]:
        """Find connected components of similar clusters."""
        n = len(cluster_ids)

        # Create adjacency matrix
        adjacency = similarity_matrix >= self.similarity_threshold
        np.fill_diagonal(adjacency, False)  # Remove self-connections

        # Find connected components using DFS
        visited = set()
        groups = []

        def dfs(node, component):
            if node in visited:
                return
            visited.add(node)
            component.append(cluster_ids[node])

            for neighbor in range(n):
                if adjacency[node, neighbor] and neighbor not in visited:
                    dfs(neighbor, component)

        for i in range(n):
            if i not in visited:
                component = []
                dfs(i, component)
                if len(component) > 1:  # Only keep groups with multiple clusters
                    groups.append(component)

        return groups

    def merge_clusters_into_target(
        self, target_cluster_id: int, clusters_to_merge: List[int]
    ):
        """Merge clusters into target and update database with proper representative."""
        if not clusters_to_merge:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current target cluster info
        cursor.execute(
            """
                       SELECT representative_embedding, face_count
                       FROM face_clusters
                       WHERE cluster_id = ?
                       """,
            (target_cluster_id,),
        )

        target_result = cursor.fetchone()
        if not target_result:
            logger.error(f"Target cluster {target_cluster_id} not found!")
            return

        target_embedding_bytes, target_face_count = target_result
        target_embedding = np.frombuffer(target_embedding_bytes, dtype=np.float32)

        # Get representative embeddings and face counts for clusters to merge
        merge_embeddings = []
        merge_face_counts = []
        total_faces_added = 0

        for cluster_id in clusters_to_merge:
            cursor.execute(
                """
                           SELECT representative_embedding, face_count
                           FROM face_clusters
                           WHERE cluster_id = ?
                           """,
                (cluster_id,),
            )

            result = cursor.fetchone()
            if result and result[0]:
                embedding_bytes, face_count = result
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                merge_embeddings.append(embedding)
                merge_face_counts.append(face_count)
                total_faces_added += face_count

        # Calculate weighted average of all representative embeddings
        all_embeddings = [target_embedding] + merge_embeddings
        all_weights = [target_face_count] + merge_face_counts

        # Weighted average: sum(embedding * weight) / sum(weights)
        weighted_sum = np.zeros_like(target_embedding)
        total_weight = 0

        for embedding, weight in zip(all_embeddings, all_weights):
            weighted_sum += embedding * weight
            total_weight += weight

        new_representative = weighted_sum / total_weight

        # Update face assignments
        for cluster_id in clusters_to_merge:
            cursor.execute(
                """
                           UPDATE face_assignments
                           SET cluster_id = ?
                           WHERE cluster_id = ?
                           """,
                (target_cluster_id, cluster_id),
            )

        # Update target cluster with new representative and face count
        cursor.execute(
            """
                       UPDATE face_clusters
                       SET representative_embedding = ?,
                           face_count               = ?
                       WHERE cluster_id = ?
                       """,
            (new_representative.tobytes(), total_weight, target_cluster_id),
        )

        # Delete merged clusters
        for cluster_id in clusters_to_merge:
            cursor.execute(
                "DELETE FROM face_clusters WHERE cluster_id = ?", (cluster_id,)
            )

        conn.commit()
        conn.close()

        self.total_merged += len(clusters_to_merge)
        self.total_faces_merged += total_faces_added

        logger.info(
            f"✅ Merged {len(clusters_to_merge)} clusters into {target_cluster_id}, new face count: {total_weight}"
        )

    def get_cluster_stats(self) -> Tuple[int, int, int, float]:
        """Get current cluster statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM face_clusters")
        total_clusters = cursor.fetchone()[0]

        cursor.execute(
            "SELECT MIN(face_count), MAX(face_count), AVG(face_count) FROM face_clusters"
        )
        min_size, max_size, avg_size = cursor.fetchone()

        # Get processing stats
        cursor.execute(
            "SELECT COUNT(*) FROM face_clusters WHERE merge_attempted IS NULL"
        )
        unprocessed_count = cursor.fetchone()[0]

        conn.close()

        logger.info(f"📊 Unprocessed clusters: {unprocessed_count:,}")

        return total_clusters, min_size or 0, max_size or 0, avg_size or 0

    def run_small_cluster_merge(self):
        """Run the small cluster merge process."""
        logger.info("🚀 Starting small cluster merge...")
        start_time = time.time()

        # Get initial stats
        total_clusters, min_size, max_size, avg_size = self.get_cluster_stats()
        logger.info(f"📊 Initial stats: {total_clusters:,} clusters")
        logger.info(f"📊 Size range: {min_size} - {max_size}, avg: {avg_size:.1f}")

        # Get smallest clusters (prioritizing unprocessed ones)
        smallest_clusters = self.get_smallest_clusters(self.batch_size)

        if len(smallest_clusters) < 2:
            logger.info("❌ Not enough clusters to process")
            return

        logger.info(f"🎯 Processing {len(smallest_clusters)} smallest clusters")

        # Show size distribution
        sizes = [c[2] for c in smallest_clusters]
        logger.info(f"📊 Size distribution: {min(sizes)} - {max(sizes)}")

        # Extract data
        cluster_ids = [c[0] for c in smallest_clusters]
        embeddings = np.array([c[1] for c in smallest_clusters])
        face_counts = [c[2] for c in smallest_clusters]

        logger.info(
            f"📊 Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}"
        )

        # Compute similarity matrix
        matrix_start = time.time()
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        matrix_time = time.time() - matrix_start

        logger.info(f"⏱️  Similarity matrix computed in {matrix_time:.2f}s")

        # Find similar pairs count
        similar_pairs = np.sum(similarity_matrix >= self.similarity_threshold) - len(
            cluster_ids
        )  # Exclude diagonal
        logger.info(
            f"🔍 Found {similar_pairs} similar pairs at threshold {self.similarity_threshold}"
        )

        # Find merge groups
        merge_groups = self.find_merge_groups(similarity_matrix, cluster_ids)

        # Mark all processed clusters (even if no merges happened)
        self.mark_clusters_as_processed(cluster_ids)

        if not merge_groups:
            logger.info("ℹ️  No merge groups found, but clusters marked as processed")
            return

        logger.info(f"🎯 Found {len(merge_groups)} merge groups")

        # Show group statistics
        group_sizes = [len(group) for group in merge_groups]
        total_clusters_in_groups = sum(group_sizes)
        logger.info(f"📊 Group sizes: {min(group_sizes)} - {max(group_sizes)}")
        logger.info(f"📊 Total clusters to be merged: {total_clusters_in_groups}")

        # Execute merges
        merge_start = time.time()
        for i, group in enumerate(merge_groups):
            if len(group) <= 1:
                continue

            # Sort by face count (descending) to make largest the target
            group_with_counts = [
                (cluster_id, face_counts[cluster_ids.index(cluster_id)])
                for cluster_id in group
            ]
            group_with_counts.sort(key=lambda x: x[1], reverse=True)

            target_id = group_with_counts[0][0]
            to_merge = [x[0] for x in group_with_counts[1:]]

            total_faces_in_group = sum(x[1] for x in group_with_counts)

            logger.info(
                f"🔄 Merging group {i + 1}/{len(merge_groups)}: {len(group)} clusters, {total_faces_in_group} faces → cluster {target_id}"
            )

            self.merge_clusters_into_target(target_id, to_merge)

        merge_time = time.time() - merge_start
        total_time = time.time() - start_time

        # Final stats
        final_total_clusters, final_min_size, final_max_size, final_avg_size = (
            self.get_cluster_stats()
        )

        logger.info("🎉 MERGE COMPLETE!")
        logger.info(
            f"📊 Clusters: {total_clusters:,} → {final_total_clusters:,} (reduced by {total_clusters - final_total_clusters:,})"
        )
        logger.info(f"📊 Total merged: {self.total_merged:,} clusters")
        logger.info(f"📊 Faces merged: {self.total_faces_merged:,} faces")
        logger.info(f"⏱️  Matrix computation: {matrix_time:.2f}s")
        logger.info(f"⏱️  Merge execution: {merge_time:.2f}s")
        logger.info(f"⏱️  Total time: {total_time:.2f}s")

        # Memory cleanup
        gc.collect()
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()


def main():
    for i in range(0, 77):
        merger = SmallClusterMerger(
            db_path="face_groups.db", similarity_threshold=0.8, batch_size=10000
        )

        merger.run_small_cluster_merge()


if __name__ == "__main__":
    main()
