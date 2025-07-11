import sqlite3
import numpy as np
import logging
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SingleCoreMerger:
    def __init__(self, db_path="face_groups.db", similarity_threshold=0.8):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.total_merged = 0
        self.total_iterations = 0

    def get_largest_unprocessed_cluster(self):
        """Get the largest cluster, prioritizing unprocessed ones."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT cluster_id, representative_embedding, face_count
                       FROM face_clusters
                       WHERE representative_embedding IS NOT NULL
                         AND face_count > 1
                       ORDER BY merge_attempted ASC, face_count DESC
                       LIMIT 1
                       """)

        result = cursor.fetchone()
        conn.close()

        if not result:
            return None

        cluster_id, embedding_bytes, face_count = result
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

        return cluster_id, embedding, face_count

    def find_similar_clusters(self, target_embedding, target_cluster_id):
        """Find all clusters similar to the target cluster."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
                       SELECT cluster_id, representative_embedding, face_count
                       FROM face_clusters
                       WHERE cluster_id != ?
                         AND representative_embedding IS NOT NULL
                         AND face_count > 0
                       """,
            (target_cluster_id,),
        )

        similar_clusters = []
        total_checked = 0

        for cluster_id, embedding_bytes, face_count in cursor.fetchall():
            if embedding_bytes:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                similarity = np.dot(target_embedding, embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding)
                )

                if similarity >= self.similarity_threshold:
                    similar_clusters.append((cluster_id, embedding, face_count))

                total_checked += 1

                if total_checked % 10000 == 0:
                    print(
                        f"    Checked {total_checked} clusters, found {len(similar_clusters)} similar"
                    )

        conn.close()
        return similar_clusters

    def mark_cluster_processed(self, cluster_id):
        """Mark cluster as processed with current timestamp."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        current_time = datetime.now().isoformat()
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

    def merge_clusters_into_target(
        self, target_cluster_id, target_embedding, target_face_count, clusters_to_merge
    ):
        """Merge clusters into target cluster."""
        if not clusters_to_merge:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get embeddings and face counts for clusters to merge
        merge_embeddings = [target_embedding]
        merge_face_counts = [target_face_count]
        total_faces_added = 0

        for cluster_id, embedding, face_count in clusters_to_merge:
            merge_embeddings.append(embedding)
            merge_face_counts.append(face_count)
            total_faces_added += face_count

        # Calculate weighted average of all embeddings
        weighted_sum = np.zeros_like(target_embedding)
        total_weight = 0

        for embedding, weight in zip(merge_embeddings, merge_face_counts):
            weighted_sum += embedding * weight
            total_weight += weight

        new_representative = weighted_sum / total_weight

        # Update face assignments
        for cluster_id, _, _ in clusters_to_merge:
            cursor.execute(
                """
                           UPDATE face_assignments
                           SET cluster_id = ?
                           WHERE cluster_id = ?
                           """,
                (target_cluster_id, cluster_id),
            )

        # Update target cluster
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
        for cluster_id, _, _ in clusters_to_merge:
            cursor.execute(
                "DELETE FROM face_clusters WHERE cluster_id = ?", (cluster_id,)
            )

        conn.commit()
        conn.close()

        return total_faces_added

    def get_stats(self):
        """Get current database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM face_clusters")
        total_clusters = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM face_clusters WHERE merge_attempted IS NULL"
        )
        unprocessed_count = cursor.fetchone()[0]

        cursor.execute(
            "SELECT MIN(face_count), MAX(face_count), AVG(face_count) FROM face_clusters"
        )
        min_size, max_size, avg_size = cursor.fetchone()

        conn.close()
        return (
            total_clusters,
            unprocessed_count,
            min_size or 0,
            max_size or 0,
            avg_size or 0,
        )

    def run_single_core_merge(self):
        """Run the single-core merge process."""
        logger.info("🚀 Starting single-core merger...")
        start_time = time.time()

        while True:
            # Get stats
            total_clusters, unprocessed_count, min_size, max_size, avg_size = (
                self.get_stats()
            )

            # Get largest unprocessed cluster
            target_cluster = self.get_largest_unprocessed_cluster()
            if not target_cluster:
                logger.info("✅ No more clusters to process!")
                break

            target_cluster_id, target_embedding, target_face_count = target_cluster

            logger.info(
                f"🎯 Processing cluster {target_cluster_id} ({target_face_count} faces)"
            )
            logger.info(
                f"📊 Stats: {total_clusters} clusters ({unprocessed_count} unprocessed), sizes: {min_size}-{max_size}, avg: {avg_size:.1f}"
            )

            # Find similar clusters
            similar_clusters = self.find_similar_clusters(
                target_embedding, target_cluster_id
            )

            # Mark as processed regardless of whether we found matches
            self.mark_cluster_processed(target_cluster_id)

            if similar_clusters:
                total_faces_in_similar = sum(c[2] for c in similar_clusters)
                logger.info(
                    f"🔍 Found {len(similar_clusters)} similar clusters with {total_faces_in_similar} faces"
                )

                # Merge them
                faces_added = self.merge_clusters_into_target(
                    target_cluster_id,
                    target_embedding,
                    target_face_count,
                    similar_clusters,
                )

                self.total_merged += len(similar_clusters)
                logger.info(
                    f"✅ Merged {len(similar_clusters)} clusters, added {faces_added} faces to cluster {target_cluster_id}"
                )
            else:
                logger.info("ℹ️  No similar clusters found")

            self.total_iterations += 1

            if self.total_iterations % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"🕐 Processed {self.total_iterations} iterations in {elapsed:.1f}s, merged {self.total_merged} clusters total"
                )

        elapsed = time.time() - start_time
        logger.info(
            f"🎉 Merge complete! Processed {self.total_iterations} iterations in {elapsed:.1f}s"
        )
        logger.info(f"📊 Total clusters merged: {self.total_merged}")


def main():
    merger = SingleCoreMerger(db_path="face_groups.db", similarity_threshold=0.8)

    merger.run_single_core_merge()


if __name__ == "__main__":
    main()
