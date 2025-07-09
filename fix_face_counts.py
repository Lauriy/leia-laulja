import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_face_counts(db_path: str = "face_groups.db"):
    """Fix the face_count field in face_clusters table by counting actual assignments."""
    logger.info("Fixing face counts in database...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get actual face counts from face_assignments
    logger.info("Counting actual faces per cluster...")
    cursor.execute("""
        SELECT cluster_id, COUNT(*) as actual_count
        FROM face_assignments 
        GROUP BY cluster_id
    """)

    actual_counts = dict(cursor.fetchall())
    logger.info(f"Found {len(actual_counts)} clusters with faces")

    # Update face_clusters table
    logger.info("Updating face_clusters table...")
    updated = 0

    for cluster_id, actual_count in actual_counts.items():
        cursor.execute(
            """
            UPDATE face_clusters 
            SET face_count = ?
            WHERE cluster_id = ?
        """,
            (actual_count, cluster_id),
        )

        if cursor.rowcount > 0:
            updated += 1

    # Remove clusters with no faces
    # cursor.execute('''
    #     DELETE FROM face_clusters
    #     WHERE cluster_id NOT IN (
    #         SELECT DISTINCT cluster_id FROM face_assignments
    #     )
    # ''')
    #
    # removed = cursor.rowcount

    conn.commit()
    conn.close()

    logger.info(f"Updated {updated} clusters")
    # logger.info(f"Removed {removed} empty clusters")
    logger.info("Face count fix complete!")


def show_stats(db_path: str = "face_groups.db"):
    """Show clustering statistics after fixing counts."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM face_clusters")
    total_clusters = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM face_assignments")
    total_faces = cursor.fetchone()[0]

    cursor.execute("SELECT MAX(face_count) FROM face_clusters")
    max_cluster_size = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(face_count) FROM face_clusters")
    avg_cluster_size = cursor.fetchone()[0]

    cursor.execute("""
        SELECT face_count, COUNT(*) as cluster_count
        FROM face_clusters 
        GROUP BY face_count 
        ORDER BY face_count DESC 
        LIMIT 20
    """)
    size_distribution = cursor.fetchall()

    cursor.execute("""
        SELECT cluster_id, face_count 
        FROM face_clusters 
        ORDER BY face_count DESC 
        LIMIT 20
    """)
    top_clusters = cursor.fetchall()

    conn.close()

    logger.info("=== CLUSTERING STATISTICS ===")
    logger.info(f"Total clusters: {total_clusters:,}")
    logger.info(f"Total faces: {total_faces:,}")
    logger.info(f"Average cluster size: {avg_cluster_size:.2f}")
    logger.info(f"Largest cluster: {max_cluster_size} faces")

    logger.info("\nTop 20 clusters by size:")
    for cluster_id, face_count in top_clusters:
        logger.info(f"  Cluster {cluster_id}: {face_count} faces")

    logger.info("\nCluster size distribution:")
    for size, count in size_distribution:
        logger.info(f"  {size} faces: {count} clusters")


def main():
    """Main function to fix face counts and show stats."""
    fix_face_counts()
    show_stats()


if __name__ == "__main__":
    main()
