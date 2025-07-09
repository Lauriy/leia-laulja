import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_representative_thumbnails(db_path="face_groups.db"):
    """Update each cluster's representative thumbnail to the highest confidence face."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all clusters
    cursor.execute("SELECT cluster_id FROM face_clusters")
    cluster_ids = [row[0] for row in cursor.fetchall()]

    total_clusters = len(cluster_ids)
    logger.info(
        f"Updating representative thumbnails for {total_clusters:,} clusters..."
    )

    updated_count = 0

    for i, cluster_id in enumerate(cluster_ids):
        if i % 1000 == 0:
            logger.info(
                f"Processing cluster {i:,}/{total_clusters:,} ({i / total_clusters * 100:.1f}%)"
            )

        # Find the highest confidence face in this cluster
        cursor.execute(
            """
            SELECT thumbnail_path, confidence
            FROM face_assignments
            WHERE cluster_id = ? AND thumbnail_path IS NOT NULL
            ORDER BY confidence DESC
            LIMIT 1
        """,
            (cluster_id,),
        )

        result = cursor.fetchone()
        if result:
            best_thumbnail, best_confidence = result

            # Update the cluster's representative thumbnail
            cursor.execute(
                """
                UPDATE face_clusters
                SET representative_thumbnail = ?
                WHERE cluster_id = ?
            """,
                (best_thumbnail, cluster_id),
            )

            updated_count += 1
        else:
            logger.warning(f"No faces found for cluster {cluster_id}")

    # Commit all changes
    conn.commit()
    conn.close()

    logger.info(f"✅ Updated {updated_count:,} cluster representative thumbnails")


def verify_representative_thumbnails(db_path="face_groups.db", sample_size=10):
    """Verify that representative thumbnails are actually the highest confidence faces."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get a sample of clusters with their representative thumbnails
    cursor.execute(
        """
        SELECT cluster_id, representative_thumbnail
        FROM face_clusters
        WHERE representative_thumbnail IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """,
        (sample_size,),
    )

    clusters_to_check = cursor.fetchall()

    logger.info(f"Verifying {len(clusters_to_check)} random clusters...")

    for cluster_id, rep_thumbnail in clusters_to_check:
        # Get the confidence of the representative thumbnail
        cursor.execute(
            """
            SELECT confidence
            FROM face_assignments
            WHERE cluster_id = ? AND thumbnail_path = ?
        """,
            (cluster_id, rep_thumbnail),
        )

        rep_confidence = cursor.fetchone()
        if not rep_confidence:
            logger.warning(
                f"❌ Cluster {cluster_id}: representative thumbnail not found in assignments"
            )
            continue

        rep_confidence = rep_confidence[0]

        # Get the highest confidence in this cluster
        cursor.execute(
            """
            SELECT MAX(confidence)
            FROM face_assignments
            WHERE cluster_id = ? AND thumbnail_path IS NOT NULL
        """,
            (cluster_id,),
        )

        max_confidence = cursor.fetchone()[0]

        if (
            abs(rep_confidence - max_confidence) < 0.001
        ):  # Allow for small floating point differences
            logger.info(
                f"✅ Cluster {cluster_id}: representative confidence {rep_confidence:.3f} matches max {max_confidence:.3f}"
            )
        else:
            logger.warning(
                f"❌ Cluster {cluster_id}: representative confidence {rep_confidence:.3f} != max {max_confidence:.3f}"
            )

    conn.close()


if __name__ == "__main__":
    # Update representative thumbnails
    update_representative_thumbnails()

    # Verify a few random ones
    verify_representative_thumbnails(sample_size=20)
