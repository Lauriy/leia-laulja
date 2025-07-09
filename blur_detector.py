import cv2
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_blur_score(image_path):
    """Calculate blur score using Laplacian variance."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        return variance
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None


def add_blur_scores_to_db(db_path="face_groups.db", base_path=""):
    """Add blur scores to face_assignments table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all face assignments without blur scores
    cursor.execute("""
                   SELECT face_id, thumbnail_path
                   FROM face_assignments
                   WHERE blur_score IS NULL
                     AND thumbnail_path IS NOT NULL
                   """)

    faces = cursor.fetchall()
    total_faces = len(faces)

    for i, (face_id, thumbnail_path) in enumerate(faces):
        if i % 1000 == 0:
            logger.info(f"Processing {i}/{total_faces} faces...")

        full_path = os.path.join(base_path, thumbnail_path)
        if not os.path.exists(full_path):
            continue

        blur_score = calculate_blur_score(full_path)
        if blur_score is not None:
            cursor.execute(
                """
                           UPDATE face_assignments
                           SET blur_score = ?
                           WHERE face_id = ?
                           """,
                (blur_score, face_id),
            )

    conn.commit()
    conn.close()
    logger.info(f"✅ Added blur scores to {total_faces} faces")


def find_blurry_clusters(db_path="face_groups.db", blur_threshold=100):
    """Find clusters with mostly blurry faces."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
                   SELECT cluster_id,
                          AVG(blur_score) as avg_blur,
                          COUNT(*)        as face_count,
                          AVG(confidence) as avg_confidence
                   FROM face_assignments
                   WHERE blur_score IS NOT NULL
                   GROUP BY cluster_id
                   HAVING AVG(blur_score) < ?
                   ORDER BY avg_blur ASC
                   """,
        (blur_threshold,),
    )

    blurry_clusters = cursor.fetchall()

    logger.info(f"Found {len(blurry_clusters)} blurry clusters:")
    for cluster_id, avg_blur, face_count, avg_conf in blurry_clusters[:20]:
        logger.info(
            f"  Cluster {cluster_id}: blur={avg_blur:.1f}, faces={face_count}, conf={avg_conf:.3f}"
        )

    conn.close()
    return blurry_clusters


def delete_blurry_clusters(
    db_path="face_groups.db", blur_threshold=100, batch_size=900
):
    """Delete clusters with low blur scores in batches."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Find blurry clusters
    cursor.execute(
        """
                   SELECT cluster_id
                   FROM (SELECT cluster_id, AVG(blur_score) as avg_blur
                         FROM face_assignments
                         WHERE blur_score IS NOT NULL
                         GROUP BY cluster_id
                         HAVING AVG(blur_score) < ?)
                   """,
        (blur_threshold,),
    )

    blurry_cluster_ids = [row[0] for row in cursor.fetchall()]

    if not blurry_cluster_ids:
        logger.info("No blurry clusters found")
        return

    logger.info(f"Found {len(blurry_cluster_ids)} blurry clusters to delete")

    # Process in batches to avoid SQL variable limit
    total_deleted = 0
    for i in range(0, len(blurry_cluster_ids), batch_size):
        batch = blurry_cluster_ids[i : i + batch_size]
        placeholders = ",".join(["?" for _ in batch])

        cursor.execute(
            f"DELETE FROM face_clusters WHERE cluster_id IN ({placeholders})", batch
        )
        deleted_count = cursor.rowcount
        total_deleted += deleted_count

        logger.info(f"Deleted batch {i // batch_size + 1}: {deleted_count} clusters")

        # Commit after each batch
        conn.commit()

    conn.close()
    logger.info(f"✅ Deleted {total_deleted} blurry clusters total")


if __name__ == "__main__":
    # Step 1: Calculate blur scores for all faces
    add_blur_scores_to_db()

    # Step 2: Find blurry clusters
    find_blurry_clusters(blur_threshold=100)

    # Step 3: Delete them in batches
    delete_blurry_clusters(blur_threshold=100)
