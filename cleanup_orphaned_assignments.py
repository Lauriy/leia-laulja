import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_orphaned_assignments(db_path="face_groups.db", dry_run=True):
    """Delete face assignments that reference non-existent clusters."""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Find orphaned assignments (assignments without corresponding clusters)
    logger.info("🔍 Finding orphaned face assignments...")
    cursor.execute('''
        SELECT COUNT(*) 
        FROM face_assignments fa
        LEFT JOIN face_clusters fc ON fa.cluster_id = fc.cluster_id
        WHERE fc.cluster_id IS NULL
    ''')
    
    orphaned_count = cursor.fetchone()[0]
    logger.info(f"📊 Found {orphaned_count:,} orphaned face assignments")
    
    if orphaned_count == 0:
        logger.info("✅ No orphaned assignments found!")
        conn.close()
        return
    
    # Show some examples
    logger.info("📝 Examples of orphaned assignments:")
    cursor.execute('''
        SELECT fa.face_id, fa.cluster_id, fa.thumbnail_path, fa.confidence
        FROM face_assignments fa
        LEFT JOIN face_clusters fc ON fa.cluster_id = fc.cluster_id
        WHERE fc.cluster_id IS NULL
        LIMIT 10
    ''')
    
    for i, (face_id, cluster_id, thumbnail_path, confidence) in enumerate(cursor.fetchall()):
        logger.info(f"  {i+1}. Face {face_id} -> Cluster {cluster_id} (conf: {confidence:.3f})")
    
    if dry_run:
        logger.info("🔍 DRY RUN MODE - No records will be deleted")
        logger.info("🔄 Run with dry_run=False to actually delete records")
        conn.close()
        return
    
    # Delete orphaned assignments
    logger.info("🗑️  Deleting orphaned face assignments...")
    cursor.execute('''
        DELETE FROM face_assignments
        WHERE cluster_id NOT IN (
            SELECT cluster_id FROM face_clusters
        )
    ''')
    
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    
    logger.info(f"✅ Deleted {deleted_count:,} orphaned face assignments")
    logger.info("💾 Database cleanup complete!")

if __name__ == "__main__":
    # First run in dry-run mode
    # cleanup_orphaned_assignments(dry_run=True)
    
    # Uncomment to actually delete orphaned assignments
    cleanup_orphaned_assignments(dry_run=False)