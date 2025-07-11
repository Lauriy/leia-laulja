import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cleanup_orphaned_thumbnails(
    db_path="face_groups.db", thumbnails_dir="face_thumbnails", dry_run=True
):
    """Delete thumbnail files that are no longer referenced in the database."""

    # Get all thumbnail paths from the database
    logger.info("📋 Getting all thumbnail paths from database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT thumbnail_path 
        FROM face_assignments 
        WHERE thumbnail_path IS NOT NULL
    """)

    db_thumbnails = set()
    for row in cursor.fetchall():
        thumbnail_path = row[0]
        if thumbnail_path:
            # Strip the face_thumbnails/ prefix if it exists
            if thumbnail_path.startswith("face_thumbnails/"):
                thumbnail_path = thumbnail_path[len("face_thumbnails/") :]
            db_thumbnails.add(thumbnail_path)

    conn.close()

    logger.info(f"📊 Found {len(db_thumbnails):,} thumbnail paths in database")

    # Get all files in the thumbnails directory
    logger.info("📁 Scanning thumbnail directory...")
    thumbnails_path = Path(thumbnails_dir)

    if not thumbnails_path.exists():
        logger.error(f"❌ Thumbnail directory {thumbnails_dir} doesn't exist!")
        return

    disk_files = set()
    total_files = 0

    for file_path in thumbnails_path.rglob("*"):
        if file_path.is_file():
            # Get relative path from thumbnails directory
            relative_path = file_path.relative_to(thumbnails_path)
            disk_files.add(str(relative_path))
            total_files += 1

    logger.info(f"📊 Found {total_files:,} files on disk")

    # Find orphaned files (on disk but not in DB)
    orphaned_files = disk_files - db_thumbnails

    logger.info(f"🗑️  Found {len(orphaned_files):,} orphaned thumbnail files")

    if not orphaned_files:
        logger.info("✅ No orphaned files to delete!")
        return

    # Show some examples
    logger.info("📝 Examples of orphaned files:")
    for i, file_path in enumerate(sorted(orphaned_files)[:10]):
        logger.info(f"  {i + 1}. {file_path}")

    if len(orphaned_files) > 10:
        logger.info(f"  ... and {len(orphaned_files) - 10:,} more")

    # Calculate total size of orphaned files
    total_size = 0
    for file_path in orphaned_files:
        full_path = thumbnails_path / file_path
        if full_path.exists():
            total_size += full_path.stat().st_size

    logger.info(f"💾 Total size of orphaned files: {total_size / 1024 / 1024:.1f} MB")

    if dry_run:
        logger.info("🔍 DRY RUN MODE - No files will be deleted")
        logger.info("🔄 Run with dry_run=False to actually delete files")
        return

    # Delete orphaned files
    logger.info("🗑️  Deleting orphaned files...")
    deleted_count = 0
    deleted_size = 0

    for file_path in orphaned_files:
        full_path = thumbnails_path / file_path
        if full_path.exists():
            try:
                file_size = full_path.stat().st_size
                full_path.unlink()
                deleted_count += 1
                deleted_size += file_size

                if deleted_count % 1000 == 0:
                    logger.info(f"  Deleted {deleted_count:,} files so far...")

            except OSError as e:
                logger.error(f"❌ Failed to delete {file_path}: {e}")

    logger.info(f"✅ Deleted {deleted_count:,} orphaned files")
    logger.info(f"💾 Freed {deleted_size / 1024 / 1024:.1f} MB of disk space")

    # Clean up empty directories
    logger.info("🧹 Cleaning up empty directories...")
    empty_dirs_removed = 0

    for dir_path in sorted(thumbnails_path.rglob("*"), reverse=True):
        if dir_path.is_dir():
            try:
                dir_path.rmdir()  # Only removes if empty
                empty_dirs_removed += 1
                logger.info(
                    f"  Removed empty directory: {dir_path.relative_to(thumbnails_path)}"
                )
            except OSError:
                pass  # Directory not empty, that's fine

    logger.info(f"🧹 Removed {empty_dirs_removed} empty directories")


def find_db_files_not_on_disk(
    db_path="face_groups.db", thumbnails_dir="face_thumbnails"
):
    """Find thumbnail paths in DB that don't exist on disk."""

    logger.info("🔍 Finding database entries with missing thumbnail files...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT face_id, thumbnail_path, cluster_id
        FROM face_assignments 
        WHERE thumbnail_path IS NOT NULL
    """)

    missing_files = []
    thumbnails_path = Path(thumbnails_dir)

    for face_id, thumbnail_path, cluster_id in cursor.fetchall():
        # Handle both formats: with and without face_thumbnails/ prefix
        if thumbnail_path.startswith("face_thumbnails/"):
            relative_path = thumbnail_path[len("face_thumbnails/") :]
        else:
            relative_path = thumbnail_path

        full_path = thumbnails_path / relative_path
        if not full_path.exists():
            missing_files.append((face_id, thumbnail_path, cluster_id))

    conn.close()

    logger.info(f"📊 Found {len(missing_files):,} database entries with missing files")

    if missing_files:
        logger.info("📝 Examples of missing files:")
        for i, (face_id, thumbnail_path, cluster_id) in enumerate(missing_files[:10]):
            logger.info(
                f"  {i + 1}. {thumbnail_path} (face_id: {face_id}, cluster: {cluster_id})"
            )

    return missing_files


if __name__ == "__main__":
    # First, find orphaned files (dry run)
    # cleanup_orphaned_thumbnails(dry_run=True)

    # Also check for missing files referenced in DB
    # find_db_files_not_on_disk()

    # Uncomment to actually delete orphaned files
    cleanup_orphaned_thumbnails(dry_run=False)
