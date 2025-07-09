CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS face_clusters_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    facecluster_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    FOREIGN KEY (facecluster_id) REFERENCES face_clusters (cluster_id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags (id) ON DELETE CASCADE,
    UNIQUE(facecluster_id, tag_id)
);

CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
CREATE INDEX IF NOT EXISTS idx_face_clusters_tags_cluster ON face_clusters_tags(facecluster_id);
CREATE INDEX IF NOT EXISTS idx_face_clusters_tags_tag ON face_clusters_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_face_clusters_face_count ON face_clusters(face_count);
CREATE INDEX IF NOT EXISTS idx_face_assignments_cluster ON face_assignments(cluster_id);
CREATE INDEX IF NOT EXISTS idx_face_assignments_confidence ON face_assignments(confidence);
CREATE INDEX IF NOT EXISTS idx_face_assignments_frame_number ON face_assignments(frame_number);
CREATE INDEX IF NOT EXISTS idx_face_assignments_timestamp ON face_assignments(timestamp);