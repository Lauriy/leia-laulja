import sqlite3


def reset_merge_attempts(db_path="face_groups.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("UPDATE face_clusters SET merge_attempted = NULL")
    affected_rows = cursor.rowcount

    conn.commit()
    conn.close()

    print(f"Reset merge_attempted for {affected_rows} clusters")


if __name__ == "__main__":
    reset_merge_attempts()
