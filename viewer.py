import os

from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db import models
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from nanodjango import Django
from django.contrib import messages

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Django(
    DEBUG=True,
    SECRET_KEY="your-secret-key-here",
    DATABASES={
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": "face_groups.db",
        }
    },
    ALLOWED_HOSTS=["*"],
    MEDIA_ROOT=os.path.join(BASE_DIR, "face_thumbnails"),
    MEDIA_URL="/face_thumbnails/",
    INSTALLED_APPS=[
        "django.contrib.auth",
        "django.contrib.contenttypes",
        "django.contrib.sessions",
        "django.contrib.messages",
        "django.contrib.staticfiles",
    ],
    MIDDLEWARE=[
        "django.middleware.security.SecurityMiddleware",
        "django.contrib.sessions.middleware.SessionMiddleware",
        "django.middleware.common.CommonMiddleware",
        "django.middleware.csrf.CsrfViewMiddleware",
        "django.contrib.auth.middleware.AuthenticationMiddleware",
        "django.contrib.messages.middleware.MessageMiddleware",
        "django.middleware.clickjacking.XFrameOptionsMiddleware",
    ],
    STATIC_URL="/static/",
)


class Tag(models.Model):
    name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "tags"
        ordering = ["name"]
        managed = False

    def __str__(self):
        return self.name


class FaceCluster(models.Model):
    cluster_id = models.AutoField(primary_key=True)
    representative_embedding = models.BinaryField()
    face_count = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    tags = models.ManyToManyField(Tag, blank=True, related_name="clusters")

    class Meta:
        db_table = "face_clusters"
        ordering = ["-face_count"]
        managed = False


class FaceAssignment(models.Model):
    face_id = models.CharField(max_length=100, primary_key=True)
    cluster = models.ForeignKey(
        FaceCluster, on_delete=models.CASCADE, db_column="cluster_id"
    )
    frame_number = models.IntegerField()
    chunk_number = models.IntegerField()
    timestamp = models.CharField(max_length=20)
    confidence = models.FloatField()
    thumbnail_path = models.CharField(max_length=500, null=True, blank=True)
    bbox_x1 = models.IntegerField()
    bbox_y1 = models.IntegerField()
    bbox_x2 = models.IntegerField()
    bbox_y2 = models.IntegerField()

    class Meta:
        db_table = "face_assignments"
        managed = False


@app.route("/")
def cluster_list(request):
    """List all face clusters, ordered by face count"""
    clusters = (
        FaceCluster.objects.prefetch_related("tags").all().order_by("-face_count")
    )

    paginator = Paginator(clusters, 50)
    page = request.GET.get("page")
    clusters_page = paginator.get_page(page)

    return render(
        request,
        "cluster_list.html",
        {
            "clusters": clusters_page,
            "total_clusters": FaceCluster.objects.count(),
        },
    )


@app.route("/cluster/<int:cluster_id>/")
def cluster_detail(request, cluster_id):
    """View faces in a specific cluster"""
    cluster = get_object_or_404(FaceCluster, cluster_id=cluster_id)
    faces = FaceAssignment.objects.filter(cluster=cluster).order_by("frame_number")

    paginator = Paginator(faces, 100)
    page = request.GET.get("page")
    faces_page = paginator.get_page(page)

    return render(
        request,
        "cluster_detail.html",
        {
            "cluster": cluster,
            "faces": faces_page,
        },
    )


@app.route("/cluster/<int:cluster_id>/add_tag/")
def add_tag(request, cluster_id):
    """Add a tag to a cluster"""
    cluster = get_object_or_404(FaceCluster, cluster_id=cluster_id)
    tag_name = request.POST.get("tag_name", "").strip()

    if tag_name:
        tag, created = Tag.objects.get_or_create(name=tag_name)
        cluster.tags.add(tag)
        return JsonResponse({"success": True, "tag": tag.name, "created": created})

    return JsonResponse({"success": False, "error": "Tag name required"})


@app.route("/cluster/<int:cluster_id>/remove_tag/<int:tag_id>/")
@login_required
def remove_tag(request, cluster_id, tag_id):
    """Remove a tag from a cluster"""
    cluster = get_object_or_404(FaceCluster, cluster_id=cluster_id)
    tag = get_object_or_404(Tag, id=tag_id)
    cluster.tags.remove(tag)
    return JsonResponse({"success": True})


@app.route("/cluster/<int:cluster_id>/delete/")
@login_required
def delete_cluster(request, cluster_id):
    """Delete a cluster and all its face assignments and thumbnail files"""
    cluster = get_object_or_404(FaceCluster, cluster_id=cluster_id)

    if request.method == "POST":
        # Get all face assignments for this cluster
        face_assignments = FaceAssignment.objects.filter(cluster=cluster)

        # Delete thumbnail files
        deleted_files = 0
        for face in face_assignments:
            if face.thumbnail_path:
                file_path = os.path.join(BASE_DIR, face.thumbnail_path)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        deleted_files += 1
                    except OSError:
                        pass  # File might be already deleted or permission issue

        # Delete from database
        face_count = face_assignments.count()
        face_assignments.delete()
        cluster.delete()

        messages.success(
            request,
            f"Cluster {cluster_id} deleted successfully! "
            f"Removed {face_count} face assignments and {deleted_files} thumbnail files.",
        )
        return redirect("/")

    return render(
        request,
        "cluster_delete.html",
        {
            "cluster": cluster,
        },
    )


@app.route("/tag/<int:tag_id>/")
def tag_detail(request, tag_id):
    """View all clusters with this tag"""
    tag = get_object_or_404(Tag, id=tag_id)
    clusters = tag.clusters.all().order_by("-face_count")

    paginator = Paginator(clusters, 50)
    page = request.GET.get("page")
    clusters_page = paginator.get_page(page)

    return render(
        request,
        "tag_detail.html",
        {
            "tag": tag,
            "clusters": clusters_page,
        },
    )


@app.route("/api/tags/search/")
def search_tags(request):
    """Search existing tags for autocomplete"""
    query = request.GET.get("q", "").strip()
    if query:
        tags = Tag.objects.filter(name__icontains=query)[:10]
        return JsonResponse({"tags": [tag.name for tag in tags]})
    return JsonResponse({"tags": []})


@app.route("/tags/")
def tag_list(request):
    """List all tags with pagination"""
    tags = Tag.objects.all().order_by("name")

    paginator = Paginator(tags, 100)
    page = request.GET.get("page")
    tags_page = paginator.get_page(page)

    return render(
        request,
        "tag_list.html",
        {
            "tags": tags_page,
            "total_tags": Tag.objects.count(),
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0:8123")
