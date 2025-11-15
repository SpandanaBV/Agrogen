from django.db import models

class SoilRecord(models.Model):
    ph = models.FloatField()
    nitrogen = models.FloatField()
    phosphorus = models.FloatField()
    potassium = models.FloatField()
    moisture = models.FloatField()
    organic_carbon = models.FloatField()
    region = models.CharField(max_length=100, null=True, blank=True)
    score = models.IntegerField()
    recommended_crops = models.TextField()
    ai_response = models.TextField()
    ai_insight = models.TextField(null=True, blank=True)  # ✅ New Field
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"SoilRecord #{self.id} ({self.region or 'Unknown'})"
