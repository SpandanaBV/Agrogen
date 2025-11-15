from rest_framework import serializers
from .models import SoilRecord


class SoilRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = SoilRecord
        fields = '__all__'
