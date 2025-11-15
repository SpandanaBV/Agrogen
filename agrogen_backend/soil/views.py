from django.shortcuts import render

# Create your views here.
from rest_framework import viewsets
from .models import SoilRecord
from .serializers import SoilRecordSerializer

class SoilRecordViewSet(viewsets.ModelViewSet):
    queryset = SoilRecord.objects.all().order_by('-created_at')
    serializer_class = SoilRecordSerializer
