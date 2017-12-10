from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from django.conf import settings
import json

# Create your views here.
def index(request):
    return HttpResponse('<p>Hello World</p>')

def get_ecg_signals(request):
    with open(settings.STATIC_ROOT + '/data/ecg_signals.json', 'r') as fp:
        ecg = fp.read()
    return JsonResponse(ecg, safe=False)
