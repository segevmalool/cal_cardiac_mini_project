from django.shortcuts import render, render_to_response
from django.http import HttpResponse, JsonResponse
from django.template import loader

from django.conf import settings
import json

# Create your views here.
def index(request):
    return HttpResponse('<p>Hello World</p>')

def dash(request):
    with open(settings.BASE_DIR + '/datavis/index.html', 'r') as fp:
        page = fp.read()
    return HttpResponse(page)

def get_ecg_signals(request):
    with open(settings.STATIC_ROOT + '/data/ecg_signals.json', 'r') as fp:
        ecg = fp.read()
    return JsonResponse(ecg, safe=False)
