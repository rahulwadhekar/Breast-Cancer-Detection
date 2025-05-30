from . import views
from django.urls import path
from django.contrib.auth import views as authentication_views
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include

from . import views

app_name = 'centralapp'
urlpatterns = [

    path('',views.mainpage,name='mainpage'),
    path('About_us/',views.About_us,name ='About_us'),
    path('Aids/',views.Aids,name ='Aids'),
    path('Cancer/',views.Cancer,name ='Cancer'),
    path('classification1/',views.classification1,name ='classification1'),
    path('classificationsvm/',views.classificationsvm,name ='classificationsvm'),

    path('Covid-19/',views.Covid_19,name ='Covid_19'),
    path('disease/',views.disease,name ='disease'),
    path('disease1/',views.disease1,name ='disease1'),
    path('Diabetes/',views.Diabetes,name ='Diabetes'),
    path('FAQS/',views.FAQS,name ='FAQS'),
    path('Heart_disorder/',views.Heart_disorder,name ='Heart_disorder'),
    path('medical_practitioners/How_to_use',views.doc_how_to_use,name ='doc_how_to_use'),
    path('friends-and-family/How_to_use',views.patients_how_to_use,name ='patients_how_to_use'),
    path('Hypertension/',views.Hypertension,name ='Hypertension'),
    path('Inside_health_records/',views.Inside_health_records,name ='Inside_health_records'),
    path('login/',views.login,name='login'),
    path('logout/', authentication_views.LogoutView.as_view(template_name='centralapp/logout.html'), name='logout'),
     path('searchBar/',views.searchBar,name='searchBar'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
