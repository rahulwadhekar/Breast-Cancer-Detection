from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib.auth.models import auth
from django.contrib.auth import authenticate,login,logout
from django.contrib import messages
from patient.models import PatientProfile
from doctor.models import DoctorProfile
import pandas as pd
from .models import Diseases
from django.contrib import messages
from tkinter.tix import IMAGE
from django.contrib import messages
from django.shortcuts import  render, redirect

from xml.etree.ElementTree import tostring
from .forms import NewUserForm
from .forms import NewUserForm
from centralapp.forms import NewUserForm
from django.contrib.auth.forms import AuthenticationForm #add this
from django.contrib.auth import login, logout, authenticate #add this
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
import cv2
import numpy as np
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from tensorflow.keras.models import load_model
import base64
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image , ImageTk

#from keras.optimizers import Adam
global up
up=""

# from patient.models import PatientProfile
# import requests
# from bs4 import BeautifulSoup



def mainpage(request):
    # page = request.get('')
    # soup = BeautifulSoup(page.text,'html.parser')
    # https://www.niams.nih.gov/health-topics/all-diseases
    # https://www.cdc.gov/diseasesconditions/index.html
    # https://www.pinehurstmedical.com/internalmedicine/internal-medicine-diseases-disorders-a-syndromes/
    # https://familydoctor.org/diseases-and-conditions/
    return render(request,'centralapp/mainpage.html')

def disease(request):
        return render(request, 'centralapp/multi_disease.html')

def disease1(request):
        return render(request, 'centralapp/multi_disease_SVM.html')
def classification(request):
	return render(request, 'disease_detect.html')


def classification1(request):
    if request.method == 'POST' and request.FILES['upload']:
        upload = request.FILES['upload']
        up=upload
        fn = up
        print("uploaded:",up)


        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        print("save ", file)
        file_url = fss.url(file)
        print("url:",file_url)


        imgpath = up

        fn = up
        IMAGE_SIZE = 64
        LEARN_RATE = 1.0e-4
        CH=3
        print(fn)
        if fn!="":

            #img = cv2.imread('C:/new/21C9588-Rice prediction/rice_web/rice_web/media/image.png',0)
            img = Image.open(fn)
            img = np.array(img.convert('L'))


            print(img)

            filename1 = 'userUploads/grey.jpeg'
            cv2.imwrite(filename1, img)


            file_url1 = fss.url(filename1)
            print("url:",file_url1)


            #convert into binary
            ret,binary = cv2.threshold(img,160,255,cv2.THRESH_BINARY)# 160 - threshold, 255 - value to assign, THRESH_BINARY_INV - Inverse binary
            #img.save('media/binary.jpeg')
            filename2 = 'userUploads/binary.jpeg'

            cv2.imwrite(filename2, binary)
            # Model Architecture and Compilation

            model = load_model(r'C:\Users\wadhe\OneDrive\Desktop\MediPharma\breast_cancerbreast.h5',compile=False)

            # adam = Adam(lr=LEARN_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
            # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

            img = Image.open(fn)
            img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
            img = np.array(img)

            img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)

            img = img.astype('float32')
            img = img / 255.0
            print('img shape:',img)
            prediction = model.predict(img)
            print(np.argmax(prediction))
            disease=np.argmax(prediction)
            print(disease)
            if disease == 0:
                Cd="breast cancer detected  "


            elif disease == 1:
                Cd=" breast cancer not  detected "
            A=Cd
            return render(request, "centralapp/multi_disease.html", {'MEDIA_URL': settings.MEDIA_URL,"predictions1": A,'file_url': file_url})

    else:

        return render(request, "centralapp/multi_disease.html")




def classificationsvm(request):
    if request.method == 'POST' and request.FILES['upload']:
        upload = request.FILES['upload']
        up=upload
        fn = up
        print("uploaded:",up)


        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        print("save ", file)
        file_url = fss.url(file)
        print("url:",file_url)


        imgpath = up

        fn = up
        IMAGE_SIZE = 64
        LEARN_RATE = 1.0e-4
        CH=3
        print(fn)
        if fn!="":

            #img = cv2.imread('C:/new/21C9588-Rice prediction/rice_web/rice_web/media/image.png',0)
            img = Image.open(fn)
            img = np.array(img.convert('L'))


            print(img)

            filename1 = 'userUploads/grey.jpeg'
            cv2.imwrite(filename1, img)


            file_url1 = fss.url(filename1)
            print("url:",file_url1)


            #convert into binary
            ret,binary = cv2.threshold(img,160,255,cv2.THRESH_BINARY)# 160 - threshold, 255 - value to assign, THRESH_BINARY_INV - Inverse binary
            #img.save('media/binary.jpeg')
            filename2 = 'userUploads/binary.jpeg'

            cv2.imwrite(filename2, binary)
            # Model Architecture and Compilation

            model = load_model(r'C:\Users\wadhe\OneDrive\Desktop\MediPharma\breast_cancer_svm.pkl',compile=False)

            # adam = Adam(lr=LEARN_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
            # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

            img = Image.open(fn)
            img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
            img = np.array(img)

            img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)

            img = img.astype('float32')
            img = img / 255.0
            print('img shape:',img)
            prediction = model.predict(img)
            print(np.argmax(prediction))
            disease=np.argmax(prediction)
            print(disease)
            if disease == 0:
                Cd="breast cancer detected  "


            elif disease == 1:
                Cd=" breast cancer not  detected "
            A=Cd
            return render(request, "centralapp/multi_disease.html", {'MEDIA_URL': settings.MEDIA_URL,"predictions1": A,'file_url': file_url})

    else:

        return render(request, "centralapp/multi_disease_SVM.html")

def login(request):
    if request.method=='POST':
        username=request.POST['username']
        password=request.POST['password']
        user=auth.authenticate(username=username,password=password)
        if user is not None:
            auth.login(request, user)
            usert =3
            # if request.user.patient.usertype=="1":
            if PatientProfile.objects.filter(patient = request.user):
                isuser = PatientProfile.objects.filter(patient = request.user)
                usert = [int(each.usertype) for each in isuser][0]
            elif DoctorProfile.objects.filter(doctor=request.user):
                isuser = DoctorProfile.objects.filter(doctor=request.user)
                usert = [int(each.usertype) for each in isuser][0]
            # usertype = [int(each.usertype) for each in isuser][0]
            # usertype = [int(each.usertype) for each in isuser]
            if usert==1:
                # return render(request,'centralapp/temp.html',{'isuser':isuser,'usert':usert})
                return redirect('centralapp:mainpage')
            elif usert==2:
                return redirect('centralapp:mainpage')
            # return render(request,'centralapp/temp.html', {'isuser':isuser,'usertype':usertype})
        # elif request.user.doctor.usertype=="2":
        else:
            messages.info(request,"Invalid Credentials!")
            return redirect('login')
    return render(request,'centralapp/login.html')

# uniquetogether



def About_us(request):
    return render(request,'centralapp/about_us.html')
def Cancer(request):
    return render(request,'centralapp/cancer.html')
def Covid_19(request):
    return render(request,'centralapp/Covid_19.html')
def disease_detect(request):
    return render(request,'centralapp/disease_detect.html')
def Diabetes(request):
    return render(request,'centralapp/diabetes.html')
def FAQS(request):
    return render(request,'centralapp/faqs.html')
def Heart_disorder(request):
    return render(request,'centralapp/heart_disorder.html')
def doc_how_to_use(request):
    return render(request,'centralapp/how_to_use_Doctor.html')
def patients_how_to_use(request):
    return render(request,'centralapp/how_to_use_User.html')
def Hypertension(request):
    return render(request,'centralapp/hypertension.html')
def Inside_health_records(request):
    return render(request,'centralapp/inside_health_records.html')
def Aids(request):
    return render(request,'centralapp/aids.html')


def searchBar(request):
    query=request.POST['searchBar']

    try:
        dis=Diseases.objects.get(name__icontains=query)
        return render(request,'centralapp/search_result.html',{"disease":dis})

    except :
        messages.error(request, f"Sorry! '{query}' does not exist in our Health Conditions dataset")
        return redirect('/')
