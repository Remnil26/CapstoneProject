from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from location_recommendation import settings
import http.client, urllib.parse

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics


def index(request):
    return render(request,'index.html')


#Completed....
def established_business(request):
    if request.method == "POST":
        industry_type = request.POST.get("industry_type")
        #Established Business Logic
        df=pd.read_csv(settings.DATASET_DIR)

        business_type = industry_type
        business_df=df.loc[(df['Type'] == business_type)]
        cord_business=business_df[['Lat','Lng']].to_numpy()
        weight_business=(business_df['Total_user_rating']*business_df['Rating']).to_numpy()


        X = cord_business
        # Use silhouette score to find optimal number of clusters to segment the data
        num_clusters = np.arange(5,10)
        results = {}
        for size in num_clusters:
            model = KMeans(n_clusters = size).fit(X,sample_weight=weight_business)
            predictions = model.predict(X)
            results[size] = silhouette_score(X, predictions)

        best_size = max(results, key=results.get)
        kmeans = KMeans(n_clusters=best_size, random_state=0).fit(cord_business,sample_weight=weight_business)

        freq = {}
        for item in kmeans.labels_:
            if item in freq:
                freq[item] += 1
            else:
                freq[item] = 1
        freq=dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
        final_recommendation=[]
        counter=0
        for item in freq:
            if counter!=3:
                final_recommendation.append(list(kmeans.cluster_centers_[item]))
                counter+=1
            else:
                break

        conn = http.client.HTTPConnection('api.positionstack.com')
        for i in range(len(final_recommendation)):
            val_cord=str(final_recommendation[i][0])+","+str(final_recommendation[i][1])
            params = urllib.parse.urlencode({
                'access_key': '5b13acf7df21e170e3ebd85d6c02b4d0',
                'query': val_cord,
                })
            conn.request('GET', '/v1/reverse?{}'.format(params))
            res = conn.getresponse()
            data = res.read()
            dummy=data.decode('utf-8')
            all_items=dummy.split(":")
            if (dummy.split(":")[9].split(',')[0]=='null'):
                val=dummy.split(":")[28].split(',')[0]
            else:
                val=dummy.split(":")[9].split(',')[0]
            if val=='null':
                val="Can't determine location"
            final_recommendation[i].append(val)

        print("#FINAL REDOMMENDATION",final_recommendation)
        
        lat = []
        lon = []
        output = []
        for i in range(len(final_recommendation)):
            lat.append(final_recommendation[i][0])
            lon.append(final_recommendation[i][1])
            final_recommendation[i][2] = final_recommendation[i][2].replace('"', '')
            output.append(final_recommendation[i][2])


        my_dict = {
                   "industry_type" : industry_type, 
                   "output" : output, 
                   "lat" : lat,
                   "lon" : lon
            }
        return render(request,'established_business.html', context = my_dict)
    
    my_dict = {}
    return render(request,'established_business.html', context = my_dict)


#Completed...
def new_business(request):
    if request.method == "POST":
        industry_type = request.POST.get("industry_type")
        #New Business Logic
        df=pd.read_csv(settings.DATASET_DIR)
        
        corel={'spa':['clothing_store','jewelry_store','shopping_mall'],
               'book_store': ['supermarket','laundry','clothing_store'],
               'laundry':['university','spa','gym'],
               'travel_agency': ['shopping_mall','cafe','shoe_store'],
               'electronics_store': ['shoe_store','clothing_store','cafe'],
               'furniture_store': ['shoe_store','clothing_store','cafe'],
               'lodging': ['night_club','cafe','shopping_mall'],
               'movie_theater':['cafe','clothing_store','shopping_mall'],
               'cafe':['university','shopping_mall','travel_agency'],
               'car_wash': ['hardware_store','electronics_store','lodging'],
               'night_club':['shopping_mall','cafe','spa'],
               'gym': ['university','laundry','lodging'],
               'aquarium':['lodging','shoe_store','electronics_store'],
               'shopping_mall':['shoe_store','clothing_store','furniture_store'],
               'clothing_store':['shoe_store','university','cafe'],
               'university':['clothing_store','shopping_mall','cafe'],
               'pet_store':['car_wash','beauty_salon','lodging'],
               'car_repair':['shopping_mall','supermarket','shoe_store'],
               'hardware_store': ['clothing_store','shoe_store','electronics_store'],
               'jewelry_store':['shopping_mall','clothing_store','shoe_store'],
               'supermarket': ['cafe','university','travel_agency'],
               'gas_station': ['lodging','furniture_store','electronics_store'],
               'beauty_salon':['clothing_store','shoe_store','shopping_mall'],
               'shoe_store':['clothing_store','shopping_mall','cafe']}

        business_type = industry_type
        business_df=df.loc[(df['Type'] == corel[business_type][0]) | (df['Type'] == corel[business_type][1]) | (df['Type'] == corel[business_type][2])]
        cord_business=business_df[['Lat','Lng']].to_numpy()
        weight_business=(business_df['Total_user_rating']*business_df['Rating']).to_numpy()

        X=cord_business
        # Use silhouette score to find optimal number of clusters to segment the data
        num_clusters = np.arange(5,10)
        results = {}
        for size in num_clusters:
            model = KMeans(n_clusters = size).fit(X,sample_weight=weight_business)
            predictions = model.predict(X)
            results[size] = silhouette_score(X, predictions)

        best_size = max(results, key=results.get)

        kmeans = KMeans(n_clusters=best_size, random_state=0).fit(cord_business,sample_weight=weight_business)

        freq = {}
        for item in kmeans.labels_:
            if item in freq:
                freq[item] += 1
            else:
                freq[item] = 1
        freq=dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
        final_recommendation=[]
        counter=0
        for item in freq:
            if counter!=3:
                final_recommendation.append(list(kmeans.cluster_centers_[item]))
                counter+=1
            else:
                break


        conn = http.client.HTTPConnection('api.positionstack.com')
        for i in range(len(final_recommendation)):
            val_cord=str(final_recommendation[i][0])+","+str(final_recommendation[i][1])
            params = urllib.parse.urlencode({
                'access_key': '5b13acf7df21e170e3ebd85d6c02b4d0',
                'query': val_cord,
                })

            conn.request('GET', '/v1/reverse?{}'.format(params))

            res = conn.getresponse()
            data = res.read()
            dummy=data.decode('utf-8')
            all_items=dummy.split(":")
            if (dummy.split(":")[9].split(',')[0]=='null'):
                val=dummy.split(":")[28].split(',')[0]
            else:
                val=dummy.split(":")[9].split(',')[0]
            if val=='null':
                val="Can't determine location"
            final_recommendation[i].append(val)

        lat = []
        lon = []
        output = []
        for i in range(len(final_recommendation)):
            lat.append(final_recommendation[i][0])
            lon.append(final_recommendation[i][1])
            final_recommendation[i][2] = final_recommendation[i][2].replace('"', '')
            output.append(final_recommendation[i][2])

        my_dict = {
                "industry_type" : industry_type,
                "output" : output, 
                "lat" : lat,
                "lon" : lon
        }
        return render(request,"new_business.html",context = my_dict)
    my_dict = {}
    return render(request,"new_business.html",context = my_dict)




def region_wise_business(request):
    if request.method == "POST":
        region_type = int(request.POST.get("region_type"))
        df=pd.read_csv(settings.DATASET_DIR)
        df['Region'].replace(['R4', 'R6', 'R1', 'R3', 'R2', 'R5', 'R7'],[4,6,1,3,2,5,7], inplace=True)
        df1 = list(df['Type'].unique())
        df['Type'].replace(['spa', 'book_store', 'laundry', 'travel_agency', 'electronics_store', 'furniture_store', 'lodging','movie_theater','cafe','car_wash','night_club','gym','aquarium','shopping_mall','clothing_store','university','pet_store','car_repair','hardware_store','jewelry_store','supermarket','gas_station','beauty_salon','shoe_store'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24], inplace=True)
        df['Rating'] = df['Rating'].astype(int)
        
        region_wise_Dataframe = df[(df['Region'] == region_type)]


        X = region_wise_Dataframe.iloc[:, [2,-1]].values
        y = region_wise_Dataframe.iloc[:, 1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.70)
        classifier= DecisionTreeClassifier(criterion='entropy', random_state=42)            
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
            
        rating_pred = {}
            
        for j in range(1,25):
            y_pred = classifier.predict([[j,region_type]])
            rating_pred[df1[j-1]] = y_pred[0]

            
        d = sorted(rating_pred.items(), key=lambda x: x[1], reverse=True)
        recommendation_list = []
        for i in d:
            if i[1] != 0:
                recommendation_list.append(i[0])
        # print(recommendation_list)

        coordinate_list = settings.region_wise_coordinates[region_type]

        my_dict = {
                    'region_type' : region_type,
                    'output': recommendation_list,
                    'coordinate_list': coordinate_list,
        }    
        
        return render(request,"region_wise_recommendation.html",context = my_dict)
        
    my_dict = {} 
    return render(request,"region_wise_recommendation.html",context = my_dict)