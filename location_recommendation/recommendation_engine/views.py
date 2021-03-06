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
import http.client
import urllib.parse
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import json


def index(request):
    return render(request, 'index.html')


# Completed....
def established_business(request):
    if request.method == "POST":
        industry_type = request.POST.get("industry_type")
        # Established Business Logic
        df = pd.read_csv(settings.DATASET_DIR)

        business_type = industry_type
        business_df = df.loc[(df['Type'] == business_type)]
        cord_business = business_df[['Lat', 'Lng']].to_numpy()
        
        weight_business = (
            business_df['Total_user_rating']*business_df['Rating']).to_numpy()

        X = cord_business
        # Use silhouette score to find optimal number of clusters to segment the data
        num_clusters = np.arange(3, 10)
        results = {}
        for size in num_clusters:
            model = KMeans(n_clusters=size).fit(
                X, sample_weight=weight_business)
            predictions = model.predict(X)
            results[size] = silhouette_score(X, predictions)

        best_size = max(results, key=results.get)
        kmeans = KMeans(n_clusters=best_size, random_state=0).fit(
            cord_business, sample_weight=weight_business)

        freq = {}
        for item in kmeans.labels_:
            if item in freq:
                freq[item] += 1
            else:
                freq[item] = 1
        freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
        final_recommendation = []
        counter = 0
        for item in freq:
            if counter != 3:
                final_recommendation.append(
                    list(kmeans.cluster_centers_[item]))
                counter += 1
            else:
                break

        conn = http.client.HTTPConnection('api.positionstack.com')
        for i in range(len(final_recommendation)):
            val_cord = str(
                final_recommendation[i][0])+","+str(final_recommendation[i][1])
            params = urllib.parse.urlencode({
                'access_key': '5b13acf7df21e170e3ebd85d6c02b4d0',
                'query': val_cord,
            })
            conn.request('GET', '/v1/reverse?{}'.format(params))
            res = conn.getresponse()
            data = res.read()
            dummy = data.decode('utf-8')
            all_items = dummy.split(":")

            try:
                if (dummy.split(":")[9].split(',')[0] != 'null'):
                    val = dummy.split(":")[9].split(',')[0]
                elif (dummy.split(":")[28].split(',')[0] != 'null'):
                    val = dummy.split(":")[28].split(',')[0]
                elif (dummy.split(":")[47].split(',')[0] != 'null'):
                    val = dummy.split(":")[47].split(',')[0]
                elif (dummy.split(":")[66].split(',')[0] != 'null'):
                    val = dummy.split(":")[66].split(',')[0]
                elif (dummy.split(":")[85].split(',')[0] != 'null'):
                    val = dummy.split(":")[85].split(',')[0]
                elif (dummy.split(":")[104].split(',')[0] != 'null'):
                    val = dummy.split(":")[104].split(',')[0]
                elif (dummy.split(":")[123].split(',')[0] != 'null'):
                    val = dummy.split(":")[123].split(',')[0]
                elif (dummy.split(":")[142].split(',')[0] != 'null'):
                    val = dummy.split(":")[142].split(',')[0]
                elif (dummy.split(":")[161].split(',')[0] != 'null'):
                    val = dummy.split(":")[161].split(',')[0]
                elif (dummy.split(":")[180].split(',')[0] != 'null'):
                    val = dummy.split(":")[180].split(',')[0]
                else:
                    val = dummy.split(":")[9].split(',')[0]
                if val == 'null':
                    val = "Can't determine location"
            except Exception as e:
                val = "Service Unavailable"
            final_recommendation[i].append(val)


        lat = []
        lon = []
        output = []
        for i in range(len(final_recommendation)):
            lat.append(final_recommendation[i][0])
            lon.append(final_recommendation[i][1])
            final_recommendation[i][2] = final_recommendation[i][2].replace(
                '"', '')
            output.append(final_recommendation[i][2])

        center_lat = sum(lat)/len(lat)

        center_lon = sum(lon)/len(lon)

        detailed_recommendation = vicinity_details(final_recommendation,industry_type,df)
        

        recommended_lonlat = []
        for i in range(len(detailed_recommendation)):
            recommended_lonlat.append([detailed_recommendation[i][1],detailed_recommendation[i][0],detailed_recommendation[i][2]])

        print(recommended_lonlat)
        

        my_dict = {
            "industry_type": industry_type,
            "output": output,
            "lat": lat,
            "lon": lon,
            "recommended_lonlat": recommended_lonlat,
            "cord_business" : cord_business.tolist(),
            'final_recomendation': detailed_recommendation,
            'center_lat': center_lat,
            'center_lon': center_lon,
        }

        return render(request, 'established_business.html', context=my_dict)

    my_dict = {}
    return render(request, 'established_business.html', context=my_dict)


# Completed...
def new_business(request):
    if request.method == "POST":
        industry_type = request.POST.get("industry_type")
        # New Business Logic
        df = pd.read_csv(settings.DATASET_DIR)

        corel = {'spa': ['clothing_store', 'jewelry_store', 'shopping_mall'],
                 'book_store': ['supermarket', 'laundry', 'clothing_store'],
                 'laundry': ['university', 'spa', 'gym'],
                 'travel_agency': ['shopping_mall', 'cafe', 'shoe_store'],
                 'electronics_store': ['shoe_store', 'clothing_store', 'cafe'],
                 'furniture_store': ['shoe_store', 'clothing_store', 'cafe'],
                 'lodging': ['night_club', 'cafe', 'shopping_mall'],
                 'movie_theater': ['cafe', 'clothing_store', 'shopping_mall'],
                 'cafe': ['university', 'shopping_mall', 'travel_agency'],
                 'car_wash': ['hardware_store', 'electronics_store', 'lodging'],
                 'night_club': ['shopping_mall', 'cafe', 'spa'],
                 'gym': ['university', 'laundry', 'lodging'],
                 'aquarium': ['lodging', 'shoe_store', 'electronics_store'],
                 'shopping_mall': ['shoe_store', 'clothing_store', 'furniture_store'],
                 'clothing_store': ['shoe_store', 'university', 'cafe'],
                 'university': ['clothing_store', 'shopping_mall', 'cafe'],
                 'pet_store': ['car_wash', 'beauty_salon', 'lodging'],
                 'car_repair': ['shopping_mall', 'supermarket', 'shoe_store'],
                 'hardware_store': ['clothing_store', 'shoe_store', 'electronics_store'],
                 'jewelry_store': ['shopping_mall', 'clothing_store', 'shoe_store'],
                 'supermarket': ['cafe', 'university', 'travel_agency'],
                 'gas_station': ['lodging', 'furniture_store', 'electronics_store'],
                 'beauty_salon': ['clothing_store', 'shoe_store', 'shopping_mall'],
                 'shoe_store': ['clothing_store', 'shopping_mall', 'cafe']}

        business_type = industry_type
        business_df = df.loc[(df['Type'] == corel[business_type][0]) | (
            df['Type'] == corel[business_type][1]) | (df['Type'] == corel[business_type][2])]
        cord_business = business_df[['Lat', 'Lng']].to_numpy()
        business_industry_df = df.loc[(df['Type'] == business_type)]
        cord_industry_details = business_industry_df[['Lat', 'Lng']].to_numpy()
        weight_business = (
            business_df['Total_user_rating']*business_df['Rating']).to_numpy()

        X = cord_business
        # Use silhouette score to find optimal number of clusters to segment the data
        num_clusters = np.arange(3, 10)
        results = {}
        for size in num_clusters:
            model = KMeans(n_clusters=size).fit(
                X, sample_weight=weight_business)
            predictions = model.predict(X)
            results[size] = silhouette_score(X, predictions)

        best_size = max(results, key=results.get)

        kmeans = KMeans(n_clusters=best_size, random_state=0).fit(
            cord_business, sample_weight=weight_business)

        freq = {}
        for item in kmeans.labels_:
            if item in freq:
                freq[item] += 1
            else:
                freq[item] = 1
        freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
        final_recommendation = []
        counter = 0
        for item in freq:
            if counter != 3:
                final_recommendation.append(
                    list(kmeans.cluster_centers_[item]))
                counter += 1
            else:
                break

        conn = http.client.HTTPConnection('api.positionstack.com')
        for i in range(len(final_recommendation)):
            val_cord = str(
                final_recommendation[i][0])+","+str(final_recommendation[i][1])
            params = urllib.parse.urlencode({
                'access_key': '5b13acf7df21e170e3ebd85d6c02b4d0',
                'query': val_cord,
            })

            conn.request('GET', '/v1/reverse?{}'.format(params))

            res = conn.getresponse()
            data = res.read()
            dummy = data.decode('utf-8')
            all_items = dummy.split(":")

            try:
                if (dummy.split(":")[9].split(',')[0] != 'null'):
                    val = dummy.split(":")[9].split(',')[0]
                elif (dummy.split(":")[28].split(',')[0] != 'null'):
                    val = dummy.split(":")[28].split(',')[0]
                elif (dummy.split(":")[47].split(',')[0] != 'null'):
                    val = dummy.split(":")[47].split(',')[0]
                elif (dummy.split(":")[66].split(',')[0] != 'null'):
                    val = dummy.split(":")[66].split(',')[0]
                elif (dummy.split(":")[85].split(',')[0] != 'null'):
                    val = dummy.split(":")[85].split(',')[0]
                elif (dummy.split(":")[104].split(',')[0] != 'null'):
                    val = dummy.split(":")[104].split(',')[0]
                elif (dummy.split(":")[123].split(',')[0] != 'null'):
                    val = dummy.split(":")[123].split(',')[0]
                elif (dummy.split(":")[142].split(',')[0] != 'null'):
                    val = dummy.split(":")[142].split(',')[0]
                elif (dummy.split(":")[161].split(',')[0] != 'null'):
                    val = dummy.split(":")[161].split(',')[0]
                elif (dummy.split(":")[180].split(',')[0] != 'null'):
                    val = dummy.split(":")[180].split(',')[0]
                else:
                    val = dummy.split(":")[9].split(',')[0]
                if val == 'null':
                    val = "Can't determine location"
            except Exception as e:
                val = "Service Unavailability"
            final_recommendation[i].append(val)

        lat = []
        lon = []
        output = []
        for i in range(len(final_recommendation)):
            lat.append(final_recommendation[i][0])
            lon.append(final_recommendation[i][1])
            final_recommendation[i][2] = final_recommendation[i][2].replace(
                '"', '')
            output.append(final_recommendation[i][2])

        center_lat = sum(lat)/len(lat)

        center_lon = sum(lon)/len(lon)

        detailed_recommendation = vicinity_details(final_recommendation,industry_type,df)

        my_dict = {
            "industry_type": industry_type,
            "output": output,
            "lat": lat,
            "lon": lon,
            "final_recomendation": final_recommendation,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'detailed_recommendation' : detailed_recommendation,
            'cord_business': cord_industry_details.tolist(),
        }
        print(final_recommendation)
        return render(request, "new_business.html", context=my_dict)
    my_dict = {}
    return render(request, "new_business.html", context=my_dict)




def region_wise_business(request):
    if request.method == "POST":
        region_type = int(request.POST.get("region_type"))
        df = pd.read_csv(settings.DATASET_DIR)
        df['Region'].replace(['R4', 'R6', 'R1', 'R3', 'R2', 'R5', 'R7'],[4,6,1,3,2,5,7], inplace=True)
        df['Rating'] = df['Rating'].astype(int)
        df2 = df.copy()
        df1 = list(df['Type'].unique())
        df['Type'].replace(['spa', 'book_store', 'laundry', 'travel_agency', 'electronics_store', 'furniture_store', 'lodging','movie_theater','cafe','car_wash','night_club','gym','aquarium','shopping_mall','clothing_store','university','pet_store','car_repair','hardware_store','jewelry_store','supermarket','gas_station','beauty_salon','shoe_store'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24], inplace=True)
        region_wise_Dataframe = df[(df['Region'] == region_type)]

        accuracy_list = []
        classifier_list = []

        for i in range(1):
            X = region_wise_Dataframe.iloc[:, [2,4,-1]].values
            y = region_wise_Dataframe.iloc[:, 1].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.70)
            temp_classifier= RandomForestClassifier(criterion='entropy', random_state=42)
            temp_classifier.fit(X_train, y_train)
            predictions = temp_classifier.predict(X_test)
            region_accuracy = metrics.accuracy_score(y_test, predictions)*100
            accuracy_list.append(region_accuracy)
            classifier_list.append(temp_classifier)

        max_accuracy = max(accuracy_list)
        max_accuracy_index = accuracy_list.index(max_accuracy)
        classifier = classifier_list[max_accuracy_index]

        rating_pred = {}

        for j in range(1,25):
            new_df = region_wise_Dataframe[(region_wise_Dataframe['Region'] == region_type)&(region_wise_Dataframe['Type'] == j)]
            if len(new_df) == 0:
                continue
            else:
                avg_rating = sum(new_df['Total_user_rating'])/len(new_df)
                y_pred = classifier.predict([[j,avg_rating,region_type]]) 
                rating_pred[df1[j-1]] = y_pred[0]


        d = sorted(rating_pred.items(), key=lambda x: x[1], reverse=True)
        
        recommendation_list = []
        predicted_rating = []
        for i in d:
            if i[1] != 0:
                recommendation_list.append(i[0])
                predicted_rating.append(i[1])



        coordinate_list = settings.region_wise_coordinates[region_type]
        lat_list = []
        lon_list = []

        for i in coordinate_list:
            lon_list.append(i[0])
            lat_list.append(i[1])

        center_lat = sum(lat_list)/len(lat_list)
        center_lon = sum(lon_list)/len(lon_list)

        top_five = recommendation_list[:5]

        count_shops = []
        avg_rating = []

        for type in top_five:
            count_df = df2[(df2['Region'] == region_type)&(df2['Type'] == type)]
            # print(count_df)
            rating = count_df['Rating'].mean(axis=0)
            avg_rating.append(round(rating,2))
            count_shops.append(len(count_df))

        for i in range(len(top_five)):
            if '_' in top_five[i]:
                top_five[i] = top_five[i].replace('_', ' ').title()
            else:
                top_five[i] = top_five[i].title()


        top_predicted_rating = predicted_rating[:5]

        my_dict = {
            'region_type': region_type,
            'top_five': top_five,
            'top_predicted_rating': top_predicted_rating,
            'coordinate_list': coordinate_list,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'total_shops': count_shops,
            'avg_rating': avg_rating,
        }

        return render(request, "region_wise_recommendation.html", context=my_dict)

    my_dict = {}
    return render(request, "region_wise_recommendation.html", context=my_dict)




def vicinity_details(recommendation_list,type_industry,df):
    
    for items in recommendation_list:
        
        details = {}
        dummy = pd.DataFrame(columns = ["Name", "Rating", "Type","Price Level", "Total_user_rating", "Locality", "Lat", "Lng", "Region"])
        for i in range(len(df)):
            x = items[0]
            y = items[1]
            x1 = df.loc[i, "Lat"]
            x2 = df.loc[i, "Lng"]
            R = 6373.0 #radius of the Earth
            lat1 = math.radians(x) #coordinates
            lon1 = math.radians(y)
            lat2 = math.radians(x1)
            lon2 = math.radians(x2)

            dlon = lon2 - lon1 #change in coordinates

            dlat = lat2 - lat1

            a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2 #Haversine formula

            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c * 1609
            #type_industry = df.loc[i, "Type"]
            #if(distance <= 2000 and distance != 0 and type_industry == df.loc[i, "Type"]):
            if(distance <= 2000 and distance != 0):
                new_row = {'Name': df.loc[i, "Name"], 'Rating': df.loc[i, "Rating"], 'Type': df.loc[i, "Type"], 'Price Level': df.loc[i, "Price Level"], 'Total_user_rating': df.loc[i, "Total_user_rating"], 'Locality': df.loc[i, "Locality"], 'Lat': df.loc[i, "Lat"], 'Lng': df.loc[i, "Lng"], 'Region': df.loc[i, "Region"]}
                dummy = dummy.append(new_row, ignore_index=True)

        current_industry_df = dummy[(dummy['Type'] == type_industry)]
        #popularity(avg_rating percentage, high)
        details["popularity_index"] = round((current_industry_df["Rating"].mean()/5)*100, 3)
        #percentage positive rating
        details["percent_positive_rating"] = round((current_industry_df[(current_industry_df['Rating'] >= 3.5)].shape[0]/current_industry_df.shape[0])*100, 3)
        #percentage negative rating
        details["percent_negative_rating"] = round((current_industry_df[(current_industry_df['Rating'] < 3.5)].shape[0]/current_industry_df.shape[0])*100, 3)
        #total_shops (all industry types)
        details["total_shops"] = dummy.shape[0]
        #total_shops (current industry type)
        details["total_shops_current_industry"] = current_industry_df.shape[0]
        #total_user_rating (current_industry_type)
        details["total_reviews"] = current_industry_df["Total_user_rating"].sum()
        #avg_rating of other insutries
        details["avg_rating_other_industries"] = round(dummy[(dummy["Type"] != type_industry)]["Rating"].mean(), 3)
        
        items.append(details)
        
    return recommendation_list
