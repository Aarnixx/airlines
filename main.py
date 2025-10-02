import requests
import pandas as pd
import random
from geopy import distance

API_KEY = "a67e8979758f4feea51637156bbdaf25"

source_city = pd.read_csv(
    r"C:\Users\aarni.annanolli\Python\LentoTilastot\airlines_flights_data.csv",usecols=[3])
destination = pd.read_csv(
    r"C:\Users\aarni.annanolli\Python\LentoTilastot\airlines_flights_data.csv",usecols=[7])

source_cities = source_city.iloc[:, 0]
destination_cities = destination.iloc[:, 0]
all_cities = pd.concat([source_cities, destination_cities])

city_counts = all_cities.value_counts()
print("City counts:\n", city_counts)
city_dictionary = {}

def find_location(city):
    url = f"https://api.geoapify.com/v1/geocode/search?text={city}&lang=en&limit=1&type=city&format=json&apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching data for {city}. Status code: {response.status_code}")
        return None

    json_data = response.json()
    return decrypt_json(json_data)

def decrypt_json(json_data):
    if "results" in json_data and len(json_data["results"]) > 0:
        first_result = json_data["results"][0]
        latitude = first_result.get("lat")
        longitude = first_result.get("lon")
        print(f"Latitude: {latitude}, Longitude: {longitude}")
        return latitude, longitude
    else:
        print("No results found in the JSON response.")
        return None

def main():
    top_cities = city_counts.index.tolist()
    random_cities = random.sample(top_cities, 6)
    for city in random_cities:
        if city not in city_dictionary:
            coords = find_location(city)
            if coords:
                city_dictionary[city] = coords
                print(f"{city}: {coords}")

if __name__ == "__main__":
    main()
