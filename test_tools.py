"""checks if the tools return the desired information """
from tools import search_bathing_spots, get_weather
 

print("=" * 60)
print("TEST 1: search without city filter")
print("=" * 60)
result = search_bathing_spots.invoke({"query": "ruhiger Naturstrand"})
print(result)


print("\n" + "=" * 60)
print("TEST 2: search with city filter")
print("=" * 60)
result = search_bathing_spots.invoke({"query": "schwimmen", "city": "Konstanz"})
print(result)

print("\n" + "=" * 60)
print("TEST 3: weather in Überlingen")
print("=" * 60)
result = get_weather.invoke({"location": "Überlingen"})
print(result)

print("\n" + "=" * 60)
print("TEST 4: weather for an unknown location")
print("=" * 60)
result = get_weather.invoke({"location": "Tokyo"})
print(result)