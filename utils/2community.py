import json
import csv


real_estate_file = "real_estate_data.json"
neighbourhood_file = "neighbourhood_map.csv"

with open(real_estate_file, "r") as f:
    real_estate_data = json.load(f)

neighbourhood_data = {}
with open(neighbourhood_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        neighbourhood_data[row["Neighbourhood"]] = row["Neighbourhood_Num"]

for entry in real_estate_data:
    neighbourhood = entry.get("community")
    if neighbourhood in neighbourhood_data:
        if "area_code" not in entry or not isinstance(entry["area_code"], list):
            entry["area_code"] = []
        if neighbourhood_data[neighbourhood] not in entry["area_code"]:
            entry["area_code"].append(neighbourhood_data[neighbourhood])

with open(real_estate_file, "w") as f:
    json.dump(real_estate_data, f, indent=4)
