import pandas as pd
import json

try:

    try:
        census_df = pd.read_excel("census_transpose.xlsx")
        print("Census file loaded successfully")
    except Exception as e:
        print(f"Error loading census file: {e}")

    try:
        crime_df = pd.read_excel("crime.xlsx")
        print("Crime file loaded successfully")
    except Exception as e:
        print(f"Error loading crime file: {e}")

    try:
        with open("real_estate_data.json", "r") as file:
            real_estate_data = json.load(file)
        print("Real estate JSON loaded successfully")
    except Exception as e:
        print(f"Error loading real estate file: {e}")

    real_estate_df = pd.json_normalize(real_estate_data)

    census_df.columns = [col.replace("*", "").strip() for col in census_df.columns]
    crime_df.columns = [col.replace("*", "").strip() for col in crime_df.columns]

    crime_df["HOOD_158"] = crime_df["HOOD_158"].astype(str).str.lstrip("0").astype(int)

    real_estate_df = real_estate_df.explode("area_code")
    real_estate_df["area_code"] = (
        real_estate_df["area_code"].astype(str).str.lstrip("0").astype(int)
    )

    crime_df = crime_df.rename(columns={"HOOD_158": "Neighbourhood Number"})
    real_estate_df = real_estate_df.rename(
        columns={"area_code": "Neighbourhood Number"}
    )

    real_estate_simple_df = real_estate_df[
        ["Neighbourhood Number", "average_property_price"]
    ]

    merged_df = pd.merge(census_df, crime_df, on="Neighbourhood Number", how="left")

    final_df = pd.merge(
        merged_df, real_estate_simple_df, on="Neighbourhood Number", how="left"
    )

    try:
        final_df.to_excel("toronto.xlsx", index=False, engine="openpyxl")
        print("Data successfully exported to toronto.xlsx using openpyxl engine")
    except Exception as e:
        print(f"Error with openpyxl engine: {e}")
        try:
            final_df.to_excel("toronto.xlsx", index=False, engine="xlsxwriter")
            print("Data successfully exported to toronto.xlsx using xlsxwriter engine")
        except Exception as e2:
            print(f"Error with xlsxwriter engine: {e2}")
            final_df.to_csv("toronto.csv", index=False)
            print("Exported to CSV instead as toronto.csv")

except Exception as e:
    print(f"General error: {e}")
