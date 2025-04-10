import json


def parse_price(price_str):
    """Convert price string to integer."""
    price_str = price_str.replace("$", "").replace(",", "").replace("+", "").strip()
    if not price_str or price_str == "-":
        return 0
    if "k" in price_str:
        return int(float(price_str.replace("k", "")) * 1000)
    elif "m" in price_str:
        return int(float(price_str.replace("m", "")) * 1000000)
    else:
        return int(float(price_str))


def format_real_estate_data(file_path):
    """Read JSON file and format the data."""
    with open(file_path, "r") as file:
        data = json.load(file)

    for item in data:

        item["average_property_price"] = parse_price(item["average_property_price"])

        for key, value in item["sub_property_prices"].items():
            item["sub_property_prices"][key] = parse_price(value)

        for key, value in item["historical_performance"].items():
            value["change"] = parse_price(value["change"])
            percent_change_str = (
                value["percent_change"].replace("%", "").replace("+", "").strip()
            )
            value["percent_change"] = round(
                float(percent_change_str) if percent_change_str != "-" else 0.0,
                3,
            )

        if "area_code" in item:
            item["area_code"] = [f"{int(code):03}" for code in item["area_code"]]

    return data


if __name__ == "__main__":
    input_file = "real_estate_data.json"
    formatted_data = format_real_estate_data(input_file)

    output_file = "real_estate_data.json"
    with open(output_file, "w") as file:
        json.dump(formatted_data, file, indent=4)

    print(f"Formatted data saved to {output_file}")
