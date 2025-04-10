import requests
from bs4 import BeautifulSoup
import json


def scrape_real_estate_data():
    url = "https://toronto.listing.ca/real-estate-prices-by-community.htm"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    tables = soup.find_all("table")
    links = []
    for table in tables:
        cells = table.find_all("td", style="width:33%;vertical-align:top;")
        for cell in cells:
            anchors = cell.find_all("a", href=True)
            for anchor in anchors:
                links.append(
                    {"community": anchor.get_text(strip=True), "href": anchor["href"]}
                )

    for link in links:
        community_url = link["href"]
        community_response = requests.get("https://toronto.listing.ca" + community_url)
        community_soup = BeautifulSoup(community_response.text, "html.parser")

        right_div = community_soup.find("div", id="right")
        if right_div:
            rboxes = right_div.find_all("div", class_="rbox")
            if len(rboxes) > 3:

                # Extract data from the "Average Property Price" section
                average_price_header = right_div.find(
                    "div", class_="rb_header", string="Average Property Price"
                )
                if average_price_header:
                    average_price_box = average_price_header.find_next_sibling(
                        "div", class_="rbox"
                    )
                    if average_price_box:
                        price_div = average_price_box.find(
                            "div",
                            style="text-align:center;font-size:30px;color:#008000;font-weight:bold;",
                        )
                        if price_div:
                            link["average_property_price"] = price_div.get_text(
                                strip=True
                            )

                        sub_prices = average_price_box.find_all(
                            "div", style="overflow:hidden;margin-top:3px;"
                        )
                        sub_price_data = {}
                        for sub_price in sub_prices:
                            category = sub_price.find("div", style="float:left;")
                            value = sub_price.find(
                                "div", style="float:right;color:#008000;"
                            )
                            if category and value:
                                sub_price_data[category.get_text(strip=True)] = (
                                    value.get_text(strip=True)
                                )
                        link["sub_property_prices"] = sub_price_data

                # Extract data from the "reg-blur" section
                reg_blur_div = right_div.find("div", class_="reg-blur")
                if reg_blur_div:
                    table = reg_blur_div.find("table", style="width:100%;")
                    if table:
                        rows = table.find_all("tr")[1:]  # Skip the header row
                        reg_blur_data = {}
                        for row in rows:
                            columns = row.find_all("td")
                            if len(columns) == 3:
                                period = columns[0].get_text(strip=True)
                                change = columns[1].get_text(strip=True)
                                percent_change = columns[2].get_text(strip=True)
                                reg_blur_data[period] = {
                                    "change": change,
                                    "percent_change": percent_change,
                                }
                        link["historical_performance"] = reg_blur_data

    # Export the data as JSON
    with open("real_estate_data.json", "w") as json_file:
        json.dump(links, json_file, indent=4)


if __name__ == "__main__":
    scrape_real_estate_data()
