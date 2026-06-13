import urllib.request
import re
import csv


def fetch_html(url, timeout=15):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")

# https://24h.pchome.com.tw/store/DSAA31?sortParm=new&sortOrder=dc&p=2


# 一直抓商品分頁，直到當前page沒有商品為止。
def get_pages_html():
    all_html = ""
    page = 1

    while True:
        url = f"https://24h.pchome.com.tw/store/DSAA31?p={page}"
        html = fetch_html(url)
        # 這頁已經沒有商品連結就停止
        if "c-prodInfoV2__link" not in html:
            break
        all_html += html
        page += 1

    return all_html


# 抓取所有html的商品id
def get_all_product_ids(all_html):

    #  <a class="c-prodInfoV2__link" href="..."> 的 href
    pattern = r'<a[^>]*class="[^"]*c-prodInfoV2__link[^"]*"[^>]*href="([^"]*)"'
    hrefs = re.findall(pattern, all_html, re.DOTALL)

    # 把 href split 成商品 ID
    return [href.split("/")[2].split("?")[0] for href in hrefs]


# 將商品存到文字檔中，每行一個。
def save_to_txt(data, filename):
    with open(filename, "w") as f:
        for id in data:
            f.write(f"{id}\n")


# 抓取商品評論的商品id
def get_all_product_reviews_ids(all_html, product_ids):

    #  <a class="c-prodInfoV2__link" href="...">   的內容
    pattern = r'<a[^>]*class="[^"]*c-prodInfoV2__link[^"]*"[^>]*>(.*?)</a>'
    blocks = re.findall(pattern, all_html, re.DOTALL)


    ids = []

    for index, block in enumerate(blocks):
        if 'c-ratingIcon__flex' in block:
            ids.append(product_ids[index])
    return ids


# 抓取商品評論大於4.9的商品id
def get_average_up_rating(reviews_ids):

    ids = []

    # 抓取美個頁面詳細資訊
    for index, review_id in  enumerate(reviews_ids):
        url = f"https://24h.pchome.com.tw/prod/{review_id}?fq=/S/{review_id.split('-')[0]}"
        html = fetch_html(url)
        
        # 抓取平均評分
        pattern = r'<div[^>]*class="[^"]*c-ratingIcon__textNumber[^"]*"[^>]*>(.*?)</div>'
        numbers = re.findall(pattern, html, re.DOTALL)

        if(float(numbers[0]) > 4.9):
            ids.append(review_id)
        

    return ids


def get_i5_processor_avg_prices(html):
    #  <a class="c-prodInfoV2__link" href="...">   的內容
    pattern = r'<a[^>]*class="[^"]*c-prodInfoV2__link[^"]*"[^>]*>(.*?)</a>'
    blocks = re.findall(pattern, html, re.DOTALL)


    total = 0
    number = 0

    for index, block in enumerate(blocks):
        if 'i5' in block:
            pattern = r'<div[^>]*class="[^"]*c-prodInfoV2__priceValue[^"]*"[^>]*>(.*?)</div>'
            prices = re.findall(pattern, block, re.DOTALL)
            price = int(prices[0].replace("$", "").replace(",", ""))
            total += price
            number += 1
            

    
    return total / number if number > 0 else 0

    # for index, block in enumerate(blocks):
    #     if 'c-ratingIcon__flex' in block:
    #         ids.append(product_ids[index])
    # return ids


def get_product_price_list(html):
    # Implementation for fetching product prices

    #  <a class="c-prodInfoV2__link" href="...">   的內容
    pattern = r'<a[^>]*class="[^"]*c-prodInfoV2__link[^"]*"[^>]*>(.*?)</a>'
    blocks = re.findall(pattern, html, re.DOTALL)

    prices = []
    for block in blocks:
        # Extract price from each block
        price_pattern = r'<div[^>]*class="[^"]*c-prodInfoV2__priceValue[^"]*"[^>]*>(.*?)</div>'
        price_matches = re.findall(price_pattern, block, re.DOTALL)
        price = int(price_matches[0].replace("$", "").replace(",", ""))
        prices.append(price)

    return prices



def save_to_csv(data, filename):

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)

def calculate_zScore_list(prices_list):

    # 平均數
    avg_price = sum(prices_list) / len(prices_list)
    # 標準差
    sd_price = 0

    for price in prices_list:
        sd_price += (price - avg_price) ** 2
    sd_price = (sd_price / len(prices_list)) ** 0.5


    # 計算 z-score
    # (X - 平均數) / 標準差
    zScores = []
    for price in prices_list:
        zScore = (price - avg_price) / sd_price
        zScores.append(zScore)
    
    return zScores



def format_product_data(product_id, product_price_list, zScores):
    # Implementation for formatting product data for CSV output
    formatted_data = []
    for index, product_id in enumerate(product_id):
        formatted_data.append([product_id, product_price_list[index], zScores[index]])
    return formatted_data

def  main():
    html = get_pages_html()

    # # 題目一開始

    # 取得所有product id
    product_ids = get_all_product_ids(html)
    save_to_txt(product_ids, "products.txt")
    # # 題目一結束

    # # 題目二開始
    reviews_ids = get_all_product_reviews_ids(html, product_ids)
    answersIDs = get_average_up_rating(reviews_ids)
    save_to_txt(answersIDs, "best-products.txt")
    # # 題目二結束

    # 題目三開始
    i5_avg_price =  get_i5_processor_avg_prices(html)
    print(f"i5 processor average price: {i5_avg_price}")
    # 題目三結束

    # 題目四開始
    product_price_list = get_product_price_list(html)
    zScores = calculate_zScore_list(product_price_list)
    formatted_data = format_product_data(product_ids, product_price_list, zScores)
    save_to_csv(formatted_data, "standardized.csv")
    # 題目四結束



if __name__ == "__main__":
    main()
