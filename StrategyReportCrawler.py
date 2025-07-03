import os
import re
import time
import json
import random
import requests
import math
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# --- 1. 全局配置 (策略研报) ---
SAVE_DIRECTORY = "eastmoney_strategy_reports"
# API模板 - qType=2 代表策略研报
API_URL_TEMPLATE = "https://reportapi.eastmoney.com/report/jg?cb=datatable&pageSize={pageSize}&beginTime={beginTime}&endTime={endTime}&pageNo={pageNo}&qType=2&_={timestamp}"
# 详情页模板
DETAIL_PAGE_URL_TEMPLATE = "https://data.eastmoney.com/report/zw_strategy.jshtml?encodeUrl={encodeUrl}"


# --- 2. 增强型稳健性配置 ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
]

# 代理IP配置 (如果不需要代理，保持列表为空即可)
# 格式: "协议://用户名:密码@IP地址:端口" 或 "协议://IP地址:端口"
# 例如: PROXIES = ["http://user:pass@127.0.0.1:7890", "https://127.0.0.1:7890"]
PROXIES = [] 

# 重试与延迟配置
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5 # 秒

# --- 3. 辅助函数 ---
def clean_filename(title: str) -> str:
    return re.sub(r'[\/\\:\*\?"<>\|]', '_', title).strip()

def get_random_proxy():
    """从代理列表中随机选择一个代理"""
    return random.choice(PROXIES) if PROXIES else None

def make_robust_request(session, url, retries=MAX_RETRIES, initial_delay=INITIAL_RETRY_DELAY):
    """
    带有指数退避重试和随机代理的增强型请求函数
    """
    for attempt in range(retries):
        try:
            proxy = get_random_proxy()
            proxies = {"http": proxy, "https": proxy} if proxy else None
            
            # 每次请求都更换User-Agent
            session.headers.update({'User-Agent': random.choice(USER_AGENTS)})

            response = session.get(url, timeout=20, proxies=proxies)
            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                print(f"  [!] 请求失败，已达最大重试次数: {e}")
                return None
            
            delay = initial_delay * (2 ** attempt)
            print(f"  [!] 请求失败: {e}. 将在 {delay} 秒后重试...")
            time.sleep(delay)
    return None

# --- 4. 主下载函数 ---
def download_strategy_reports():
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    print(f"所有【策略研报】将保存在: {os.path.abspath(SAVE_DIRECTORY)}")

    session = requests.Session()
    session.headers.update({'Referer': 'https://data.eastmoney.com/'})

    end_time = datetime.now()
    begin_time = end_time - timedelta(days=2*365)
    end_time_str = end_time.strftime('%Y-%m-%d')
    begin_time_str = begin_time.strftime('%Y-%m-%d')

    # --- 动态获取总页数 ---
    print("正在动态获取报告总数...")
    first_page_url = API_URL_TEMPLATE.format(pageSize=50, beginTime=begin_time_str, endTime=end_time_str, pageNo=1, timestamp=int(time.time() * 1000))
    first_response = make_robust_request(session, first_page_url)
    
    if not first_response:
        print("无法获取初始数据，程序退出。")
        return

    try:
        json_match = re.search(r'\((.*)\)', first_response.text)
        data = json.loads(json_match.group(1))
        total_hits = data.get('hits', 0)
        total_pages = math.ceil(total_hits / 50)
        print(f"检测到总共有 {total_hits} 篇报告, 共 {total_pages} 页。")
    except (ValueError, AttributeError, json.JSONDecodeError):
        print("解析初始数据失败，无法确定总页数，程序退出。")
        return

    # --- 主循环 ---
    for page in range(1, total_pages + 1):
        print(f"\n{'='*20} 正在处理第 {page}/{total_pages} 页 {'='*20}")
        
        api_url = API_URL_TEMPLATE.format(pageSize=50, beginTime=begin_time_str, endTime=end_time_str, pageNo=page, timestamp=int(time.time() * 1000))
        list_response = make_robust_request(session, api_url)

        if not list_response:
            continue
        
        try:
            json_match = re.search(r'\((.*)\)', list_response.text)
            data = json.loads(json_match.group(1))
            reports_on_page = data.get('data')

            if not reports_on_page:
                print("当前页没有更多数据。")
                continue

            for report in reports_on_page:
                title, encode_url, org_name, publish_date = (
                    report.get('title'), report.get('encodeUrl'), 
                    report.get('orgSName', '未知机构'), report.get('publishDate', '0000-00-00')[:10]
                )

                if not all([title, encode_url]): continue
                
                safe_title = clean_filename(title)
                filename = f"{publish_date}_{org_name}_{safe_title}.pdf"
                save_path = os.path.join(SAVE_DIRECTORY, filename)
                
                if os.path.exists(save_path): continue

                detail_url = DETAIL_PAGE_URL_TEMPLATE.format(encodeUrl=encode_url)
                detail_response = make_robust_request(session, detail_url)
                if not detail_response: continue

                soup = BeautifulSoup(detail_response.content, 'html.parser')
                pdf_link_element = soup.select_one("a.pdf-link")
                if not pdf_link_element or not pdf_link_element.get('href'): continue
                
                pdf_url = pdf_link_element['href']
                
                pdf_response = make_robust_request(session, pdf_url)
                if not pdf_response: continue
                
                total_size = int(pdf_response.headers.get('content-length', 0))
                with open(save_path, 'wb') as f, tqdm(desc=f"-> {safe_title[:25]}...", total=total_size, unit='B', unit_scale=True) as bar:
                    for chunk in pdf_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
                
                time.sleep(random.uniform(0.5, 1.0)) # 每个文件下载后的短暂延时
        except (ValueError, AttributeError, json.JSONDecodeError) as e:
            print(f"解析第 {page} 页数据时发生错误: {e}")
        
        time.sleep(random.uniform(1, 3)) # 每处理完一页，多休息一会儿

    print(f"\n{'='*20} 所有【策略研报】下载任务已完成 {'='*20}")

if __name__ == "__main__":
    download_strategy_reports()