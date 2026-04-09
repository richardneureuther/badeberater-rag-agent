"""
Scrape bodensee.de swimming spots into a JSONL file for RAG.

Pipeline:
  1. Fetch the overview page and collect all detail-page URLs.
  2. For each detail page: extract title, subtitle, description,
     contact, opening hours, prices.
  3. Build one JSON object per spot with a `text` field (what the
     RAG will embed) plus structured metadata, and write JSONL.
"""
import json
import re
import time
from urllib.parse import urljoin


import requests
from bs4 import BeautifulSoup

BASE = "https://www.bodensee.de"
OVERVIEW = f"{BASE}/erleben/baden-im-bodensee"
HEADERS = {"User-Agent": "Mozilla/5.0 (RAG-scraper; contact:xxxxxxxxxxxxxxx)"} #redacted for github
OUT = "bodensee_swimming.jsonl"

        
def get_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


#collect detail urls 
def collect_detail_urls() -> list[str]:
    soup = get_soup(OVERVIEW)
    urls = set()
    prefix = "/erleben/baden-im-bodensee/"
    skip = {
        "/erleben/baden-im-bodensee",
        "/erleben/baden-im-bodensee/barrierefreies-baden-am-bodensee",
    }
    for a in soup.select("a[href]"):
        href = a["href"]
        if href.startswith(prefix) and href.rstrip("/") not in skip:
            urls.add(urljoin(BASE, href.split("?")[0].rstrip("/")))
    return sorted(urls)


#parse one detail page 
def clean(txt: str) -> str:
    return re.sub(r"\s+", " ", txt).strip()

#if the price or time is a placeholder or refers to an external soruce detect it 
def is_placeholder(text: str) -> bool:
    if not text or len(text) < 5:
        return True
    # Real data contains concrete tokens: digits, euro, time markers
    has_real_content = bool(re.search(r"\d|€|uhr", text.lower()))
    return not has_real_content


#parse the website 
def parse_detail(url: str) -> dict:
    soup = get_soup(url)

    #get title
    h1 = soup.find("h1")
    title, subtitle = "", ""
    if h1:
        strong = h1.find("strong")
        if strong:
            title = clean(strong.get_text())
            subtitle = clean(h1.get_text().replace(strong.get_text(), ""))
        else:
            title = clean(h1.get_text())

    main = soup.find("main") or soup

    #get the description paragraph
    accordion = main.find("section", class_="c-accordion-container-text")
    description_parts = []
    for p in main.find_all("p"):
        #detecgt the accordion for prices etc
        if accordion and (p in accordion.descendants or
                          (accordion in list(p.find_all_previous("section")))):
            continue
        text = clean(p.get_text())
        if not text:
            continue
        #skip website noise
        if any(s in text.lower() for s in ("newsletter", "impressum", "cookie",
                                            "geprüfte unterkünfte", "sichere buchung")):
            continue
        description_parts.append(text)
    description = "\n\n".join(description_parts)

    #parse the accordion part to get opening ours, prices etc
    sections = {}
    if accordion:
        for panel in accordion.select("div.panel.panel-default"):
            title_el = panel.select_one(".panel-title")
            body_el = panel.select_one(".panel-body")
            if not title_el or not body_el:
                continue
            heading = clean(title_el.get_text()).lower()
            # body text: preserve line breaks as " | " so tables stay readable
            body_text = clean(body_el.get_text(" | ", strip=True))
            sections[heading] = body_text

    def pick(*keys):
        for k in keys:
            for h, v in sections.items():
                if k in h:
                    return v
        return ""

    kontakt         = pick("kontakt")
    oeffnungszeiten = pick("öffnungszeit", "oeffnungszeit")
    preise          = pick("eintrittspreis", "preise", "eintritt")

    #extraxt contact info from the info blob
    phone = ""
    m = re.search(r"(\+?\d[\d\s()/-]{6,}\d)", kontakt)
    if m:
        phone = clean(m.group(1))

    plz_ort = ""
    m = re.search(r"\b(\d{4,5})\s+([A-ZÄÖÜ][\wäöüß\-\. ]+)", kontakt)
    if m:
        plz_ort = f"{m.group(1)} {m.group(2).strip()}"


    #assemble all info for embedding
    text_blocks = [f"# {title}"]
    if subtitle:        text_blocks.append(subtitle)
    if description:     text_blocks.append(description)
    if kontakt:         text_blocks.append(f"Kontakt: {kontakt}")
    if oeffnungszeiten: text_blocks.append(f"Öffnungszeiten: {oeffnungszeiten}")
    if preise:          text_blocks.append(f"Eintrittspreise: {preise}")
    full_text = "\n\n".join(text_blocks)

    #if we have parsed a placeholder, replace it with an empty string
    if is_placeholder(oeffnungszeiten):
        oeffnungszeiten = ""
    if is_placeholder(preise):
         preise = ""


    return {
        "id": url.rsplit("/", 1)[-1],
        "url": url,
        "title": title,
        "subtitle": subtitle,
        "description": description,
        "kontakt": kontakt,
        "telefon": phone,
        "plz_ort": plz_ort,
        "oeffnungszeiten": oeffnungszeiten,
        "eintrittspreise": preise,
        "text": full_text,
        "source": "bodensee.de",
        "language": "de",
    }


#drive everything and store it in a jsonl
def main():
    print("Step 1: collecting detail URLs ...")
    urls = collect_detail_urls()
    print(f"  found {len(urls)} detail pages")

    print("Step 2: scraping each page ...")
    records = []
    for i, url in enumerate(urls, 1):
        try:
            rec = parse_detail(url)
            records.append(rec)
            print(f"  [{i:02d}/{len(urls)}] {rec['title'][:60]}")
        except Exception as e:
            print(f"  [{i:02d}/{len(urls)}] FAILED {url}: {e}")
        time.sleep(0.5)  # be polite

    print(f"Step 3: writing {OUT}")
    with open(OUT, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Done. {len(records)} records written.")

#run
if __name__ == "__main__":
    main()
