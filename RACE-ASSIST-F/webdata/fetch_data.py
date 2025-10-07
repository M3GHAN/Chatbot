import re
import time
from typing import Dict, List

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup

# Programme URLs
PROGRAM_URLS: Dict[str, str] = {
    "business_analytics": "https://race.reva.edu.in/pg-diploma-msc-in-business-analytics/",
    "mtech_ai": "https://race.reva.edu.in/pg-diploma-m-tech-in-artificial-intelligence",
    "msc_ai": "https://race.reva.edu.in/pg-diploma-m-tech-ms-in-artificial-intelligence/",
    "mtech_cybersecurity": "https://race.reva.edu.in/m-tech-in-cybersecurity/",
    "msc_cybersecurity": "https://race.reva.edu.in/pg-diploma-m-tech-in-cyber-security/",
    "msc_cloud_architecture": "https://race.reva.edu.in/msc-in-cloud-architecture-and-security",
}


def extract_section_by_heading(soup: BeautifulSoup, heading_text: str, max_depth: int = 3) -> str:
    """Find a heading by text and collect following sibling text until a same/higher-level heading."""
    pattern = re.compile(re.escape(heading_text), re.IGNORECASE)
    heading = None
    for level in range(1, max_depth + 1):
        heading = soup.find(f"h{level}", string=pattern)
        if heading:
            break
    if not heading:
        return ""
    texts: List[str] = []
    current_level = int(heading.name[1])
    for sib in heading.find_next_siblings():
        if sib.name and re.match(r"h[1-6]", sib.name):
            if int(sib.name[1]) <= current_level:
                break
        if sib.name in {"p", "li", "div", "span"}:
            t = sib.get_text(" ", strip=True)
            if t:
                texts.append(t)
    return "\n".join(texts)


def extract_admission_info(soup: BeautifulSoup) -> str:
    """Extract Admission Process section text."""
    admission_heading = soup.find(lambda tag: tag and "Admission Process" in tag.get_text())
    if not admission_heading:
        return ""
    texts: List[str] = []
    name = admission_heading.name if admission_heading.name else ""
    if re.match(r"h[1-6]", name):
        current_level = int(name[1])
    else:
        current_level = 7
    for sib in admission_heading.find_next_siblings():
        if sib.name and re.match(r"h[1-6]", sib.name):
            if int(sib.name[1]) <= current_level:
                break
        if sib.name in {"p", "li", "div", "span"}:
            t = sib.get_text(" ", strip=True)
            if t:
                texts.append(t)
    return "\n".join(texts)


def extract_curriculum_modules(soup: BeautifulSoup) -> str:
    """Extract semester-wise modules from accordion containers."""
    semesters = []
    for container in soup.select(".vc_tta-container"):
        sem_title_el = container.find(["h2", "h3"])
        if not sem_title_el:
            continue
        sem_title = sem_title_el.get_text(strip=True)

        modules = []
        for panel in container.select(".vc_tta-panel"):
            title_el = panel.select_one(".vc_tta-title-text")
            body_el = panel.select_one(".vc_tta-panel-body")
            title = title_el.get_text(strip=True) if title_el else "Untitled"
            body = body_el.get_text(" ", strip=True) if body_el else ""
            modules.append(f"- {title}\n  {body}")

        if modules:
            semesters.append(sem_title + "\n" + "\n".join(modules))
    return "\n\n".join(semesters)


def extract_program_info(soup: BeautifulSoup) -> str:
    """Extract Start Date, Duration, Recognition, and Program Fee."""
    info = []
    for col in soup.select(".eligibility-wrap .wpb_column"):
        title_el = col.find("h4")
        value_el = col.find("p")
        if title_el and value_el:
            title = title_el.get_text(strip=True)
            value = value_el.get_text(strip=True)
            info.append(f"{title}: {value}")
    return "\n".join(info)


def extract_sections_from_html(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    return {
        "Become an Analytics Consultant": extract_section_by_heading(soup, "Become an Analytics Consultant"),
        "Program Features": extract_section_by_heading(soup, "Program feature"),
        "Why Business Analytics with RACE": extract_section_by_heading(soup, "Why Business Analytics with RACE"),
        "Program Info": extract_program_info(soup),
        "Curriculum Details": extract_curriculum_modules(soup),
        "Admission Process": extract_admission_info(soup),
    }


def setup_driver() -> webdriver.Chrome:
    options = Options()
    # options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36")
    return webdriver.Chrome(options=options)


def scrape_program_pages(urls: Dict[str, str]) -> None:
    driver = setup_driver()
    try:
        for key, url in urls.items():
            print(f"Processing {key} -> {url}")
            driver.get(url)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(5)
            # Expand hidden accordions
            driver.execute_script(
                "document.querySelectorAll('.vc_tta-panel .vc_tta-panel-body').forEach(el => el.style.display='block');"
            )
            time.sleep(1)
            sections = extract_sections_from_html(driver.page_source)
            with open(f"{key}.txt", "w", encoding="utf-8") as f:
                for name, text in sections.items():
                    f.write(name + "\n")
                    f.write(text.strip() + "\n\n")
            print(f"  Saved {key}.txt")
    finally:
        driver.quit()


def main():
    scrape_program_pages(PROGRAM_URLS)


if __name__ == "__main__":
    main()
