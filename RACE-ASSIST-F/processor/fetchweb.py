"""
Load REVA RACE programme pages with Selenium, wait for JS,
then save all visible text into .txt files.
"""

import time
from typing import Dict

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Programme URLs
PROGRAM_URLS: Dict[str, str] = {
    "home_page": "https://race.reva.edu.in/",
    "about": "https://race.reva.edu.in/about/",
    "contact": "https://race.reva.edu.in/contact/",
    # program
    "M.Sc. in Business Analytics": "https://race.reva.edu.in/pg-diploma-msc-in-business-analytics/",
    "M.Tech. in Artificial Intelligence": "https://race.reva.edu.in/pg-diploma-m-tech-in-artificial-intelligence/",
    "PG Diploma or M.Sc. in Artificial Intelligence": "https://race.reva.edu.in/pg-diploma-m-tech-ms-in-artificial-intelligence/",
    "M.Tech. in Cybersecurity": "https://race.reva.edu.in/m-tech-in-cybersecurity/",
    "PG Diploma or M.Sc. in Cybersecurity": "https://race.reva.edu.in/pg-diploma-m-tech-in-cyber-security/",
    "PG Diploma or M.Sc. in in Cloud Architecture": "https://race.reva.edu.in/msc-in-cloud-architecture-and-security/",
    # get certified 
    "Advanced Diploma in Cybersecurity and Privacy Management": "https://race.reva.edu.in/advanced-diploma-in-cybersecurity-and-privacy-management/",
    "Certified DevOps Specialist with Terraform, Kubernetes, Jenkins, DevSecOps, and AIOps": "https://race.reva.edu.in/certified-devops-specialist-with-terraform-kubernetes-and-docker",
    "Certified Ethical Hacker": "https://race.reva.edu.in/certified-ethical-hacker-ceh/",
    "Certified Pentesting Professional": "https://race.reva.edu.in/certified-penetration-testing-professional-cpent/",
    "Certified AI Engineer": "https://race.reva.edu.in/certified-ai-engineer/"
}


def setup_driver() -> webdriver.Chrome:
    options = Options()
    # options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def scrape_full_text(urls: Dict[str, str]) -> None:
    driver = setup_driver()
    try:
        for key, url in urls.items():
            print(f"Processing {key} -> {url}")
            driver.get(url)

            # Wait for body + some known element to ensure JS has loaded
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(5)  # let late JS finish

            # Expand accordions (curriculum sections)
            driver.execute_script(
                "document.querySelectorAll('.vc_tta-panel .vc_tta-panel-body')"
                ".forEach(el => el.style.display='block');"
            )
            time.sleep(1)

            # Extract all text
            soup = BeautifulSoup(driver.page_source, "html.parser")
            text = soup.get_text("\n", strip=True)

            with open(f"{key}_full.txt", "w", encoding="utf-8") as f:
                f.write(text)

            print(f"  Saved {key}_full.txt")
    finally:
        driver.quit()


def main():
    scrape_full_text(PROGRAM_URLS)


if __name__ == "__main__":
    main()
