"""One-off generator for sample job postings (run from repo root if needed)."""
import csv
import random
from datetime import date, timedelta

random.seed(42)
start = date(2025, 1, 6)
weeks = 10
titles = [
    ("Senior Software Engineer", "python react aws kubernetes agile"),
    ("Data Scientist", "machine learning python sql deep learning"),
    ("DevOps Engineer", "kubernetes terraform aws jenkins devops"),
    ("Full Stack Developer", "javascript react node sql full stack"),
    ("Product Manager", "product management agile communication excel"),
    ("Data Engineer", "python sql etl snowflake databricks"),
    ("ML Engineer", "machine learning pytorch python cloud computing"),
    ("Backend Engineer", "java spring boot sql microservices"),
    ("Frontend Engineer", "javascript react typescript css"),
    ("Business Analyst", "excel power bi communication sql"),
    ("QA Engineer", "manual testing agile selenium"),
    ("Cloud Architect", "aws azure terraform kubernetes"),
    ("Security Analyst", "cybersecurity python aws"),
    ("Java Developer", "java spring boot sql"),
    ("HR Specialist", "communication excel"),
]
cities = ["Bangalore", "Hyderabad", "Pune", "Mumbai", "Remote"]
rows = []
domains = ["B2B", "fintech", "healthcare"]
for w in range(weeks):
    d0 = start + timedelta(weeks=w)
    for i in range(9):
        t, sk = random.choice(titles)
        dom = random.choice(domains)
        desc = f"We need {sk}. Team collaborates on modern stack. {dom} domain."
        loc = random.choice(cities)
        smin = random.choice([8, 10, 12, 15, 18, None])
        if smin:
            smax = smin + random.choice([4, 6, 8, 10])
        else:
            smax = ""
            smin = ""
        suf = random.choice(["", " II", " (Remote)"])
        rows.append(
            {
                "post_date": (d0 + timedelta(days=i)).isoformat(),
                "job_title": t + suf,
                "description": desc,
                "location": loc,
                "salary_min_lpa": smin if smin != "" else "",
                "salary_max_lpa": smax if smax != "" else "",
            }
        )

out = "data/market_pulse/job_postings_sample.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "post_date",
            "job_title",
            "description",
            "location",
            "salary_min_lpa",
            "salary_max_lpa",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} rows to {out}")
