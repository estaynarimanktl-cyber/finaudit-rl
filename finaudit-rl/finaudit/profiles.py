"""Fictional company profiles and chart of accounts."""

from __future__ import annotations

COMPANIES = [
    "Aral Meridian Logistics LLP",
    "Zhana Talap Trading JSC",
    "Steppe Orchard Foods LLP",
    "Nomad Circuit Systems",
    "Tien Shan Components LLP",
    "Caspian Ledger Services",
    "Saryarka Textile Group",
    "Alatau Precision Works",
    "Koktem Health Supplies",
    "Orken Digital Infrastructure",
    "Ulytau Industrial Supply",
    "Zhaiyk River Transport",
]

ENTITIES = [
    ("ENT-A", "Head Office"),
    ("ENT-B", "Northern Branch"),
    ("ENT-C", "Western Branch"),
]

COUNTERPARTIES = [
    "Atyrau Marine Supply",
    "Bereke Retail",
    "Caspian Freight",
    "Dala Agro",
    "Emba Metals",
    "Zhaiyk Energy",
    "Turan Office",
    "Ulytau Mining",
    "Sarybel Foods",
    "Kaskad IT",
    "Alem Pharma",
    "Batys Machinery",
    "Nurly Logistics",
    "Sunkar Aviation",
    "Tobol Chemicals",
    "Orken Facilities",
    "Korgan Trade",
    "Aktobe Systems",
]

LINE_ITEMS = {
    "sale": [
        "Freight services",
        "Warehouse handling",
        "Consulting hours",
        "Software licence",
        "Equipment rental",
        "Transport surcharge",
        "Customs brokerage",
        "Maintenance contract",
        "Packaging services",
        "Delivery service",
    ],
    "purchase": [
        "Equipment parts",
        "Packaging materials",
        "Maintenance visit",
        "Safety inspection",
        "Office supplies",
        "Fuel and lubricants",
        "IT support",
        "Insurance premium",
        "Utilities",
        "Professional fees",
    ],
}

ACCOUNTS = {
    "1000": {"name": "Cash and bank", "type": "asset", "normal": "debit"},
    "1100": {"name": "Accounts receivable", "type": "asset", "normal": "debit"},
    "1200": {"name": "Prepaid expenses", "type": "asset", "normal": "debit"},
    "2000": {"name": "Accounts payable", "type": "liability", "normal": "credit"},
    "2100": {"name": "Accrued expenses", "type": "liability", "normal": "credit"},
    "2200": {"name": "Customer credits", "type": "liability", "normal": "credit"},
    "3000": {"name": "Share capital and equity", "type": "equity", "normal": "credit"},
    "4000": {"name": "Revenue", "type": "income", "normal": "credit"},
    "5000": {"name": "Operating expenses", "type": "expense", "normal": "debit"},
    "5100": {"name": "FX gain or loss", "type": "income", "normal": "credit"},
    "9999": {"name": "Suspense and clearing", "type": "asset", "normal": "debit"},
}

PERIODS = [
    ("2026-01-01", "2026-03-31", "Q1-2026"),
    ("2026-04-01", "2026-06-30", "Q2-2026"),
    ("2026-07-01", "2026-09-30", "Q3-2026"),
    ("2026-10-01", "2026-12-31", "Q4-2026"),
]

TIMEZONES = ["Asia/Almaty", "UTC", "Europe/Berlin", "Asia/Dubai", "Asia/Qyzylorda"]

BASE_CURRENCIES = ["USD", "EUR", "KZT"]
FOREIGN_CURRENCIES = {
    "USD": ["EUR", "GBP", "KZT", "AED", "JPY"],
    "EUR": ["USD", "GBP", "KZT", "AED", "JPY"],
    "KZT": ["USD", "EUR", "GBP", "AED", "JPY"],
}
