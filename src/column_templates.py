# src/column_templates.py
"""
Declarative column templates for WA Bonds CSVs.
- Each table has: file_patterns (to select files), required and optional fields.
- Patterns can be exact names (post-standardisation) or regex via 're:<pattern>'.
- Canonical names used downstream: postcode, date, weekly_rent, days_held, period, bonds_held.
"""

SCHEMA_TEMPLATES = {
    "lodgements": {
        "file_patterns": [r"lodg"],
        "required": {
            "postcode": ["postcode", "post_code", "poa", "poa_code", "re:poa[_ ]?code"],
            "date": [
                "lodgement_date", "date_lodged", "bond_lodgement_date",
                "received_date", "date", "period", "re:lodg(e)?ment[_ ]?date"
            ],
            "weekly_rent": [
                "weekly_rent_amount", "weekly_rent", "rent", "weeklyrent",
                "re:weekly[_ ]?rent"
            ],
        },
        "optional": {
            "locality": ["locality_name", "locality", "suburb", "suburb_locality"]
        },
    },

    "disposals": {
        "file_patterns": [r"dispos", r"refund"],
        "required": {
            "postcode": ["postcode", "post_code", "poa", "poa_code", "re:poa[_ ]?code"],
            "date": [
                # <-- add the header you actually have:
                "disbursed_date",
                # common alternatives:
                "disposal_date", "refund_date", "end_date",
                "date", "period", "re:(dispos|refund).*date"
            ],
            "days_held": [
                # <-- add the header you actually have:
                "days_bond_held",
                # common alternatives:
                "days_held", "daysheld", "tenure_days", "lease_days", "tenancy_days",
                "re:(days|tenure).*held"
            ],
        },
        "optional": {},
    },

        "stock": {
            "file_patterns": [r"post.?code", r"stock", r"bonds_by_postcode"],
            "required": {
                "postcode": ["postcode", "post_code", "poa", "poa_code", "re:poa[_ ]?code"],
                "bonds_held": [
                    "bonds_held", "bonds", "stock_bonds", "count_bonds", "bondsheld",
                    "re:(bonds|stock).*held"
                ],
            },
            "optional": {
                "period": ["month", "period", "date", "re:(month|period|date)"]
            },
    },

}