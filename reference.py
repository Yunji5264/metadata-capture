import pandas as pd
import re
import os

DATA_DIR = "../data"
METADATA_DIR = "../metadata"
CATALOG_PATH = "../catalog.json"

DEFAULT_EXTS = {
    ".csv",       # comma-separated values
    ".tsv",       # tab-separated values
    ".xlsx",      # Excel (modern)
    ".xls",       # Excel (legacy)
    ".parquet",   # Parquet columnar format
    ".json",      # JSON (line-delimited or normal)
    ".geojson",   # GeoJSON
    ".zip",       # zipped Shapefiles
    ".shp",       # raw Shapefile (optional: if user has unzipped)
}
PERF_DIR = os.path.join(METADATA_DIR, "perf")
CACHE_DIR = os.path.join(METADATA_DIR, "_df_cache")

REF_SEMANTIC = "../ref/ref_semantic"

pairs = ["reg","academie","dep","arr_dep","epci","canton","com","com_arr","iris"]

ref_dict = {
    p: pd.read_csv(f"../ref/ref_spatial/{p}.csv", dtype=str) for p in pairs
}

HIER = {
    "spatial": [
        [
            "region",
            "academie",
            "departement",
            "arrondissement_departemental",
            "canton",
            "commune",
            "arrondissement_communal",
            "iris",
            ("geometry", "wkt_geojson"),
            "latlon_pair",
            "address",
        ],
        [
            "epci",
            "commune",
            "arrondissement_communal",
            "iris",
            ("geometry", "wkt_geojson"),
            "latlon_pair",
            "address",
        ],
    ],
    "temporal": [["year", "quarter", "month", "date"],["year", "week", "date"]]
}

SPATIAL_NAME_MAP = {
    "reg": "region",
    "academie": "academie",
    "dep": "departement",
    "arr_dep": "arrondissement_departemental",
    "epci": "epci",
    "canton": "canton",
    "com": "commune",
    "com_arr": "arrondissement_communal",
    "iris": "iris",
    "geometry": "geometry",
    "wkt_geojson": "wkt_geojson",
    "latlon_pair": "latlon_pair",
    "address": "address",
}

THEME_FOLDER_STRUCTURE = {
    "Well-being": {
        "Current Well-being": {
            "Health": {
                "Physical health": {
                    "Pain and discomfort": {},
                    "Energy and fatigue": {},
                    "Sleep and rest": {},
                    "Longevity & survival": {
                        "Life expectancy": {},
                        "Mortality": {}
                    },
                    "Disease burden": {
                        "Chronic conditions": {},
                        "Overweight & obesity": {},
                        "Risk behaviours": {
                            "Smoking & tobacco": {},
                            "Harmful alcohol use": {},
                            "Physical inactivity": {},
                            "Diet & nutrition": {}
                        }
                    }
                },
                "Mental health": {
                    "Positive feelings": {},
                    "Thinking, learning, memory and concentration": {
                        "Speed": {},
                        "Clarity": {}
                    },
                    "Self-esteem": {},
                    "Body image and appearance": {},
                    "Negative feelings": {},
                    "Mental state": {
                        "Emotional well-being": {},
                        "Cognitive function": {}
                    },
                    "Suicide & self-harm": {}
                },
                "Access to care": {
                    "Financial accessibility": {},
                    "Service availability": {
                        "Geographic accessibility": {},
                        "Coverage": {}
                    }
                },
                "Health systems & services": {
                    "Expenditure & financing": {},
                    "Workforce & resources": {},
                    "Utilisation & access": {},
                    "Quality & outcomes": {},
                    "Pharmaceuticals & medicines": {},
                    "Preventive services & screening": {},
                    "Health inequalities": {}
                },
                "Level of independence": {
                    "Mobility": {},
                    "Activities of daily living": {
                        "Taking care of oneself": {},
                        "Managing belongings appropriately": {}
                    },
                    "Dependence on medication and medical aids": {}
                }
            },
            "Education & Skills": {
                "Educational outcomes": {
                    "Attainment": {
                        "Years of schooling": {},
                        "Upper secondary attainment": {},
                        "Completion": {}
                    },
                    "Performance": {
                        "Literacy": {},
                        "Numeracy & science": {}
                    }
                },
                "Skills & learning": {
                    "Lifelong learning": {
                        "Adult education": {},
                        "Training opportunities": {}
                    },
                    "Skills level": {
                        "Digital skills": {},
                        "Employability skills": {}
                    }
                }
            },
            "Income & Wealth": {
                "Income": {
                    "Household income": {},
                    "Distribution": {
                        "Inequality": {},
                        "Poverty": {},
                        "Income inequality (Gini)": {},
                        "Relative poverty rate": {}
                    }
                },
                "Wealth": {
                    "Net wealth": {},
                    "Economic security": {
                        "Financial resilience": {},
                        "Perceived security": {},
                        "Feeling of having enough": {}
                    }
                }
            },
            "Jobs & Earnings": {
                "Employment quantity": {
                    "Participation": {
                        "Employment": {},
                        "Labour force participation": {}
                    },
                    "Unemployment": {
                        "General unemployment": {},
                        "Long-term unemployment": {}
                    }
                },
                "Job quality": {
                    "Wage level": {},
                    "Stability": {
                        "Contract type": {},
                        "Job security": {}
                    },
                    "Working conditions": {
                        "Occupational safety": {},
                        "Job satisfaction": {}
                    }
                },
                "Additional aspects": {
                    "Youth NEET rate": {},
                    "Informal employment rate": {},
                    "Work capacity": {}
                }
            },
            "Housing": {
                "Housing conditions": {
                    "Overcrowding": {},
                    "Facilities": {}
                },
                "Housing affordability": {
                    "Cost burden": {},
                    "Homelessness": {}
                }
            },
            "Environment Quality": {
                "Environmental exposure": {
                    "Air quality": {},
                    "Noise exposure": {}
                },
                "Perceptions & access": {
                    "Environmental satisfaction": {},
                    "Green space accessibility": {}
                },
                "Domestic environment": {
                    "Crowding": {},
                    "Available space": {},
                    "Cleanliness": {},
                    "Opportunities for privacy": {},
                    "Available equipment": {},
                    "Building construction quality": {}
                },
                "Basic services & utilities": {
                    "Transport": {},
                    "Drinking water": {},
                    "Gas": {},
                    "Electricity": {},
                    "Sewage networks": {}
                },
                "Urbanisation level": {},
                "Comfort and security": {}
            },
            "Safety": {
                "Personal safety": {
                    "Homicide and assault": {},
                    "Crime incidence": {},
                    "Perceived safety": {}
                },
                "Road safety": {
                    "Traffic injuries": {},
                    "Transport infrastructure safety": {}
                }
            },
            "Civic Engagement & Governance": {
                "Participation": {
                    "Electoral participation": {},
                    "Voter turnout": {},
                    "Civic participation (consultation, petitions)": {}
                },
                "Trust & satisfaction": {
                    "Institutional trust": {},
                    "Public service satisfaction": {},
                    "Access to justice": {},
                    "Perceived corruption": {}
                }
            },
            "Social Connections": {
                "Social support": {
                    "Reliance network": {},
                    "Help in times of need": {},
                    "Loneliness": {}
                },
                "Social participation": {
                    "Community participation": {},
                    "Informal care": {}
                },
                "Personal relations": {},
                "Sexual activity": {}
            },
            "Subjective Well-being": {
                "Life satisfaction": {},
                "Affective balance": {
                    "Positive vs negative emotions": {},
                    "Positive affect": {},
                    "Negative affect": {}
                }
            },
            "Work-life Balance": {
                "Long working hours": {},
                "Commuting time": {},
                "Unpaid work": {},
                "Leisure time": {},
                "Childcare availability": {},
                "Time use balance": {}
            },
            "Spirituality / Religion / Personal Beliefs": {}
        },
        "Resources for Future Well-being": {
            "Natural Capital": {
                "Ecosystems & biodiversity": {
                    "Protected areas": {},
                    "Forest cover": {},
                    "Species conservation": {}
                },
                "Climate & sustainability": {
                    "Emissions": {},
                    "Renewable energy": {},
                    "Freshwater resources": {},
                    "Green & blue infrastructure": {}
                }
            },
            "Human Capital": {
                "Health stock": {
                    "Longevity": {},
                    "Child development": {}
                },
                "Education & skills stock": {
                    "Higher education": {},
                    "Foundational skills": {},
                    "Adult skills": {}
                }
            },
            "Social Capital": {
                "Trust & norms": {
                    "Interpersonal trust": {},
                    "Institutional trust": {}
                },
                "Inclusion & cohesion": {
                    "Gender equality": {},
                    "Anti-discrimination": {},
                    "Civic inclusion": {}
                }
            },
            "Economic & Produced Capital": {
                "Infrastructure & innovation": {
                    "Fixed capital": {},
                    "Infrastructure quality": {},
                    "Innovation capacity": {}
                },
                "Wealth sustainability": {
                    "Adjusted savings": {},
                    "Resource depletion": {}
                }
            }
        }
    },
    # "Ageing-friendly": {
    #     "Outdoor Environment & Mobility": {
    #         "Physical environment": {
    #             "Walkability": {
    #                 "Pavement & sidewalks": {},
    #                 "Street crossings": {}
    #             },
    #             "Accessibility of public spaces": {
    #                 "Parks & green spaces": {},
    #                 "Seating & rest areas": {}
    #             }
    #         },
    #         "Transportation": {
    #             "Public transport": {
    #                 "Affordability": {},
    #                 "Reliability": {}
    #             },
    #             "Mobility services": {
    #                 "Paratransit": {},
    #                 "Community transport": {}
    #             }
    #         }
    #     },
    #     "Housing & Living Environment": {
    #         "Housing design": {
    #             "Accessibility": {
    #                 "Step-free access": {},
    #                 "Adaptable interior": {}
    #             },
    #             "Safety & comfort": {
    #                 "Thermal comfort": {},
    #                 "Safety devices": {}
    #             }
    #         },
    #         "Affordability & availability": {
    #             "Affordability": {},
    #             "Availability": {}
    #         }
    #     },
    #     "Social Participation": {
    #         "Cultural & recreational opportunities": {
    #             "Venue access": {},
    #             "Activity diversity": {}
    #         },
    #         "Community participation": {
    #             "Intergenerational": {},
    #             "Inclusive participation": {}
    #         }
    #     },
    #     "Respect & Social Inclusion": {
    #         "Attitudes towards older people": {
    #             "Societal attitudes": {},
    #             "Representation": {}
    #         },
    #         "Interpersonal relations": {
    #             "Family relations": {},
    #             "Community relations": {}
    #         }
    #     },
    #     "Civic Participation & Employment": {
    #         "Employment opportunities": {
    #             "Work flexibility": {},
    #             "Learning & skills": {}
    #         },
    #         "Civic engagement": {
    #             "Volunteering": {},
    #             "Political participation": {}
    #         }
    #     },
    #     "Communication & Information": {
    #         "Communication channels": {
    #             "Traditional media": {},
    #             "Digital inclusion": {}
    #         },
    #         "Information delivery": {
    #             "Readability": {},
    #             "Availability": {}
    #         }
    #     },
    #     "Community Support & Health Services": {
    #         "Community support": {
    #             "Social care": {},
    #             "Home help": {}
    #         },
    #         "Health services": {
    #             "Primary care": {},
    #             "Long-term & palliative care": {
    #                 "Long-term care": {},
    #                 "Palliative care": {}
    #             }
    #         }
    #     },
    #     "Security & Safety": {
    #         "Personal safety": {
    #             "Crime prevention": {},
    #             "Emergency response": {}
    #         },
    #         "Financial security": {
    #             "Income security": {},
    #             "Consumer protection": {}
    #         }
    #     }
    # }
}






THEME_FOLDER_STRUCTURE_old = {
    "Well-Being": {
        "Physical Health": {
            "Pain and discomfort": {},
            "Energy and fatigue": {},
            "Sleep and rest": {}
        },
        "Psychological Health": {
            "Positive feelings": {},
            "Thinking, learning, memory and concentration": {
                "Speed": {},
                "Clarity": {}
            },
            "Self-esteem": {},
            "Body image and appearance": {},
            "Negative feelings": {}
        },
        "Level of Independence": {
            "Mobility": {},
            "Activities of daily living": {
                "Taking care of oneself": {},
                "Managing one's belongings appropriately": {}
            },
            "Dependence on medication and medical aids": {},
            "Education and Skills": {
                "Years of schooling": {},
                "Upper secondary attainment": {},
                "Foundational skills": {
                    "Literacy": {},
                    "Numeracy": {},
                    "Digital skills": {}
                },
                "Lifelong learning": {
                    "Adult education": {},
                    "Training opportunities": {}
                }
            },
            "Income and Wealth": {
                "Household income": {},
                "Wealth distribution": {},
                "Income inequality (Gini)": {},
                "Relative poverty rate": {},
                "Financial resources": {
                    "Independence": {},
                    "Feeling of having enough": {}
                }
            },
            "Jobs and Employment": {
                "Employment rate": {},
                "Unemployment rate": {},
                "Job quality": {},
                "Job security": {},
                "Youth NEET rate": {},
                "Informal employment rate": {},
                "Work capacity": {}
            },
            "Safety": {
                "Personal security (homicide, assault)": {},
                "Perceived safety": {},
                "Road safety (traffic injuries)": {}
            }
        },
        "Civic Engagement and Governance": {
            "Political participation": {
                "Voter turnout": {},
                "Civic participation (consultation, petitions)": {}
            },
            "Governance quality": {
                "Trust in institutions": {},
                "Access to justice": {},
                "Perceived corruption": {}
            }
        },
        "Subjective Well-being": {
            "Life evaluation": {
                "Life satisfaction": {}
            },
            "Affect balance": {
                "Positive vs negative emotions": {}
            }
        },
        "Work-Life Balance": {
            "Working hours (long hours incidence)": {},
            "Leisure time": {},
            "Childcare availability": {},
            "Time use balance": {}
        },
        "Environment": {
            "Comfort and security": {},
            "Domestic environment": {
                "Crowding": {},
                "Available space": {},
                "Cleanliness": {},
                "Opportunities for privacy": {},
                "Available equipment": {},
                "Building construction quality": {}
            },
            "Health care and social care": {
                "Accessibility": {},
                "Quality": {}
            },
            "Opportunities to acquire new information and skills": {},
            "Participation in recreational and leisure activities": {},
            "Physical environment": {
                "Air pollution (PM2.5 exposure)": {},
                "Noise": {},
                "Traffic": {},
                "Climate": {},
                "Access to green space": {}
            },
            "Infrastructure": {
                "Transport": {},
                "Drinking water": {},
                "Gas": {},
                "Electricity": {},
                "Sewage networks": {}
            },
            "Urbanisation level": {}
        },
        "Social Relationships": {
            "Personal relations": {},
            "Social support (help in times of need)": {},
            "Social isolation / loneliness": {},
            "Sexual activity": {}
        },
        "Spirituality/Religion/Personal Beliefs": {}
    }
}


# Ordered patterns from more specific to less specific; ISO-like first.
TEMP_NAME_PATTERNS = [
    ("quarter", re.compile(r"\b(?P<y>(18|19|20)\d{2})[\-_ ]?Q(?P<q>[1-4])\b", re.I)),
    ("quarter", re.compile(r"\bQ(?P<q>[1-4])[\-_ ]?(?P<y>(18|19|20)\d{2})\b", re.I)),
    ("semester", re.compile(r"\bS(?P<s>[12])[\-_ ]?(?P<y>(18|19|20)\d{2})\b", re.I)),
    ("semester", re.compile(r"\b(?P<y>(18|19|20)\d{2})[\-_ ]?S(?P<s>[12])\b", re.I)),
    ("week",    re.compile(r"\b(?P<y>(18|19|20)\d{2})[\-_ ]?W(?P<w>[0-5]\d)\b", re.I)),
    ("week",    re.compile(r"\bW(?P<w>[0-5]\d)[\-_ ]?(?P<y>(18|19|20)\d{2})\b", re.I)),
    # YYYY-MM / YYYY_M / YYYYMM
    ("month",   re.compile(r"\b(?P<y>(18|19|20)\d{2})[\-_/ ]?(?P<m>0?[1-9]|1[0-2])\b")),
    ("month",   re.compile(r"\b(?P<y>(18|19|20)\d{2})(?P<m>0[1-9]|1[0-2])\b")),
    # YYYY-MM-DD / YYYYMMDD
    ("date",     re.compile(r"\b(?P<y>(18|19|20)\d{2})[\-/_ ]?(?P<m>0[1-9]|1[0-2])[\-/_ ]?(?P<d>0[1-9]|[12]\d|3[01])\b")),
    # Year alone
    ("year",    re.compile(r"\b(?P<y>(18|19|20)\d{2})\b")),
]
