application_schema = {
    "type": "record",
    "namespace": "com.example",
    "name": "Application",
    "fields": [
        {"name": "SK_ID_CURR", "type": "long"},
        {"name": "NAME_CONTRACT_TYPE", "type": "string"},
        {"name": "CODE_GENDER", "type": "string"},
        {"name": "FLAG_OWN_CAR", "type": "string"},
        {"name": "FLAG_OWN_REALTY", "type": "string"},
        {"name": "CNT_CHILDREN", "type": "int"},
        {"name": "AMT_INCOME_TOTAL", "type": "double"},
        {"name": "AMT_CREDIT", "type": "double"},
        {
            "name": "AMT_ANNUITY",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {"name": "AMT_GOODS_PRICE", "type": "double"},
        {
            "name": "NAME_TYPE_SUITE",
            "type": ["null", "string"],
            "default": None,
        },  # nullable
        {"name": "NAME_INCOME_TYPE", "type": "string"},
        {"name": "NAME_EDUCATION_TYPE", "type": "string"},
        {"name": "NAME_FAMILY_STATUS", "type": "string"},
        {"name": "NAME_HOUSING_TYPE", "type": "string"},
        {"name": "REGION_POPULATION_RELATIVE", "type": "double"},
        {"name": "DAYS_BIRTH", "type": "int"},
        {"name": "DAYS_EMPLOYED", "type": "int"},
        {"name": "DAYS_REGISTRATION", "type": "double"},
        {"name": "DAYS_ID_PUBLISH", "type": "int"},
        {
            "name": "OWN_CAR_AGE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {"name": "FLAG_MOBIL", "type": "int"},
        {"name": "FLAG_EMP_PHONE", "type": "int"},
        {"name": "FLAG_WORK_PHONE", "type": "int"},
        {"name": "FLAG_CONT_MOBILE", "type": "int"},
        {"name": "FLAG_PHONE", "type": "int"},
        {"name": "FLAG_EMAIL", "type": "int"},
        {
            "name": "OCCUPATION_TYPE",
            "type": ["null", "string"],
            "default": None,
        },  # nullable
        {"name": "CNT_FAM_MEMBERS", "type": "double"},
        {"name": "REGION_RATING_CLIENT", "type": "int"},
        {"name": "REGION_RATING_CLIENT_W_CITY", "type": "int"},
        {"name": "WEEKDAY_APPR_PROCESS_START", "type": "string"},
        {"name": "HOUR_APPR_PROCESS_START", "type": "int"},
        {"name": "REG_REGION_NOT_LIVE_REGION", "type": "int"},
        {"name": "REG_REGION_NOT_WORK_REGION", "type": "int"},
        {"name": "LIVE_REGION_NOT_WORK_REGION", "type": "int"},
        {"name": "REG_CITY_NOT_LIVE_CITY", "type": "int"},
        {"name": "REG_CITY_NOT_WORK_CITY", "type": "int"},
        {"name": "LIVE_CITY_NOT_WORK_CITY", "type": "int"},
        {"name": "ORGANIZATION_TYPE", "type": "string"},
        {
            "name": "EXT_SOURCE_1",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "EXT_SOURCE_2",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "EXT_SOURCE_3",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "APARTMENTS_AVG",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "BASEMENTAREA_AVG",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "YEARS_BEGINEXPLUATATION_AVG",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "YEARS_BUILD_AVG",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "COMMONAREA_AVG",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "ELEVATORS_AVG",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "ENTRANCES_AVG",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "FLOORSMAX_AVG",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "FLOORSMIN_AVG",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "LANDAREA_AVG",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "LIVINGAPARTMENTS_AVG",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "LIVINGAREA_AVG",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "NONLIVINGAPARTMENTS_AVG",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "NONLIVINGAREA_AVG",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "APARTMENTS_MODE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "BASEMENTAREA_MODE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "YEARS_BEGINEXPLUATATION_MODE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "YEARS_BUILD_MODE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "COMMONAREA_MODE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "ELEVATORS_MODE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "ENTRANCES_MODE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "FLOORSMAX_MODE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "FLOORSMIN_MODE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "LANDAREA_MODE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "LIVINGAPARTMENTS_MODE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "LIVINGAREA_MODE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "NONLIVINGAPARTMENTS_MODE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "NONLIVINGAREA_MODE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "APARTMENTS_MEDI",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "BASEMENTAREA_MEDI",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "YEARS_BEGINEXPLUATATION_MEDI",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "YEARS_BUILD_MEDI",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "COMMONAREA_MEDI",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "ELEVATORS_MEDI",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "ENTRANCES_MEDI",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "FLOORSMAX_MEDI",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "FLOORSMIN_MEDI",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "LANDAREA_MEDI",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "LIVINGAPARTMENTS_MEDI",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "LIVINGAREA_MEDI",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "NONLIVINGAPARTMENTS_MEDI",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "NONLIVINGAREA_MEDI",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "FONDKAPREMONT_MODE",
            "type": ["null", "string"],
            "default": None,
        },  # nullable
        {
            "name": "HOUSETYPE_MODE",
            "type": ["null", "string"],
            "default": None,
        },  # nullable
        {
            "name": "TOTALAREA_MODE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "WALLSMATERIAL_MODE",
            "type": ["null", "string"],
            "default": None,
        },  # nullable
        {
            "name": "EMERGENCYSTATE_MODE",
            "type": ["null", "string"],
            "default": None,
        },  # nullable
        {
            "name": "OBS_30_CNT_SOCIAL_CIRCLE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "DEF_30_CNT_SOCIAL_CIRCLE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "OBS_60_CNT_SOCIAL_CIRCLE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "DEF_60_CNT_SOCIAL_CIRCLE",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {"name": "DAYS_LAST_PHONE_CHANGE", "type": "double"},
        {"name": "FLAG_DOCUMENT_2", "type": "int"},
        {"name": "FLAG_DOCUMENT_3", "type": "int"},
        {"name": "FLAG_DOCUMENT_4", "type": "int"},
        {"name": "FLAG_DOCUMENT_5", "type": "int"},
        {"name": "FLAG_DOCUMENT_6", "type": "int"},
        {"name": "FLAG_DOCUMENT_7", "type": "int"},
        {"name": "FLAG_DOCUMENT_8", "type": "int"},
        {"name": "FLAG_DOCUMENT_9", "type": "int"},
        {"name": "FLAG_DOCUMENT_10", "type": "int"},
        {"name": "FLAG_DOCUMENT_11", "type": "int"},
        {"name": "FLAG_DOCUMENT_12", "type": "int"},
        {"name": "FLAG_DOCUMENT_13", "type": "int"},
        {"name": "FLAG_DOCUMENT_14", "type": "int"},
        {"name": "FLAG_DOCUMENT_15", "type": "int"},
        {"name": "FLAG_DOCUMENT_16", "type": "int"},
        {"name": "FLAG_DOCUMENT_17", "type": "int"},
        {"name": "FLAG_DOCUMENT_18", "type": "int"},
        {"name": "FLAG_DOCUMENT_19", "type": "int"},
        {"name": "FLAG_DOCUMENT_20", "type": "int"},
        {"name": "FLAG_DOCUMENT_21", "type": "int"},
        {
            "name": "AMT_REQ_CREDIT_BUREAU_HOUR",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "AMT_REQ_CREDIT_BUREAU_DAY",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "AMT_REQ_CREDIT_BUREAU_WEEK",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "AMT_REQ_CREDIT_BUREAU_MON",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "AMT_REQ_CREDIT_BUREAU_QRT",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
        {
            "name": "AMT_REQ_CREDIT_BUREAU_YEAR",
            "type": ["null", "double"],
            "default": None,
        },  # nullable
    ],
}

pos_cash_balance_schema = {
    "type": "record",
    "namespace": "com.example",
    "name": "POSCashBalance",
    "fields": [
        {"name": "SK_ID_PREV", "type": "long"},
        {"name": "SK_ID_CURR", "type": "long"},
        {"name": "MONTHS_BALANCE", "type": "int"},
        {
            "name": "CNT_INSTALMENT",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "CNT_INSTALMENT_FUTURE",
            "type": ["null", "double"],
            "default": None,
        },
        {"name": "NAME_CONTRACT_STATUS", "type": "string"},
        {"name": "SK_DPD", "type": "int"},
        {"name": "SK_DPD_DEF", "type": "int"},
    ],
}

credit_card_balance_schema = {
    "type": "record",
    "namespace": "com.example",
    "name": "CreditCardBalance",
    "fields": [
        {"name": "SK_ID_PREV", "type": "long"},
        {"name": "SK_ID_CURR", "type": "long"},
        {"name": "MONTHS_BALANCE", "type": "int"},
        {"name": "AMT_BALANCE", "type": "double"},
        {"name": "AMT_CREDIT_LIMIT_ACTUAL", "type": "long"},
        {
            "name": "AMT_DRAWINGS_ATM_CURRENT",
            "type": ["null", "double"],
            "default": None,
        },
        {"name": "AMT_DRAWINGS_CURRENT", "type": "double"},
        {
            "name": "AMT_DRAWINGS_OTHER_CURRENT",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "AMT_DRAWINGS_POS_CURRENT",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "AMT_INST_MIN_REGULARITY",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "AMT_PAYMENT_CURRENT",
            "type": ["null", "double"],
            "default": None,
        },
        {"name": "AMT_PAYMENT_TOTAL_CURRENT", "type": "double"},
        {"name": "AMT_RECEIVABLE_PRINCIPAL", "type": "double"},
        {"name": "AMT_RECIVABLE", "type": "double"},
        {"name": "AMT_TOTAL_RECEIVABLE", "type": "double"},
        {
            "name": "CNT_DRAWINGS_ATM_CURRENT",
            "type": ["null", "double"],
            "default": None,
        },
        {"name": "CNT_DRAWINGS_CURRENT", "type": "int"},
        {
            "name": "CNT_DRAWINGS_OTHER_CURRENT",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "CNT_DRAWINGS_POS_CURRENT",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "CNT_INSTALMENT_MATURE_CUM",
            "type": ["null", "double"],
            "default": None,
        },
        {"name": "NAME_CONTRACT_STATUS", "type": "string"},
        {"name": "SK_DPD", "type": "int"},
        {"name": "SK_DPD_DEF", "type": "int"},
    ],
}

installments_payments_schema = {
    "type": "record",
    "namespace": "com.example",
    "name": "InstallmentsPayments",
    "fields": [
        {"name": "SK_ID_PREV", "type": "long"},
        {"name": "SK_ID_CURR", "type": "long"},
        {"name": "NUM_INSTALMENT_VERSION", "type": "double"},
        {"name": "NUM_INSTALMENT_NUMBER", "type": "int"},
        {"name": "DAYS_INSTALMENT", "type": "double"},
        {
            "name": "DAYS_ENTRY_PAYMENT",
            "type": ["null", "double"],
            "default": None,
        },
        {"name": "AMT_INSTALMENT", "type": "double"},
        {
            "name": "AMT_PAYMENT",
            "type": ["null", "double"],
            "default": None,
        },
    ],
}

bureau_schema = {
    "type": "record",
    "namespace": "com.example",
    "name": "Bureau",
    "fields": [
        {"name": "SK_ID_CURR", "type": "long"},
        {"name": "SK_ID_BUREAU", "type": "long"},
        {"name": "CREDIT_ACTIVE", "type": "string"},
        {"name": "CREDIT_CURRENCY", "type": "string"},
        {"name": "DAYS_CREDIT", "type": "int"},
        {"name": "CREDIT_DAY_OVERDUE", "type": "int"},
        {
            "name": "DAYS_CREDIT_ENDDATE",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "DAYS_ENDDATE_FACT",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "AMT_CREDIT_MAX_OVERDUE",
            "type": ["null", "double"],
            "default": None,
        },
        {"name": "CNT_CREDIT_PROLONG", "type": "int"},
        {
            "name": "AMT_CREDIT_SUM",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "AMT_CREDIT_SUM_DEBT",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "AMT_CREDIT_SUM_LIMIT",
            "type": ["null", "double"],
            "default": None,
        },
        {"name": "AMT_CREDIT_SUM_OVERDUE", "type": "double"},
        {"name": "CREDIT_TYPE", "type": "string"},
        {"name": "DAYS_CREDIT_UPDATE", "type": "int"},
        {
            "name": "AMT_ANNUITY",
            "type": ["null", "double"],
            "default": None,
        },
    ],
}

previous_application_schema = {
    "type": "record",
    "namespace": "com.example",
    "name": "PreviousApplication",
    "fields": [
        {"name": "SK_ID_PREV", "type": "long"},
        {"name": "SK_ID_CURR", "type": "long"},
        {"name": "NAME_CONTRACT_TYPE", "type": "string"},
        {
            "name": "AMT_ANNUITY",
            "type": ["null", "double"],
            "default": None,
        },
        {"name": "AMT_APPLICATION", "type": "double"},
        {"name": "AMT_CREDIT", "type": ["null", "double"], "default": None},
        {
            "name": "AMT_DOWN_PAYMENT",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "AMT_GOODS_PRICE",
            "type": ["null", "double"],
            "default": None,
        },
        {"name": "WEEKDAY_APPR_PROCESS_START", "type": "string"},
        {"name": "HOUR_APPR_PROCESS_START", "type": "int"},
        {"name": "FLAG_LAST_APPL_PER_CONTRACT", "type": "string"},
        {"name": "NFLAG_LAST_APPL_IN_DAY", "type": "int"},
        {
            "name": "RATE_DOWN_PAYMENT",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "RATE_INTEREST_PRIMARY",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "RATE_INTEREST_PRIVILEGED",
            "type": ["null", "double"],
            "default": None,
        },
        {"name": "NAME_CASH_LOAN_PURPOSE", "type": "string"},
        {"name": "NAME_CONTRACT_STATUS", "type": "string"},
        {"name": "DAYS_DECISION", "type": "int"},
        {"name": "NAME_PAYMENT_TYPE", "type": "string"},
        {"name": "CODE_REJECT_REASON", "type": "string"},
        {
            "name": "NAME_TYPE_SUITE",
            "type": ["null", "string"],
            "default": None,
        },
        {"name": "NAME_CLIENT_TYPE", "type": "string"},
        {"name": "NAME_GOODS_CATEGORY", "type": "string"},
        {"name": "NAME_PORTFOLIO", "type": "string"},
        {"name": "NAME_PRODUCT_TYPE", "type": "string"},
        {"name": "CHANNEL_TYPE", "type": "string"},
        {"name": "SELLERPLACE_AREA", "type": "int"},
        {"name": "NAME_SELLER_INDUSTRY", "type": "string"},
        {
            "name": "CNT_PAYMENT",
            "type": ["null", "double"],
            "default": None,
        },
        {"name": "NAME_YIELD_GROUP", "type": "string"},
        {
            "name": "PRODUCT_COMBINATION",
            "type": ["null", "string"],
            "default": None,
        },
        {
            "name": "DAYS_FIRST_DRAWING",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "DAYS_FIRST_DUE",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "DAYS_LAST_DUE_1ST_VERSION",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "DAYS_LAST_DUE",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "DAYS_TERMINATION",
            "type": ["null", "double"],
            "default": None,
        },
        {
            "name": "NFLAG_INSURED_ON_APPROVAL",
            "type": ["null", "double"],
            "default": None,
        },
    ],
}

bureau_balance_schema = {
    "type": "record",
    "namespace": "com.example",
    "name": "BureauBalance",
    "fields": [
        {"name": "SK_ID_BUREAU", "type": "long"},
        {"name": "MONTHS_BALANCE", "type": "int"},
        {"name": "STATUS", "type": "string"},
    ],
}
