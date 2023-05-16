import pandas as pd

PD_SELECTED_FEATURES = [
    "id", "purpose", "initial_list_status", "emp_length",
    "pub_rec", "funded_amnt", "grade", "addr_state", "term",
    "int_rate", "default"
]

LGD_SELECTED_FEATURES = [
    "id", "purpose", "initial_list_status", "emp_length",
    "pub_rec", "grade", "addr_state", "term",
    "int_rate", "LGD"
]

EAD_SELECTED_FEATURES = [
    "id", "purpose", "initial_list_status", "emp_length",
    "pub_rec", "grade", "addr_state", "term",
    "int_rate", "EAD"
]

def PD_feature_trasform(df):
    temp_df = df.copy()

    temp_df = temp_df[PD_SELECTED_FEATURES]

    encoded_temp_df = pd.get_dummies(temp_df['grade'], prefix='grade')
    temp_df = temp_df.drop('grade', axis=1)
    temp_df = pd.concat([temp_df, encoded_temp_df], axis=1)

    ADDR_STATE_MAP = {
        "ID": ["ID"],
        "NE_ME": ["NE","ME"],
        "ND": ["ND"],
        "MS_AR_OK": ["MS", "OK", "AR"],
        "SD_LA_AL_AK_IN": ["SD", "LA", "AL", "AK", "IN"],
        "WV_MO_MD_OH": ["WV", "MO", "MD", "OH"],
        "NY_TN_KY_NJ_SC_NC_CT": ["NY", "TN", "KY", "NJ", "SC", "NC", "CT"],
        "IL": ["IL"],
        "PA_NM": ["PA", "NM"],
        "TX_VT": ["TX", "VT"],
        "FL_NV_GA": ["FL", "NV", "GA"],
        "VA_MN": ["MN", "VA"],
        "KS_DE_MA_MI_MT": ["KS", "DE", "MA", "MI", "MT"],
        "AZ_HI_WY_WI": ["AZ", "HI", "WY", "WI"],
        "RI_CA_NH": ["RI", "CA", "NH"],
        "WA_CO_OR_UT": ["WA", "CO", "OR", "UT"],
        "DC": ["DC"]
    }

    reverse_map = {state: group for group, states in ADDR_STATE_MAP.items() for state in states}

    temp_df['addr_state'] = temp_df['addr_state'].map(reverse_map)

    encoded_temp_df = pd.get_dummies(temp_df['addr_state'], prefix='addr_state')
    temp_df = temp_df.drop('addr_state', axis=1)
    temp_df = pd.concat([temp_df, encoded_temp_df], axis=1)

    PURPOSES_MAP = {
        "cred_card": ["credit_card"],
        "vacation": ["vacation"],
        "car": ["car"],
        "home_improv_debt_consol": ["home_improvement", "debt_consolidation"],
        "moving": ["moving"],
        "renewable_energy": ["renewable_energy"],
        "major_purchase_medical": ["major_purchase", "medical"],
        "other": ["other"],
        "wedding": ["wedding"],
        "small_business": ["small_business"],
        "house": ["house"]
    }

    reverse_map = {purpose: group for group, purposes in PURPOSES_MAP.items() for purpose in purposes}
    temp_df['purpose'] = temp_df['purpose'].map(reverse_map)

    encoded_temp_df = pd.get_dummies(temp_df['purpose'], prefix='purpose')
    temp_df = temp_df.drop('purpose', axis=1)
    temp_df = pd.concat([temp_df, encoded_temp_df], axis=1)

    temp_df['term'] = temp_df['term'].str.strip().str.replace(' months', '').astype(int)
    encoded_temp_df = pd.get_dummies(temp_df['term'], prefix='term')
    temp_df = temp_df.drop('term', axis=1)
    temp_df = pd.concat([temp_df, encoded_temp_df], axis=1)

    encoded_temp_df = pd.get_dummies(temp_df['initial_list_status'], prefix='initial_list_status')
    temp_df = temp_df.drop('initial_list_status', axis=1)
    temp_df = pd.concat([temp_df, encoded_temp_df], axis=1)

    mapping = {
        '0': 0,
        '< 1 year': 1,
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10
    }

    # Apply the mapping to convert the values
    temp_df['emp_length'] = temp_df['emp_length'].map(mapping)

    EMP_LENGTH_MAP = {
        "0": [0],
        "1_2_3_4_5_6": [1, 2, 3, 4, 5, 6],
        "7": [7],
        "8": [8],
        "9": [9],
        "10": [10],
    }
    reverse_map = {emp_length: group for group, emp_lengths in EMP_LENGTH_MAP.items() for emp_length in emp_lengths}
    temp_df['emp_length'] = temp_df['emp_length'].map(reverse_map)

    encoded_temp_df = pd.get_dummies(temp_df['emp_length'], prefix='emp_length')
    temp_df = temp_df.drop('emp_length', axis=1)
    temp_df = pd.concat([temp_df, encoded_temp_df], axis=1)



    bins = [0, 1, 16]

    labels = ['0-1', '1-16']

    temp_df['pub_rec_category'] = pd.cut(temp_df['pub_rec'], bins=bins, labels=labels)

    encoded_temp_df = pd.get_dummies(temp_df['pub_rec_category'], prefix='pub_rec_category')
    temp_df = temp_df.drop('pub_rec_category', axis=1)
    temp_df = temp_df.drop('pub_rec', axis=1)
    temp_df = pd.concat([temp_df, encoded_temp_df], axis=1)

    bins = [900, 8800, 16600, 24400, 32200, 40000]

    labels = ['900-8800', '8800-16600', '16600-24400', '32200-32200', '32200-40000']

    temp_df['funded_amnt_category'] = pd.cut(temp_df['funded_amnt'], bins=bins, labels=labels)

    encoded_temp_df = pd.get_dummies(temp_df['funded_amnt_category'], prefix='funded_amnt_category')
    temp_df = temp_df.drop('funded_amnt_category', axis=1)
    temp_df = temp_df.drop('funded_amnt', axis=1)
    temp_df = pd.concat([temp_df, encoded_temp_df], axis=1)

    bins = [0, 9.6, 13.9, 18.2, 22.4, 26.7, 33]

    labels = ['0-9.6%', '9.6-13.9%', '13.9-18.2%', '18.2-22.4%', '22.4-26.7%', '25-33%']

    temp_df['int_rate_category'] = pd.cut(temp_df['int_rate'], bins=bins, labels=labels)

    encoded_temp_df = pd.get_dummies(temp_df['int_rate_category'], prefix='int_rate_category')
    temp_df = temp_df.drop('int_rate_category', axis=1)
    temp_df = temp_df.drop('int_rate', axis=1)
    temp_df = pd.concat([temp_df, encoded_temp_df], axis=1)

    return temp_df

def LGD_feature_trasform(df):
    temp_df = df.copy()
    DEFAULT_CATEGORIES = [
        "Charged Off",
        "Does not meet the credit policy. Status:Charged Off"
    ]

    temp_df = temp_df[temp_df["loan_status"].isin(DEFAULT_CATEGORIES)]

    TARGET_VARIABLE = 'LGD'
    temp_df[TARGET_VARIABLE] = (temp_df["recoveries"] / temp_df["funded_amnt"])
    temp_df[TARGET_VARIABLE].describe()

    temp_df = temp_df[LGD_SELECTED_FEATURES]

    encoded_df = pd.get_dummies(temp_df['grade'], prefix='grade')
    temp_df = temp_df.drop('grade', axis=1)
    temp_df = pd.concat([temp_df, encoded_df], axis=1)

    ADDR_STATE_MAP = {
        "ID": ["ID"],
        "NE_ME": ["NE","ME"],
        "ND": ["ND"],
        "MS_AR_OK": ["MS", "OK", "AR"],
        "SD_LA_AL_AK_IN": ["SD", "LA", "AL", "AK", "IN"],
        "WV_MO_MD_OH": ["WV", "MO", "MD", "OH"],
        "NY_TN_KY_NJ_SC_NC_CT": ["NY", "TN", "KY", "NJ", "SC", "NC", "CT"],
        "IL": ["IL"],
        "PA_NM": ["PA", "NM"],
        "TX_VT": ["TX", "VT"],
        "FL_NV_GA": ["FL", "NV", "GA"],
        "VA_MN": ["MN", "VA"],
        "KS_DE_MA_MI_MT": ["KS", "DE", "MA", "MI", "MT"],
        "AZ_HI_WY_WI": ["AZ", "HI", "WY", "WI"],
        "RI_CA_NH": ["RI", "CA", "NH"],
        "WA_CO_OR_UT": ["WA", "CO", "OR", "UT"],
        "DC": ["DC"]
    }

    reverse_map = {state: group for group, states in ADDR_STATE_MAP.items() for state in states}

    temp_df["addr_state"] = temp_df["addr_state"].map(reverse_map)

    encoded_df = pd.get_dummies(temp_df["addr_state"], prefix="addr_state")
    temp_df = temp_df.drop("addr_state", axis=1)
    temp_df = pd.concat([temp_df, encoded_df], axis=1)

    PURPOSES_MAP = {
        "cred_card": ["credit_card"],
        "vacation": ["vacation"],
        "car": ["car"],
        "home_improv_debt_consol": ["home_improvement", "debt_consolidation"],
        "moving": ["moving"],
        "renewable_energy": ["renewable_energy"],
        "major_purchase_medical": ["major_purchase", "medical"],
        "other": ["other"],
        "wedding": ["wedding"],
        "small_business": ["small_business"],
        "house": ["house"]
    }

    reverse_map = {purpose: group for group, purposes in PURPOSES_MAP.items() for purpose in purposes}
    temp_df["purpose"] = temp_df["purpose"].map(reverse_map)

    encoded_df = pd.get_dummies(temp_df["purpose"], prefix="purpose")
    temp_df = temp_df.drop("purpose", axis=1)
    temp_df = pd.concat([temp_df, encoded_df], axis=1)

    encoded_df = pd.get_dummies(temp_df["initial_list_status"], prefix="initial_list_status")
    temp_df = temp_df.drop("initial_list_status", axis=1)
    temp_df = pd.concat([temp_df, encoded_df], axis=1)


    EMP_LENGTH_MAP = {
        "0": 0,
        "< 1 year": 1,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10
    }
    temp_df["emp_length"] = temp_df["emp_length"].map(EMP_LENGTH_MAP)

    TERM_MAP = {
        " 36 months": 36.0,
        " 60 months": 60.0,
    }
    temp_df["term"] = temp_df["term"].map(TERM_MAP)

    return temp_df

def EAD_feature_trasform(df):
    temp_df = df.copy()
    DEFAULT_CATEGORIES = [
        "Charged Off",
        "Does not meet the credit policy. Status:Charged Off"
    ]

    temp_df = temp_df[temp_df["loan_status"].isin(DEFAULT_CATEGORIES)]

    TARGET_VARIABLE = 'EAD'
    temp_df[TARGET_VARIABLE] = (temp_df["funded_amnt"] - temp_df["total_rec_prncp"])

    temp_df[TARGET_VARIABLE].describe()

    temp_df = temp_df[EAD_SELECTED_FEATURES]

    encoded_df = pd.get_dummies(temp_df['grade'], prefix='grade')
    temp_df = temp_df.drop('grade', axis=1)
    temp_df = pd.concat([temp_df, encoded_df], axis=1)

    ADDR_STATE_MAP = {
        "ID": ["ID"],
        "NE_ME": ["NE","ME"],
        "ND": ["ND"],
        "MS_AR_OK": ["MS", "OK", "AR"],
        "SD_LA_AL_AK_IN": ["SD", "LA", "AL", "AK", "IN"],
        "WV_MO_MD_OH": ["WV", "MO", "MD", "OH"],
        "NY_TN_KY_NJ_SC_NC_CT": ["NY", "TN", "KY", "NJ", "SC", "NC", "CT"],
        "IL": ["IL"],
        "PA_NM": ["PA", "NM"],
        "TX_VT": ["TX", "VT"],
        "FL_NV_GA": ["FL", "NV", "GA"],
        "VA_MN": ["MN", "VA"],
        "KS_DE_MA_MI_MT": ["KS", "DE", "MA", "MI", "MT"],
        "AZ_HI_WY_WI": ["AZ", "HI", "WY", "WI"],
        "RI_CA_NH": ["RI", "CA", "NH"],
        "WA_CO_OR_UT": ["WA", "CO", "OR", "UT"],
        "DC": ["DC"]
    }

    reverse_map = {state: group for group, states in ADDR_STATE_MAP.items() for state in states}

    temp_df["addr_state"] = temp_df["addr_state"].map(reverse_map)

    encoded_df = pd.get_dummies(temp_df["addr_state"], prefix="addr_state")
    temp_df = temp_df.drop("addr_state", axis=1)
    temp_df = pd.concat([temp_df, encoded_df], axis=1)

    PURPOSES_MAP = {
        "cred_card": ["credit_card"],
        "vacation": ["vacation"],
        "car": ["car"],
        "home_improv_debt_consol": ["home_improvement", "debt_consolidation"],
        "moving": ["moving"],
        "renewable_energy": ["renewable_energy"],
        "major_purchase_medical": ["major_purchase", "medical"],
        "other": ["other"],
        "wedding": ["wedding"],
        "small_business": ["small_business"],
        "house": ["house"]
    }

    reverse_map = {purpose: group for group, purposes in PURPOSES_MAP.items() for purpose in purposes}
    temp_df["purpose"] = temp_df["purpose"].map(reverse_map)

    encoded_df = pd.get_dummies(temp_df["purpose"], prefix="purpose")
    temp_df = temp_df.drop("purpose", axis=1)
    temp_df = pd.concat([temp_df, encoded_df], axis=1)

    encoded_df = pd.get_dummies(temp_df["initial_list_status"], prefix="initial_list_status")
    temp_df = temp_df.drop("initial_list_status", axis=1)
    temp_df = pd.concat([temp_df, encoded_df], axis=1)


    EMP_LENGTH_MAP = {
        "0": 0,
        "< 1 year": 1,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10
    }
    temp_df["emp_length"] = temp_df["emp_length"].map(EMP_LENGTH_MAP)

    TERM_MAP = {
        " 36 months": 36.0,
        " 60 months": 60.0,
    }
    temp_df["term"] = temp_df["term"].map(TERM_MAP)
    return temp_df