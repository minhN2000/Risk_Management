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

EMP_LENGTH_MAP_1 = {
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


EMP_LENGTH_MAP_2 = {
    "0": [0],
    "1_2_3_4_5_6": [1, 2, 3, 4, 5, 6],
    "7": [7],
    "8": [8],
    "9": [9],
    "10": [10],
}

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

TERM_MAP = {
    " 36 months": 36.0,
    " 60 months": 60.0,
}

def encode_and_drop(df, col, prefix):
    """
    This function is used to perform one-hot encoding on a specified column and drop the original column.

    Parameters:
    df (pd.DataFrame): input dataframe
    col (str): column name to be encoded
    prefix (str): prefix for the new columns after encoding

    Returns:
    df (pd.DataFrame): updated dataframe after encoding and dropping the original column
    """
    encoded_df = pd.get_dummies(df[col], prefix=prefix)
    df = df.drop(col, axis=1)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def binning_and_encode(df, col, bins, labels):
    """
    This function is used to perform binning on a specified column and encode the binned data.

    Parameters:
    df (pd.DataFrame): input dataframe
    col (str): column name to be binned
    bins (list): boundaries for binning
    labels (list): labels for the corresponding bins

    Returns:
    df (pd.DataFrame): updated dataframe after binning and encoding
    """

    new_col = col + '_category'
    df[new_col] = pd.cut(df[col], bins=bins, labels=labels)
    encode_and_drop(df, new_col, new_col)
    df = df.drop(col, axis=1)
    return df

def map_and_encode(df, col, MAP, prefix):
    """
    This function is used to map the values in a specified column using a provided dictionary and encode the mapped data.

    Parameters:
    df (pd.DataFrame): input dataframe
    col (str): column name to be mapped and encoded
    MAP (dict): dictionary for mapping
    prefix (str): prefix for the new columns after encoding

    Returns:
    df (pd.DataFrame): updated dataframe after mapping and encoding
    """
    reverse_map = {value: key for key, values in MAP.items() for value in values}
    df[col] = df[col].map(reverse_map)
    df = encode_and_drop(df, col, prefix)
    return df


def PD_feature_trasform(df):
    """
    This function is used to transform the data before doing PD model training

    Parameters:
    df (pd.DataFrame): input dataframe

    Returns:
    df (pd.DataFrame): updated dataframe after tranforming the data on specific features
    """
    temp_df = df.copy()

    temp_df = temp_df[PD_SELECTED_FEATURES]

    temp_df = encode_and_drop(temp_df, col='grade', prefix='grade')

    temp_df = map_and_encode(temp_df, col='addr_state', MAP=ADDR_STATE_MAP, prefix='addr_state')

    temp_df = map_and_encode(temp_df, col='purpose', MAP=PURPOSES_MAP, prefix='purpose')

    temp_df['term'] = temp_df['term'].str.strip().str.replace(' months', '').astype(int)
    temp_df = encode_and_drop(temp_df, col='term', prefix='term')
    
    temp_df = encode_and_drop(temp_df, col='initial_list_status', prefix='initial_list_status')

    
    temp_df['emp_length'] = temp_df['emp_length'].map(EMP_LENGTH_MAP_1)
    temp_df = map_and_encode(temp_df, col='emp_length', MAP=EMP_LENGTH_MAP_2, prefix='emp_length')

    bins = [0, 1, 16]
    labels = ['0-1', '1-16']
    binning_and_encode(temp_df, col='pub_rec', bins=bins, labels=labels)
    
    bins = [900, 8800, 16600, 24400, 32200, 40000]
    labels = ['900-8800', '8800-16600', '16600-24400', '32200-32200', '32200-40000']
    binning_and_encode(temp_df, col='funded_amnt', bins=bins, labels=labels)

    bins = [0, 9.6, 13.9, 18.2, 22.4, 26.7, 33]
    labels = ['0-9.6%', '9.6-13.9%', '13.9-18.2%', '18.2-22.4%', '22.4-26.7%', '25-33%']
    binning_and_encode(temp_df, col='int_rate', bins=bins, labels=labels)

    return temp_df

def LGD_feature_trasform(df):
    """
    This function is used to transform the data before doing LGD model training

    Parameters:
    df (pd.DataFrame): input dataframe

    Returns:
    df (pd.DataFrame): updated dataframe after tranforming the data on specific features
    """
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

    temp_df = encode_and_drop(temp_df, col='grade', prefix='grade')

    temp_df = map_and_encode(temp_df, col='addr_state', MAP=ADDR_STATE_MAP, prefix='addr_state')

    temp_df = map_and_encode(temp_df, col='purpose', MAP=PURPOSES_MAP, prefix='purpose')

    temp_df = encode_and_drop(temp_df, col='initial_list_status', prefix='initial_list_status')

    temp_df["emp_length"] = temp_df["emp_length"].map(EMP_LENGTH_MAP_1)

    temp_df["term"] = temp_df["term"].map(TERM_MAP)

    return temp_df

def EAD_feature_trasform(df):
    """
    This function is used to transform the data before doing EAD model training

    Parameters:
    df (pd.DataFrame): input dataframe

    Returns:
    df (pd.DataFrame): updated dataframe after tranforming the data on specific features
    """
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

    temp_df = encode_and_drop(temp_df, col='grade', prefix='grade')

    temp_df = map_and_encode(temp_df, col='addr_state', MAP=ADDR_STATE_MAP, prefix='addr_state')

    temp_df = map_and_encode(temp_df, col='purpose', MAP=PURPOSES_MAP, prefix='purpose')
    
    temp_df = encode_and_drop(temp_df, col='initial_list_status', prefix='initial_list_status')

    temp_df["emp_length"] = temp_df["emp_length"].map(EMP_LENGTH_MAP_1)

    temp_df["term"] = temp_df["term"].map(TERM_MAP)
    return temp_df