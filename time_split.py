import pandas as pd

DATA_PATH = "dataset/crime_data.csv"   
STATE_COL = "STATE/UT"
DIST_COL = "DISTRICT"
YEAR_COL = "YEAR"

TRAIN_COLUMNS = [
    'MURDER','ATTEMPT TO MURDER','CULPABLE HOMICIDE NOT AMOUNTING TO MURDER',
    'RAPE','CUSTODIAL RAPE','OTHER RAPE','KIDNAPPING & ABDUCTION',
    'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS','KIDNAPPING AND ABDUCTION OF OTHERS',
    'DACOITY','PREPARATION AND ASSEMBLY FOR DACOITY','ROBBERY','BURGLARY',
    'THEFT','AUTO THEFT','OTHER THEFT','RIOTS','CRIMINAL BREACH OF TRUST',
    'CHEATING','COUNTERFIETING','ARSON','HURT/GREVIOUS HURT','DOWRY DEATHS',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY','INSULT TO MODESTY OF WOMEN',
    'CRUELTY BY HUSBAND OR HIS RELATIVES','IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES',
    'CAUSING DEATH BY NEGLIGENCE','OTHER IPC CRIMES'
]

def load_df(path=DATA_PATH):
    df = pd.read_csv(path)
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df = df.sort_values([STATE_COL, DIST_COL, YEAR_COL]).reset_index(drop=True)
    return df

def last_year_holdout_splits(df, min_train_years=2, drop_insufficient=True):
    results = []
    grouped = df.groupby([STATE_COL, DIST_COL])
    for (s, d), g in grouped:
        g = g.sort_values(YEAR_COL)
        if len(g) < min_train_years + 1:
            if drop_insufficient:
                continue
        train_df = g.iloc[:-1]
        test_df = g.iloc[-1:]
        results.append({
            'state': s,
            'district': d,
            'train': train_df,
            'test': test_df,
            'train_years': train_df[YEAR_COL].tolist(),
            'test_year': int(test_df[YEAR_COL].iloc[0])
        })
    return results

def fixed_year_split(df, test_year, min_train_years=2, drop_insufficient=True):
    results = []
    grouped = df.groupby([STATE_COL, DIST_COL])
    for (s, d), g in grouped:
        g = g.sort_values(YEAR_COL)
        train_df = g[g[YEAR_COL] < test_year]
        test_df = g[g[YEAR_COL] == test_year]
        if test_df.empty:
            continue
        if len(train_df) < min_train_years and drop_insufficient:
            continue
        results.append({
            'state': s,
            'district': d,
            'train': train_df,
            'test': test_df,
            'train_years': train_df[YEAR_COL].tolist(),
            'test_year': test_year
        })
    return results

if __name__ == "__main__":
    df = load_df()

    splits_last = last_year_holdout_splits(df)
    print(f"Generated {len(splits_last)} last-year splits")
    
    splits_2012 = fixed_year_split(df, test_year=2012)
    print(f"Generated {len(splits_2012)} splits for test_year=2012")

    meta_last = []
    for s in splits_last:
        meta_last.append({
            'STATE': s['state'],
            'DISTRICT': s['district'],
            'TRAIN_YEARS': s['train_years'],
            'TEST_YEAR': s['test_year']
        })
    pd.DataFrame(meta_last).to_csv("splits_last_year_meta.csv", index=False)

    meta_2012 = []
    for s in splits_2012:
        meta_2012.append({
            'STATE': s['state'],
            'DISTRICT': s['district'],
            'TRAIN_YEARS': s['train_years'],
            'TEST_YEAR': s['test_year']
        })
    pd.DataFrame(meta_2012).to_csv("splits_2012_meta.csv", index=False)

    print("Meta files saved successfully.")
