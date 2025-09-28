import numpy as np
import pandas as pd

from src.features.engineering import compute_wa_aggregates


def test_wa_aggregates_invariant_to_input_order():
    # Construct a small panel with deliberate unsorted ordering
    sa2 = ['A','A','B','B','A','B']
    month = pd.to_datetime(['2024-01-01','2024-03-01','2024-01-01','2024-02-01','2024-02-01','2024-03-01'])
    stock = [10, 12, 8, 9, 11, 10]
    rent_mom_1m = [np.nan, 0.1, np.nan, 0.05, 0.2, 0.07]
    count_lodgements = [1,2,1,1,1,2]
    count_disposals = [0,1,0,1,1,1]

    order = [2,0,5,1,3,4]  # jumbled order
    df_unsorted = pd.DataFrame({
        'sa2_code': [sa2[i] for i in order],
        'month': [month[i] for i in order],
        'stock_bonds': [stock[i] for i in order],
        'rent_mom_1m': [rent_mom_1m[i] for i in order],
        'count_lodgements': [count_lodgements[i] for i in order],
        'count_disposals': [count_disposals[i] for i in order],
    })

    df_sorted = df_unsorted.sort_values(['sa2_code','month']).reset_index(drop=True)

    out_unsorted = compute_wa_aggregates(df_unsorted)
    out_sorted = compute_wa_aggregates(df_sorted)

    # Exactly equal, including order by month
    assert out_unsorted.equals(out_sorted)
