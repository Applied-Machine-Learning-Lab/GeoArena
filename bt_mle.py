# copyright to chatbot arena
# https://colab.research.google.com/drive/19VPOril2FjCX34lJoo7qn4r6adgKLioY?ref=news.lmarena.ai#scrollTo=C4xnVybEy0OO
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def compute_bt(
    df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None
):
    from sklearn.linear_model import LogisticRegression
    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    # if no tie, create a zero matrix
    if sum(df["winner"].isin(["tie", "tie (bothbad)"])) == 0:
        ptbl_tie = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    else:
        ptbl_tie = pd.pivot_table(
            df[df["winner"].isin(["tie", "tie (bothbad)"])],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        ptbl_tie = ptbl_tie + ptbl_tie.T

    ptbl_b_win = pd.pivot_table(
        df[df["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    # if "mixtral-8x7b-instruct-v0.1" in models.index:
    #     # anchor
    #     elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]
    anchor_model = "openai/gpt-4o"  # Choose based on your data
    if anchor_model in models.index:
        elo_scores += 1000 - elo_scores[models[anchor_model]]
    return pd.Series(elo_scores, index=models.index)

def compute_bt_scores(leaderboard, SCALE=400, BASE=10, INIT_RATING=1000):
    
    battles_list = []
    seen_timestamps = set()  # To avoid duplicates based on timestamp
    
    for model, info in leaderboard.items():
        history = info.get("history", [])
        for entry in history:
            timestamp = entry.get("timestamp")
            if timestamp in seen_timestamps:
                continue  # Skip duplicate
            seen_timestamps.add(timestamp)
            
            opponent = entry.get("opponent")
            result = entry.get("result")
            
            if result == "win":
                winner = "model_a"  # model_a is the current model
            elif result == "loss":
                winner = "model_b"
            elif result == "tie":
                winner = "tie"
            else:
                continue  # Invalid result

            battles_list.append({
                "model_a": model,
                "model_b": opponent,
                "winner": winner,
                "timestamp": timestamp,
                # Add other fields if needed, like language, etc.
            })
    
    battles = pd.DataFrame(battles_list)
    
    bt_scores = compute_bt(battles)
    return {m: float(s) for m, s in bt_scores.items()}

