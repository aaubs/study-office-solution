"""Study office, week 6: who to talk to, and what the rule gets wrong. (Solved example for the assignment.)"""
import pandas as pd
import streamlit as st

import portable

URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"
SHOW = ["student_id", "programme", "international", "logins_last3", "missed_last3", "quiz_mean", "fees_owed"]

st.set_page_config(page_title="Study office · week 6", page_icon="🎓", layout="wide")


@st.cache_resource
def load():
    model = portable.Model("model")
    new, hist = pd.read_csv(URL + "new_week6.csv"), pd.read_csv(URL + "history_week6.csv")
    last_year = hist[hist["cohort"] == 2025].copy()            # the model never saw 2025: an honest check
    new["risk"], last_year["risk"] = model.predict_proba(new), model.predict_proba(last_year)
    return model, new, last_year


def boxes(df, k):
    """The four boxes if the office talks to the k highest risks."""
    talk, left = df["risk"].rank(ascending=False, method="first") <= k, df["left"] == 1
    return {"reached": int((talk & left).sum()), "worried": int((talk & ~left).sum()),
            "missed": int((~talk & left).sum()), "fine": int((~talk & ~left).sum())}


model, new, last_year = load()

st.title("🎓 Study office · end of week 6")
st.caption("The model ranks students by their risk of leaving later this semester. It does not decide anything: "
           "an adviser reads the list and offers help. Students can say no.")

with st.sidebar:
    st.header("The rule")
    k = st.slider("Conversations this week", 10, 200, 40, step=5)
    st.header("What the mistakes cost (assumptions)")
    talk_cost = st.number_input("One conversation (DKK)", 0, 5_000, 500, step=100)
    worry_cost = st.number_input("A student worried for nothing (DKK)", 0, 20_000, 2_000, step=500)
    leave_cost = st.number_input("A student who leaves (DKK)", 0, 200_000, 60_000, step=5_000)
    helps = st.slider("Share of students at risk a conversation keeps", 0.0, 1.0, 0.30, 0.05)

tab_list, tab_rule, tab_groups = st.tabs(["📋 This week's list", "⚖️ What the rule gets wrong", "👥 Per group"])

with tab_list:
    top = new.nlargest(k, "risk").reset_index(drop=True)
    st.subheader(f"The {k} students to talk to this week")
    st.dataframe(top[SHOW + ["risk"]].style.format({"risk": "{:.0%}", "quiz_mean": "{:.0f}"}),
                 hide_index=True, width="stretch")

    st.subheader("Why is a student on the list?")
    who = st.selectbox("Student", top["student_id"])
    row = top[top["student_id"] == who]
    why = model.contributions(row).iloc[0].sort_values()
    st.write(f"Risk **{row['risk'].iloc[0]:.0%}**. What pushes it up (red) or down (blue), compared with an average student:")
    st.bar_chart(pd.DataFrame({"up": why.clip(lower=0), "down": why.clip(upper=0)}), color=["#1f5fbf", "#d2542a"],
                 horizontal=True)
    st.info("Suggested message: an invitation to talk, never a warning. Few logins alone is a weak reason, "
            "especially for international students (see *Per group*).")

with tab_rule:
    b = boxes(last_year, k)
    st.subheader(f"If the office had talked to the top {k} in 2025")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Reached in time", b["reached"])
    c2.metric("📞 Worried for nothing", b["worried"])
    c3.metric("🚪 Missed", b["missed"])
    c4.metric("✅ Left alone, and fine", b["fine"])
    left_total = b["reached"] + b["missed"]
    st.write(f"Of the {k} students contacted, **{b['reached'] / k:.0%}** were really at risk (precision). "
             f"Of the {left_total} students who left, the office reached **{b['reached'] / left_total:.0%}** (recall).")

    value = b["reached"] * helps * leave_cost - k * talk_cost - b["worried"] * worry_cost
    st.metric("Net value of the rule (DKK)", f"{value:,.0f}")

    curve = pd.DataFrame([{"conversations": n, **boxes(last_year, n)} for n in range(10, 401, 10)])
    curve["net value (DKK)"] = (curve["reached"] * helps * leave_cost - curve["conversations"] * talk_cost
                                - curve["worried"] * worry_cost)
    best = curve.loc[curve["net value (DKK)"].idxmax(), "conversations"]
    st.line_chart(curve.set_index("conversations")[["reached", "worried", "missed"]])
    st.line_chart(curve.set_index("conversations")["net value (DKK)"])
    st.caption(f"With these assumptions, about {best} conversations would pay off most. The office has capacity "
               f"for {k}. The costs are assumptions: change them in the sidebar.")

with tab_groups:
    st.subheader(f"The same rule (top {k}), international and domestic students, 2025")
    talk = last_year["risk"].rank(ascending=False, method="first") <= k
    rows = []
    for g, sub in last_year.groupby("international"):
        t, left = talk[sub.index], sub["left"] == 1
        rows.append({"group": "international" if g else "domestic", "students": len(sub),
                     "really left": f"{left.mean():.1%}", "average predicted risk": f"{sub['risk'].mean():.1%}",
                     "leavers reached (recall)": f"{(t & left).sum() / max(1, left.sum()):.0%}",
                     "worried for nothing": int((t & ~left).sum())})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.warning("Both groups leave about as often, but the model reaches fewer of the international students who "
               "leave. International students log in less, for reasons that have nothing to do with risk, so a "
               "login is not the same signal for everyone. Check this before using the list.")
