import streamlit as st
import pandas as pd
from portable import PortableRiskModel

st.set_page_config(page_title='Study Office Risk', layout='wide')
st.title('Study Office — Week 6 risk review')
st.caption('The model ranks students for human review. It does not make an automatic decision.')

@st.cache_resource
def load_model():
    return PortableRiskModel('model')

@st.cache_data
def load_data():
    return pd.read_csv('history_week6.csv'), pd.read_csv('new_week6.csv')

model = load_model()
history, new = load_data()

# 2026 ranking
new_ranked = new.copy()
new_ranked['risk'] = model.predict_risk(new_ranked)
new_ranked = new_ranked.sort_values('risk', ascending=False).reset_index(drop=True)
new_ranked['talk_this_week'] = False
new_ranked.loc[:39, 'talk_this_week'] = True

# 2025 evaluation data
val = history[history['cohort'] == 2025].copy()
val['risk'] = model.predict_risk(val)

st.sidebar.header('Office rule')
rule = st.sidebar.radio('Choose the rule', ['Risk cut-off', 'Number of conversations'])
if rule == 'Risk cut-off':
    threshold = st.sidebar.slider('Risk cut-off', 0.0, 1.0, 0.125, 0.01)
    contacted = val.risk >= threshold
    current = new_ranked[new_ranked.risk >= threshold]
else:
    capacity = st.sidebar.slider('Number of conversations', 1, 100, 40)
    threshold = float(new_ranked.iloc[min(capacity, len(new_ranked))-1].risk)
    contacted = val.risk >= threshold
    current = new_ranked.head(capacity)

# Confusion matrix on 2025
left = val.left.astype(bool)
tp = int((contacted & left).sum())
fp = int((contacted & ~left).sum())
fn = int((~contacted & left).sum())
tn = int((~contacted & ~left).sum())
precision = tp/(tp+fp) if tp+fp else 0
recall = tp/(tp+fn) if tp+fn else 0

c1,c2,c3,c4 = st.columns(4)
c1.metric('Reached in time', tp)
c2.metric('Worried for nothing', fp)
c3.metric('Missed', fn)
c4.metric('Precision', f'{precision:.1%}')
st.metric('Recall', f'{recall:.1%}')

st.subheader("This week's ranked list")
st.write(f'{len(current)} students selected by the current rule. The operational capacity is 40 conversations.')
st.dataframe(current[['student_id','programme','international','fees_owed','submitted_share','missed_last3','quiz_mean','risk','talk_this_week']].head(100), use_container_width=True)

st.subheader('2025 mistakes in plain words')
st.write(f'{tp} students reached in time, {fp} worried for nothing, and {fn} missed.')

st.subheader('International vs domestic')
rows=[]
for label,value in [('Domestic',0),('International',1)]:
    g=val[val.international==value]
    gc=contacted.loc[g.index]
    gl=g.left.astype(bool)
    gtp=int((gc&gl).sum()); gfp=int((gc&~gl).sum()); gfn=int((~gc&gl).sum()); gtn=int((~gc&~gl).sum())
    rows.append({'group':label,'students':len(g),'share_left':g.left.mean(),'reached_in_time':gtp,'worried_for_nothing':gfp,'missed':gfn,'precision':gtp/(gtp+gfp) if gtp+gfp else 0,'recall':gtp/(gtp+gfn) if gtp+gfn else 0})
st.dataframe(pd.DataFrame(rows).set_index('group'), use_container_width=True)

st.subheader('One extra view: programme mix')
st.dataframe(current.groupby('programme').size().rename('students_selected').sort_values(ascending=False).to_frame(), use_container_width=True)

st.info('A human adviser should review the list, decide whether to contact each student, explain the support process, and allow students to object. The score is not an automatic decision.')
