import streamlit as st
import login

import sqlite3
conn=sqlite3.connect('data.db')
c=conn.cursor()

def channelid_list():
    c.execute('CREATE TABLE IF NOT EXISTS channelid_list(chennel_id TEXT)')

def add_channel_id(chennel_id):
    c.execute('INSERT INTO channelid_list(chennel_id) VALUES(?)',(chennel_id))
    conn.commit()

c.execute('SELECT product_name,info,launch_date FROM userstable')
show=c.fetchall()[0]

st.subheader(show)
if st.checkbox('Bid'):
    st.write('directed to the chatbox')
    chennel_id=st.text_input("provide your channel ID")
    if chennel_id:
        channelid_list()
        add_channel_id([chennel_id])
c.execute('SELECT * FROM channelid_list')
data=c.fetchall()
st.subheader(data)
    