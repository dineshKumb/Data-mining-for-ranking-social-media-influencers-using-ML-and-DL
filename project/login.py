import streamlit as st
from PIL import Image
#DB Management
# from googleapiclient.discovery import build
import pandas as pd
import re
import time
import requests

api_key='AIzaSyDjF6w3QnVjpYhMjA6oIy3pf-12BKbMRgI'

import sqlite3
conn=sqlite3.connect('data.db')
c=conn.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT, email TEXT,product_name TEXT, info TEXT, launch_date TEXT)')

def add_userdata(username, password, email):
    c.execute('INSERT INTO userstable(username, password, email) VALUES(?,?,?)',(username,password, email))
    conn.commit()

def add_productdata(product_name, info, launch_date, email):
    c.execute('UPDATE userstable SET product_name=?, info=?, launch_date=? WHERE email=?',(product_name,info,launch_date,email))
    conn.commit()

def login_user(email,password):
	c.execute('SELECT * FROM userstable WHERE email =? AND password = ?',(email,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

def convertTobinary(photo):
    with open(photo, 'rb') as f:
        binarydata=f.read()
    return binarydata

# def tabel_list():
#     c.execute('CREATE TABLE IF NOT EXISTS tabel_list(company_name TEXT)')

# def add_tabel_name(company_name_):
#     c.execute('INSERT INTO tabel_list(company_name) VALUES(?)',(company_name_))
#     conn.commit()

# c.execute('DROP TABLE userstable')
# def create_product_info_table():
#     c.execute('CREATE TABLE IF NOT EXISTS Productinfo(product_name TEXT, info TEXT, launch_date TEXT, product_image BLOB)')
#     conn.commit()
# c.execute('DROP TABLE tabel_list')
# c.execute('DROP TABLE channelid_list')
#st.set_page_config(page_title="Login Page", page_icon=":tada:", layout="wide")
def channelid_list():
    c.execute('CREATE TABLE IF NOT EXISTS channelid_list(chennel_id TEXT, company_name TEXT)')

def add_channel_id(chennel_id, company_name):
    c.execute('INSERT INTO channelid_list(chennel_id, company_name) VALUES(?,?)',(chennel_id, company_name))
    conn.commit()

def main():
    st.subheader("Welcome...!")
    if st.checkbox('Influencer'):
        c.execute('SELECT product_name,info,launch_date,username FROM userstable')
        show=c.fetchall()[0]
        st.subheader("Product name:- {} ".format(show[0]))
        st.subheader("Info:- {} ".format(show[1]))
        st.subheader("Product launch date:- {} ".format(show[2]))   
        st.subheader("Comapny name:- {} ".format(show[3])) 

        if st.checkbox('Bid'):
            st.write('directed to the chatbox')
            chennel_id=st.text_input("provide your channel ID")
            company_name=st.text_input("write the name of the company who posted above information")
            if company_name:
                channelid_list()
                add_channel_id(chennel_id,company_name)
    menu=["Home","Login","Sign up Here"]
    choice=st.sidebar.selectbox("Menu",menu)

    if choice=="Home":
        st.sidebar.title("Home")
        # c.execute('INSERT INTO userstable(username, password, email) VALUES(?,?,?)',(username,password, email))
        # conn.commit()
        
        # c.execute('SELECT * FROM channelid_list')
        # data=c.fetchall()
        # st.subheader(data)

    elif choice=="Login":
        st.sidebar.title("Please login here")


        email=st.sidebar.text_input("email")
        password=st.sidebar.text_input("Password",type="password")

        

        if st.sidebar.checkbox("Login"):
            
            create_usertable()
            result = login_user(email,password)
           
            if result:
                st.sidebar.success("Logged in as {}".format(email))
                task=st.selectbox("Task",["Profile","Find influencer","Influencer Analysis"])

                if task=="Profile":
                    #direct to the profile.
                    c.execute('SELECT username FROM userstable WHERE email=? AND password=?',(email,password))
                    username1=c.fetchall()[0][0]
                    myvar={'username':username1}
                    exec(open('profile.py').read(),myvar)
                                    
                
                elif task=="Find influencer":
                    #direct to the posting page.
                    #exec(open('find_influencer.py').read())
                    #c.execute('DROP TABLE Productinfo')
                    st.title("Find influencer")
                    st.subheader('To find an influencer you should first create a post')
                    company_name=st.text_input('Enter your company name')
                    product_name=st.text_input('write your product name')
                    launch_date=st.text_input('launch date')
                    info=st.text_input('Write your product details here')
                    if st.checkbox('Save'):
                        add_productdata(product_name, info, launch_date,email)
                        st.write(info)

                    # if st.checkbox("go"):
                    #     exec(open('feed.py').read())
                    

                elif task=="Influencer Analysis":
                    st.subheader("Influencer Analysis")
                    youtube=build('youtube','v3', developerKey=api_key)
                    def get_channel_stats(youtube, channel_ids): 
                        request=youtube.channels().list(
                            part='snippet,contentDetails,statistics',
                            id=','.join(channel_ids))
                        response= request.execute()
                        all_data=[]
                        for i in range(len(response['items'])):
                            data = dict(Subscribers=response['items'][i]['statistics']['subscriberCount'],
                                    Views=response['items'][i]['statistics']['viewCount'],
                                    Total_videos=response['items'][i]['statistics']['videoCount'],
                                    channel_id=response['items'][i]['contentDetails']['relatedPlaylists']['uploads'])
                            all_data.append(data)  
                            return pd.DataFrame(all_data)
                    channel_stats=[]

                    channelid_list()
                    
                    st.write('Click on this box to see ranking of the bidders')
                    if st.checkbox('Check Rank'):
                        company_name=st.text_input('Write yout company name')
                        if company_name:
                            c.execute('SELECT chennel_id FROM channelid_list WHERE company_name=?',([company_name]))
                            channel_ids=c.fetchall()
                            st.subheader('These are the channel ids of the bidders')
                            for i in channel_ids:
                                st.write(i[0])
                            global rank
                            st.subheader('This is the rank of the bidders based on the total views and subscribers')
                            for i in channel_ids:
                                channel_satistics=get_channel_stats(youtube, i)
                                channel_stats.append(channel_satistics)
                                ranklist=pd.concat(channel_stats)
                                ranklist['Subscribers']=pd.to_numeric(ranklist['Subscribers'])
                                ranklist['Views']=pd.to_numeric(ranklist['Views'])
                                ranklist['Total_videos']=pd.to_numeric(ranklist['Total_videos'])
                                rank=ranklist.sort_values(['Views','Subscribers'], ascending=False)
                            rank[:10]
                    
                    #direct to the influencer analysis page.
                    global channelID
                    a=st.text_input('Provide a Channel ID')
                    if st.checkbox('Analyze'):
                        channelID={'channelID':a}
                        exec(open('API.py').read(),channelID)

            
            else:
                st.warning("Inccorect Username/Password")
        
    elif choice=="Sign up Here":
        st.sidebar.title("Create account here")
        
        username=st.sidebar.text_input("User Name")
        email=st.sidebar.text_input("Enter Email")
        password=st.sidebar.text_input("Password",type="password")
        confirmpassword=st.sidebar.text_input("ConfirmPassword",type="password")

        if st.sidebar.button("Create Account"):
            create_usertable()
            add_userdata(username,password,email)
            st.sidebar.success("Account created as {}".format(username))

    

if __name__=="__main__":
    main()
