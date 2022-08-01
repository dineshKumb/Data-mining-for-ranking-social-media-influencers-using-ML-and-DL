import streamlit as st
from PIL import Image
#DB Management

import sqlite3
conn=sqlite3.connect('data.db')
c=conn.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT, email TEXT,product_name TEXT, info TEXT, launch_date TEXT, product_image BLOB)')

def add_userdata(username, password, email):
    c.execute('INSERT INTO userstable(username, password, email) VALUES(?,?,?)',(username,password, email))
    conn.commit()

def add_productdata(product_name, info, launch_date, product_image, email):
    c.execute('UPDATE userstable SET product_name=?, info=?, launch_date=?, product_image=? WHERE email=?',(product_name,info,launch_date,product_image,email))
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

#c.execute('DROP TABLE userstable')
# def create_product_info_table():
#     c.execute('CREATE TABLE IF NOT EXISTS Productinfo(product_name TEXT, info TEXT, launch_date TEXT, product_image BLOB)')
#     conn.commit()

#c.execute('DROP TABLE Productinfo')
#st.set_page_config(page_title="Login Page", page_icon=":tada:", layout="wide")
def load_image(product_image):
    img= Image.open(product_image)
    return img



def main():
    st.subheader("Welcome...!")
    menu=["Home","Login","Sign up Here"]
    choice=st.sidebar.selectbox("Menu",menu)

    if choice=="Home":
        st.sidebar.title("Home")
        # c.execute('INSERT INTO userstable(username, password, email) VALUES(?,?,?)',(username,password, email))
        # conn.commit()

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

        
