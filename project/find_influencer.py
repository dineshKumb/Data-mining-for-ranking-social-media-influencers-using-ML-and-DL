import streamlit as st
from PIL import Image
import login
import numpy as np
import sqlite3
conn=sqlite3.connect('data.db')
c=conn.cursor()

# def add_productdata(product_name, info, launch_date, img_array):
#     c.execute('INSERT INTO userstable(product_name, info, launch_date, product_image) VALUES(?,?,?,?)',(product_name, info, launch_date, img_array))
#     conn.commit()

def load_image(product_image):
    img= Image.open(product_image)
    return img


#c.execute('DROP TABLE Productinfo')

st.title("Find influencer")
st.subheader('To find an influencer you should first create a post')
product_name=st.text_input('write your product name')
launch_date=st.text_input('launch date')
info=st.text_input('Write your product details here')
st.write(info)

# product_image=st.file_uploader('Upload Images', type=['png','jpg','jpeg'])

# if product_image is not None:
#     image = Image.open(product_image)
#     st.image(load_image(product_image),width=250)
#     img_array=product_image.read()
#     if st.checkbox("upload"):
#         login.add_productdata(product_name, info, launch_date, img_array)


   


