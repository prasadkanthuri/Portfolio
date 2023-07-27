import streamlit as st
import pickle
import numpy as np

RF = pickle.load(open("RF.pkl",'rb'))
scalar_x = pickle.load(open("scalar_x.pkl",'rb'))

st.title("Customer Segmentation Analysis")

col1,col2,col3 = st.columns(3)


with col1:
    income = st.number_input("Enter the Income:")
    age = st.number_input("Enter the Age: ")
    total_spent = st.number_input("Enter Total amount spent")
    children = st.number_input("Enter the no.of children:")

with col2:

    living_status = st.selectbox("Select the living status",['With Partner','Without Partner'])
    education = st.selectbox("Select the education status",['Under graduate', 'Graduate', 'Post graduate'])
    family_size = st.number_input("Enter the no.of persons in family:")
    is_parent = st.selectbox("Is customer a parent..? ", ['Yes', 'No'])

with col3:
    customer_duration = st.number_input("Customer for (Days as duration) ")
    teen_home = st.number_input("Enter the no.of teens at home")
    kids_home = st.number_input("Enter the no.of kids at home")


# Encoding
if living_status == "With Partner":
    living_status = 1
else:
    living_status = 0

if education == 'Under graduate':
    education =2
elif education =='Graduate':
    education=0
else:
    education=1

if is_parent =="Yes":
    is_parent=1
else:
    is_parent=0

if st.button("Predict the cluster"):
    input_vector = [income,age,total_spent,children,living_status,education,family_size,is_parent,customer_duration,teen_home,kids_home]

    input_array = np.array(input_vector)

    reshaped_array = input_array.reshape(1,-1)

    scaled_array = scalar_x.transform(reshaped_array)
    result = RF.predict(scaled_array)[0]
    st.header("The given customer belongs to the Cluster {} ".format(result))

    st.subheader("Here are the characteristics of Cluster {}".format(result))
    if result == 1:
        st.text(
            """1. High income and High spent customers
2. Mostly not a Parent.
3. No kids at home.
4. Maximum family size will be 2.
5. Single parent with one teen at home will be a subset of this group."""
        )
    elif result ==0:
        st.text(
        """
        1. Medium income and Low spent.
2. Definitely a Parent with a maximum of 3 children at home.
3. Most of the children might be Teens.
4. Single parents are the subset of this group.
5. Consists of all age groups.
        """)

    elif result == 2:
        st.text("""
        1. Low income and Low spent
2. Maximum family size will be 3.
3. Can have only one or no child. Probability of child being kid is more than a teen
4. Single parent can be a subset of this cluster
        """)

    else:
        st.text("""
        1. Medium salary and Medium spent customers.
2. Mostly a parent with a maximum of 2 children.
3. Family size can be of maximum 4.
4. Probability of Teen at home is more comaring to the kid.
        """)









