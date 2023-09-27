import streamlit as st
import pickle 
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import sklearn
from datetime import date
#load preprocessor and model
preprocessor=pickle.load(open(r'/home/ramy/deskop/ahmed/Data-Scince-and-analysis-and-AI/Data-science-cousre-Epsilon/Machine-learning/projects/Final-project/Starbucks-classifcation/streamlit/preprocsser','rb'))
model=pickle.load(open(r'/home/ramy/deskop/ahmed/Data-Scince-and-analysis-and-AI/Data-science-cousre-Epsilon/Machine-learning/projects/Final-project/Starbucks-classifcation/streamlit/demo','rb'))
data=pd.read_csv("data")
data["age_group"]=pd.cut(x=data["age"],bins=[18,30,40,50,60,70,80,100],
          labels=["18-30 age","30-40 age","40-50 age","50-60 age","60-70 age","70-80 age","80-above age"])
# streamlit layout

st.set_page_config(page_title="Prediction offers in Starbucks",layout="wide")

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "Pages",
    ("Info", "EDA", "Model and preduction")
)
#####################################################################################################################################

# Using "with" notation
if add_selectbox =="Info":
    st.title("Prediction offers in Starbucks mobile app, By [Ahmed Ramadan](https://www.linkedin.com/in/ahmed-ramadan-18b873230/)")

                    
    st.header("About :")

    st.markdown("Overview: \n This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. Not all users receive the same offer, and that is the challenge to solve with this data set. Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products. Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement. You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer.")
    st.markdown("-----------------------------------")
    data=pd.read_csv("data")

    st.markdown("This is the sample of dataframe after cleaning:")
    sample=st.dataframe(data.sample(15))
    btn=st.button("Display another sample")
    if btn:
        print(sample)


    st.markdown("-----------------------------------")

#####################################################################################################################################

if add_selectbox =="EDA":
    st.subheader("In Exploratory data analysis (EDA) we have 3 type")
    st.markdown("1) Univarate")
    st.markdown("2) Bivarate")
    st.markdown("3) Multivarate")
    sb=st.selectbox("__Select what type to show visualization it__",["Univarate","Bivarate","Multivarate"])
    #########################
    if sb== "Univarate":
        columns=data.columns.drop(["person","value/offer id","age_group"]).to_list()
        uni=st.selectbox("choose column : ",columns)
        ###########
        if uni=="gender":
            df=data.gender.value_counts().reset_index()
            fig=px.bar(data_frame=df,x="index",y="gender",text_auto="0.2s",
                labels={"index":"gender",
                        "gender":"count"})
            fig.update_traces(textfont_size=12,textposition="outside")
            fig.update_layout(title_text="count of gender",title_x=0.5)
            st.plotly_chart(fig)
        ###########
        if uni=="event":
            df=data.event.value_counts().reset_index()
            fig=px.bar(data_frame=df,x="index",y="event",text_auto="0.2s",
                labels={"index":"event",
                        "event":"count"})
            fig.update_traces(textfont_size=12,textposition="outside")
            fig.update_layout(title_text="count of event",title_x=0.5)
            st.plotly_chart(fig)
        ########### 
        if uni=="offer_type":
            df=data.offer_type.value_counts().reset_index()
            fig=px.bar(data_frame=df,x="index",y="offer_type",text_auto="0.2s",
                labels={"index":"offer_type",
                        "offer_type":"count"})
            fig.update_traces(textfont_size=12,textposition="outside")
            fig.update_layout(title_text="count of offer_type",title_x=0.5)
            st.plotly_chart(fig)
        ###########
        #numerical
        numerical=data.select_dtypes(exclude="O").columns.to_list()
        for col in numerical:
            if uni==col:
                plt.title(f"Histgrom of {col}")
                fig, ax = plt.subplots()
                ax=sns.histplot(data[col])
                st.pyplot(fig)
                break
    #########################
    if sb== "Bivarate":

        col1,col2,col3=st.columns(3)
        with col1:
            # event and offer_type
            df=data.groupby(["offer_type","event"]).agg({"event":"count"}).rename(columns={"event":"count"}).reset_index()
            fig=px.bar(data_frame=df,x="offer_type",y="count",color="event", barmode='group',text_auto="0.2s")
            fig.update_traces(textfont_size=12,textposition="outside")
            fig.update_layout(title_text="event VS offer_type",title_x=0.5)
            st.plotly_chart(fig)
        with col2:
            # gender and offer_type
            df=data.groupby(["offer_type","gender"]).agg({"gender":"count"}).rename(columns={"gender":"count"}).reset_index()
            fig=px.bar(data_frame=df,x="offer_type",y="count",color="gender", barmode='group',text_auto="0.2s")
            fig.update_traces(textfont_size=12,textposition="outside")
            fig.update_layout(title_text="gender VS offer_type",title_x=0.5)
            st.plotly_chart(fig)
        with col3:
            # age and offer_type
            df=data.groupby(["offer_type","age_group"]).agg({"age_group":"count"}).rename(columns={"age_group":"count"}).reset_index()
            fig=px.bar(data_frame=df,x="age_group",y="count",color="offer_type", barmode='group',text_auto="0.2s")
            fig.update_traces(textfont_size=12,textposition="outside")
            fig.update_layout(title_text="age VS offer_type",title_x=0.5)
            st.plotly_chart(fig)
            st.markdown("- age between 50 and 60 that have most vote in all offers")
        col1,col2=st.columns(2)
        with col1:
            # login_days and offer_type
            fig=px.box(data_frame=data,x="login_days",color="offer_type")
            fig.update_layout(title_text="login_days VS offer_type",title_x=0.5)
            st.plotly_chart(fig)
        with col2:
            # income and offer_type
            fig=px.box(data_frame=data,x="income",color="offer_type",)
            fig.update_layout(title_text="income VS offer_type",title_x=0.5)
            st.plotly_chart(fig)
    #########################
    if sb== "Multivarate":
        # gender , age_group and offer_type
        df=data.groupby(["age_group","gender","offer_type"]).agg({"offer_type":"count"}).rename(columns={"offer_type":"count"}).reset_index()
        fig=px.sunburst(data_frame=df,path=["age_group","gender","offer_type"],values="count")
        fig.update_layout(title_text="[age_group , gender] VS offer_type",title_x=0.5)
        st.plotly_chart(fig)
        st.markdown("- Most vote for age between 50 and 60 male and most vote in offers are bogo and discount")

        # gender , event and offer_type

        df=data.groupby(["event","gender","offer_type"]).agg({"offer_type":"count"}).rename(columns={"offer_type":"count"}).reset_index()
        fig=px.sunburst(data_frame=df,path=["event","gender","offer_type"],values="count")
        fig.update_layout(title_text="[event , gender] VS offer_type",title_x=0.5)
        st.plotly_chart(fig)
        st.markdown("- males that are recievied  offer of bogo those are most vote from last graph")

        #correlation
        fig,ax=plt.subplots()
        ax=sns.heatmap(data.corr(),annot=True,fmt=".1f")
        st.pyplot(fig)
        st.markdown("- __There are realation between :__")
        st.markdown("1) age - income")
        st.markdown("2) rewared - [ difficult , duration]")
        st.markdown("3) difficult - duration")
####################################################################################################################################

if add_selectbox =="Model and preduction":
    st.title("In this part : ")
    st.header("After clean data and understand data , I will create model to predict type and details of offer.")
    st.subheader("Enter inputs to prdict offer :")
    gender=st.selectbox("Gender : ",["M","F","O"])
    age=st.number_input("Age : ")
    became_member_on=st.date_input("date when customer created an app account",value=date(2018,7,26),min_value=date(2013,7,29),max_value=date(2018,7,26))
    income=st.number_input("Income : ")
    event=st.selectbox("Event : ",["offer received","transaction","offer viewed","offer completed"])
    time=st.number_input("time : ")
    df=pd.DataFrame({"gender":gender,
                 "age":age,
                 "became_member_on":became_member_on,
                 "income":income,
                 "event":event,
                 "time":time},index=[0])

    #function preprecessing
    def clean(df):
        df.became_member_on=pd.to_datetime(df.became_member_on)
        df.insert(loc=3,value=df.became_member_on.dt.year,column="became_member_year") # year
        df.insert(loc=4,value=df.became_member_on.dt.month,column="became_member_month") # month
        df.insert(loc=5,value=df.became_member_on.dt.day,column="became_member_day") # day
        #to get login days
        max_day=date(2018,7,26)
        max_day=pd.to_datetime(max_day)
        value=(max_day - df.became_member_on).dt.days
        df.insert(loc=6,value=value,column="login_days")
        
        #convert time to object
        df.time=df.time.astype("O")
        
        df.drop("became_member_on",axis=1,inplace=True)
        
        return df
    data=clean(df)
    pre=preprocessor.transform(data)
    value_predict=model.predict(pre)[0]
        
    btn=st.button("Predict")
    if btn:
         if value_predict == 0:
            st.write("No offer")
         elif value_predict==1:
            st.write("offer is discount")
            st.write(f"Details : 5% discount ,minimum required spend to complete an offer =20 Dollar ,offer is open for 10 day ")
         elif value_predict==2:
            st.write("offer is discount")
            st.write(f"Details : 3% discount ,minimum required spend to complete an offer =7 Dollar ,offer is open for 7 day ")
         elif value_predict== 3:
            st.write("offer is discount")
            st.write(f"Details : 2% discount ,minimum required spend to complete an offer =10 Dollar ,offer is open for 7 day ")
            
         elif value_predict==4:
            st.write("offer is informational")
            st.write("offer is open for 4 day ")
        
         elif value_predict==5:
            st.write("offer is buy one get one [BOGO]")
            st.write("Details : minimum required spend to complete an offer =10 Dollar ,offer is open for 5 day ")
            
         elif value_predict==6:
            st.write("offer is informational")
            st.write("Details : offer is open for 3 day ")
            
         elif value_predict==7:
            st.write("offer is buy one get one [BOGO]")
            st.write("Details :minimum required spend to complete an offer =5 Dollar ,offer is open for 7 day ")
