# Team members

Geetansh Kumar - geetansh@buffalo.edu - 50607410

Sataakshi Bhangelwal -  sataaksh@buffalo.edu - 50607324

Rudraksh Agarwal - rudraksh@buffalo.edu - 50604938

# Final Product Highlights

App is CI/CD enabled, for every push in the main branch, the latest version is deployed to the public URL. This has been done like any other real production website. One-click deplot system.

SQL Lite Database is persisted even on the public URL. So, the functionality is same for local and URL.

Restart option given if anything goes wrong, you can again load from raw and pre-process it again.

Querying enabled on both raw data and pre-processed data.

Code is modularized and structure like a production level code. Easy to understand, scalable and error free.

# Navigation
All the analysis for each group member can be found in the combined Python Notebook with proper section header names including the work and team member name in the header name. (Phase 2 nb is the exp folder in DIC_Project_phase2.ipynb)

app/ contains the complete app code and at the you can see the steps for building and running the app.

exp/ contains the final Python notebook and reports for all phases including final reasearch paper format report.

# A Deep Dive into Historical Olympic Performance Trends

This project aims to analyze the historical trends of Olympic performances, focusing on medal tallies, country-wise achievements, sport-wise achievements, and participation in various sports. By examining these patterns, we can discover insights such as the performance of certain countries in specific sports, shifts in performance over time, and variations in participation based on factors like gender and athlete age, etc. The goal is to identify and explore significant trends of the Olympics over the years.
In addition, we might take this analysis a step further by predicting the 2024 Olympic medal tally, with a plan to compare these predictions against the actual results (data strategy yet to be defined).

# Questions Geetansh

1. First Question:- What is the general trend in women participation country wise over the years? What countries are doing well and how do they compare to the best performing countries? (Phase 2)

This general question can lead to multiple hypotheses including the need of promoting women empowerment, understanding and highlighting gender disparity, urgent need to change in policies and awareness, etc.

This is a significant question because in this day and age we should ideally have equal women participation in sports, especially in a major competition like the Olympics. This also helps us identify which countries need to focus on the gender disparity issue more.

2. Second Question :- Are there any sports which are on the decline and losing popularity among participants? Also, are there some sports which have gained popularity over the recent years? (Phase 2)

This general question can again lead to multiple hypotheses including the need to spread awareness about some particular sports, predict which sport to remove from the Olympics, what sports have previously suffered this fate, etc.

This is a significant question because we should be aware of which sports are losing popularity to save them from getting extinct. On the world stage we should be able to identify which sports are on the rise, it can be beneficial for marketing and branding (great opportunity for money making). 


# Questions Sataakshi 

1. Question 1 - How do the trends in medal counts for team sports compare to those for individual sports across different countries over the years, and what insights can be drawn from these comparisons regarding each country's performance in the Olympic Games?  (Phase 2)

What?
This analysis compares the trends in medal counts for team sports versus individual sports across different countries over the years. Using queries to extract data I have analysed how the number of medals have been won in team events vs in individual events for different countries. This study provides an analysis of how countries perform in team versus individual competitions.
Why?

The question of how trends in medal counts for team sports compare to those for individual sports across different countries is significant for several reasons. It enhances our understanding of national strengths, allowing us to identify which countries excel in team versus individual sports and informing national sports policies. This analysis can also guide investment in sports programs. So, this question provides valuable insights that can shape the future of sports.

2. Question 2 - How has the participation of women athletes in various sports evolved, and what trends can be observed in terms of minimum and maximum participation levels across selected sports? (Phase 2)

What?
The analysis of women's participation in various sports over time reveals significant trends in their involvement. By filtering the data for women athletes, we observe how participation levels have increased in these years in all sports. 

Why?

The analysis aims to highlight advancements in women's participation in sports, demonstrating progress in inclusion and representation. By examining trends over time, we can assess the effectiveness of initiatives aimed at increasing female participation and identify sports where participation still lags. This information informs sports organizations and advocates about the current situation of women’s sports. This will help in gender equity in olympics.

# Questions Rudraksh

1. Table Tennis and Tennis are similar yet different sports. The players I have seen in both games seem to have different builds. The hypothesis is that we can build a model using Height, Weight, and athlete’s country to predict which sport they belong to. (Phase 2)

2. In athletics, height, weight, age, and country are major indicators of success in the Olympics. We have made models to achieve the same. (Predict the success/failure of athletics based on their height weight, and Country they belong to.) (Phase 2)

# How to run the code:
1. pip install -r app/requirements.txt
2. streamlit run app/app.py

Public URL to access UI
https://olympics-trends.streamlit.app/

App folder structure
app/
    --- csv_files/
        --- Olympic_Athlete_Biography.csv
        --- Olympic_Athlete_Event_Details.csv
        --- Olympic_Country_Profiles.csv
        --- Olympic_Event_Results.csv
        --- Olympic_Games_Summary.csv
        --- Olympic_Medal_Tally_History.csv
        --- population_total_long.csv

    --- app.py (main streamlit file that needs to be used)
    --- streamlit_helper.py (some helpers for streamlit)

    --- olympics_data.db (stores all the raw and processed data in tables)

    --- db_crud.py (database realted operationns CRUD)

    --- preprocess.py (preprocessing of raw data script)
    
    --- hypothesis.py (4 hypothesis)

# What steps to follow for the App run? We have got you covered!

1. Open the publicily hosted URL : https://olympics-trends.streamlit.app/
2. Click the upload button to upload all the required datasets. You'll get a prompt stating the success.
3. You'll be able to see all the tables and below you also have the add, delete and modify functionality for the selected table.
4. Now, Click the run preprocessing button to run the preprocessing script (available in the code) and you'll get all the preprocessed tables which will be later used for ML models. Again you have the add, delete and modify functionality for the selected table.
5. Below, is the list for all 4 hypthesis and 4 buttons for running
6. Click run hypothesis one by one only.
7. You'll see the visualizations and outputs for every hypothesis.
