Job Recommendation System using Natural Language Understanding

Project Proposal

Team Members:
Akash Poddar and 
Avani Rao

Objective-
The objective of this project is to develop a Job Recommendation System that leverages Natural Language Understanding (NLU) and Machine Learning to match job seekers with relevant job opportunities based on their skills, experience, and preferences. By analyzing resumes and job descriptions, the system provides personalized job recommendations to streamline the job search process.

Problem Statement-
Traditional job search platforms often fail to provide personalized job recommendations that align with users' skills and career goals. Generic filtering methods lack contextual understanding and fail to capture job seekers’ nuanced expertise. To address this, our system integrates Natural Language Processing (NLP) and Machine Learning (ML) techniques to deliver accurate job recommendations.

Methodology

1. Data Collection & Scraping-

Job-related data is scraped from Indeed using the Apify API.
Extracted attributes include:
Job Title, Salary Estimates, Company Ratings, Locations, Industries, and Job Descriptions.
A dataset of 5000 job postings for 30+ technical roles is curated.

2. Feature Engineering & Preprocessing-

Text Cleaning & Processing:
Convert all text to lowercase.
Remove stopwords, punctuation, and special characters.
Tokenization & Lemmatization using SpaCy.
Handling Numerical Features:
Process salary estimates, company ratings, and outliers using feature scaling.
TF-IDF Vectorization:
Convert job descriptions and resumes into numerical vectors for text similarity analysis.

3. Machine Learning Model-

K-Nearest Neighbors (KNN) Algorithm is used to find the most relevant job postings based on:
Resume Skills Extraction (via SpaCy NLP library).
Cosine Similarity between TF-IDF vectors (resume vs. job descriptions).
The model ranks job recommendations based on relevance scores.

4. Streamlit Web Application-

A Streamlit-based web application is developed for user interaction.
Users can upload their resume (PDF format), specify the number of recommendations, and download job matches as a CSV.
The app provides real-time job recommendations based on extracted skills.

5. Deployment on Google Cloud Platform (GCP)-

The application is containerized using Docker.
Hosted on Google Cloud Run for scalability and global accessibility.
An external IP address is provided for users to access the deployed system online.

Expected Outcomes-

Efficient resume parsing and skill extraction using NLP.
High-accuracy job recommendations based on KNN-based similarity scoring.
Scalable web application that allows users to receive job recommendations in real-time.
Deployment on Google Cloud ensures accessibility and reliability.

Future Enhancements-

Integrating Deep Learning & Large Language Models (LLMs) like GPT for improved skill extraction.
Real-time labor market trend analysis to refine recommendations.
Multilingual support for global accessibility.
Skill gap analysis to suggest relevant learning resources.

Conclusion-

This project leverages Natural Language Understanding, Machine Learning, and Cloud Deployment to create a robust Job Recommendation System. By combining TF-IDF, KNN, and skill extraction techniques, it offers a data-driven, user-centric approach to help job seekers find the most relevant job opportunities efficiently.
