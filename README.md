---
title: Book Recommender
emoji: 😻
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
license: mit
short_description: This project implements an intelligent book recommendation
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
# 📚 Book Recommender System

This project is a book recommendation system designed to suggest books to users based on their preferences and reading history. It utilizes collaborative filtering and content-based filtering techniques to provide personalized book recommendations.

---

### Explanation of the Code
1. **Data Loading:**
   - The book dataset (`books_with_emotions.csv`) is loaded using `pandas`.
   - Book descriptions are loaded from a text file (`tagged_description.txt`) and split into individual documents.

2. **Embeddings and Vector Database:**
   - Text embeddings are generated using a pre-trained Hugging Face model (`sentence-transformers/all-MiniLM-L6-v2`).
   - The embeddings are stored in ChromaDB for semantic similarity search.

3. **Recommendation Logic:**
   - The `retrieve_semantic_recommendations` function retrieves book recommendations based on semantic similarity, category, and emotional tone.
   - The `recommend_books` function formats the recommendations for display.

4. **Gradio UI:**
   - A Gradio dashboard is created with input components (textbox, dropdowns) and an output component (gallery).
   - The `recommend_books` function is linked to the "Find recommendations" button.

5. **Launching the App:**
   - The Gradio dashboard is launched using `dashboard.launch()`.


## � Features
- 📌 **Data Preprocessing:** Cleans and structures raw book and user interaction data.
- 📖 **Book Recommendations:** Provides personalized book suggestions based on user preferences.
- 🔍 **Search Functionality:** Allows users to search for books by title, author, or genre.
- 📊 **User Interaction Tracking:** Tracks user interactions to improve recommendation accuracy over time.
- ⚙️ **Customizable Parameters:** Adjust recommendation algorithms to suit different user bases.

---

## 🔍 How It Works

1. **Data Collection:** Gathers data on books and user interactions (ratings, reviews, etc.).
2. **Data Processing:** Cleans and structures the data for analysis.
3. **Model Training:** Uses collaborative filtering and content-based filtering to train recommendation models.
4. **Recommendation Generation:** Generates book recommendations based on user data and preferences.
5. **User Feedback:** Incorporates user feedback to refine and improve recommendations.

---

## 🎯 Why This Project?

- 📌 Helps users discover new books tailored to their interests.
- 🔍 Enhances user experience by providing relevant and personalized recommendations.
- 📊 Tracks user interactions to continuously improve recommendation accuracy.
- 🛠️ Customizable for different user bases and book genres.

---

## 🛠 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mohamedkaram400/llm-book-recommendation-system

cd book-recommender

```

### 2️⃣ Create and Activate a Conda Environment

```bash
conda create --name llm-book-recommendation-system python=3.11 -y
conda llm-book-recommendation-system
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Running the Book Recommender System

Once you've installed the dependencies, follow these steps to run the project:

```
python3 book-recommender/app/gradio-dhashboard.py
```

### Go to ``` http://127.0.0.1:7860/```

### 📜 License
This project is open-source under the MIT License.

