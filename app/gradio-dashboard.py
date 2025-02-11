import pandas as pd 
import numpy as np 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings 
import gradio as gr  


#############################################################
# Load the dataset containing book information and emotions #
#############################################################
books = pd.read_csv("../data/processed/books_with_emotions.csv")


###############################################
#   Add a column for larger book thumbnails   #
###############################################
books['large_thumbnail'] = books['thumbnail'] + '&fife=w800'  # Append a parameter to the URL for larger images
# Replace missing thumbnails with a default image
books['large_thumbnail'] = np.where(books['large_thumbnail'].isna(),
                                    '../img/cover-not-found.jpg', 
                                    books['large_thumbnail'])


#######################################################################
#   Load book descriptions from a text file and Store the documents   #
#######################################################################
row_documents = TextLoader('../data/external/tagged_description.txt').load()
# Split the text into individual documents (one per line)
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(row_documents)

# Use a pre-trained embedding model from Hugging Face for text embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store the documents in ChromaDB for vector-based similarity search
db_books = Chroma.from_documents(documents, embedding=embedding_model)


#################################################################################################
#   Function to retrieve book recommendations based on semantic similarity, category, and tone  #
#################################################################################################
def retrieve_semantic_recommendations(
        query: str,  # User query (e.g., book description)
        category: str = None,  # Filter by category (e.g., Fiction, Non-Fiction)
        tone: str = None,  # Filter by emotional tone (e.g., Happy, Sad)
        initial_top_k: int = 50,  # Number of initial recommendations to retrieve
        final_top_k: int = 16,  # Number of final recommendations to return
) -> pd.DataFrame:
    
    # Perform a similarity search to retrieve the top-k recommendations
    recs = db_books.similarity_search(query, k=initial_top_k)
    # Extract ISBNs from the recommendations and clean them (remove colons and quotes)
    books_list = [int(rec.page_content.strip('"').split()[0].replace(':', '')) for rec in recs]

    # Filter the books DataFrame to include only the recommended books
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # Apply category filter if specified
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Sort recommendations by emotional tone if specified
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

##########################################################
#   Function to format and display book recommendations  #
##########################################################
def recommend_books(
        query: str,  # User query (e.g., book description)
        category: str, 
        tone: str  
):
    # Retrieve recommendations based on the query, category, and tone
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    # Format each recommendation for display
    for _, row in recommendations.iterrows():
        # Truncate the description to 30 words
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        # Format authors' names (e.g., "Author1 and Author2" or "Author1, Author2, and Author3")
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # Create a caption for the book (title, authors, and truncated description)
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        # Add the book's thumbnail and caption to the results
        results.append((row["large_thumbnail"], caption))
    return results


# Define available categories and tones for the dropdown menus
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


##################################
#   Build the Gradio dashboard   #
##################################
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")  # Title of the dashboard

    with gr.Row():
        # Input components
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    # Output component for displaying recommendations
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    # Link the submit button to the recommend_books function
    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

#########################
# Launch the dashboard  #
#########################
if __name__ == "__main__":
    dashboard.launch()