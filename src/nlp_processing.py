import pandas as pd
import string
import nltk

from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load preprocessed dataset
df = pd.read_csv(
    r"C:\Users\Mihi\Desktop\Behaviour Pattern Identification Model\dataset\processed\behaviour_processed.csv"
)

# Show columns
print(df.columns.tolist())

def clean_text(text):

    # convert to lowercase
    text = str(text).lower()

    # remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # remove stopwords
    stop_words = set(stopwords.words('english'))

    words = text.split()

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(filtered_words)

df['cleaned_feedback'] = df['Peer Feedback'].apply(clean_text)

df.to_csv(
    r"C:\Users\Mihi\Desktop\Behaviour Pattern Identification Model\dataset\processed\nlp_processed.csv",
    index=False
)

print("NLP processing completed successfully.")

