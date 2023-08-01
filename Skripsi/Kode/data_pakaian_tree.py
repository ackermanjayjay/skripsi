from sklearn.feature_extraction.text import TfidfVectorizer
import joblib,os
import pickle

# load vector tfidf
# pakaian_vectorizer = pickle.load(open("model-tfidf/tf-idf-pakaian.pkl","rb"))

# load_vector_pakaian=open("model-tfidf/tf-idf-pakaian.pkl","rb")
# load_tfidf=joblib.load(load_vector_pakaian)

#Load model
def loaded_model_tree_pakaian_40Persen(text):
   loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("model-tfidf/feature_tf-idf-pakaian.pkl", "rb"))))
   predictor = joblib.load(open(os.path.join("model-tree\model_dec_pakaian_40persen.pkl"),'rb'))
   prediction = predictor.predict(loaded_vec.fit_transform([text]))
   return prediction


