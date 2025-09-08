import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import string
import joblib



model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

def get_semantic_similarity(question,option):
    embeddings = model.encode([question,option])

    return cosine_similarity([embeddings[0]],[embeddings[1]])[0][0]



mcqs_hard_df = pd.read_csv('arc_challenge_full.csv',encoding='latin1')
mcqs_easy_df = pd.read_csv('arc_easy_full.csv',encoding='latin1')
QueAnsDf = pd.read_csv('sciq_full.csv',encoding='latin1')

#Merging all 3 of them
QueAnsDf = QueAnsDf[['question','correct_answer','distractor1','distractor2','distractor3']]

QueAnsDf['choices'] = QueAnsDf.apply(
    lambda row : [row['correct_answer'],row['distractor1'],row['distractor2'],row['distractor3']],axis=1
)

def find_correct_answer_key(row):
    choices = row['choices']
    correct_answer=row['correct_answer']
    try:
        index = choices.index(correct_answer)
        return ['A','B','C','D'][index]
    except ValueError:
        return np.nan
    
QueAnsDf['answerKey'] = QueAnsDf.apply(find_correct_answer_key,axis=1)

QueAnsDf['id'] = range(len(QueAnsDf))
desired_columns = ['id','question','choices','answerKey']

mcqs_easy_df = mcqs_easy_df[desired_columns]
mcqs_hard_df  = mcqs_hard_df[desired_columns]
QueAnsDf = QueAnsDf[desired_columns]

print('Question and answers Dataset head :-\n',QueAnsDf.head())

df_list = [mcqs_easy_df,mcqs_hard_df,QueAnsDf]
df = pd.concat(df_list,ignore_index=True)

new_data = []

for index,row in df.iterrows():
    try:
        choices_list = eval(str(row['choices']))
    except(SyntaxError,TypeError):
        continue
    for idx,choice in enumerate(choices_list):
        label=''
        text =''

        if isinstance(choice,dict):
            label = choice.get('label','')
            text = choice.get('text','')
        elif isinstance(choice,str):
            label = ['A','B','C','D'][idx]
            text = choice
        else:
            continue
        is_correct = 1 if label == row['answerKey'] else 0

        new_data.append({
            'question' : row['question'],
            'text' : text,
            'is_correct' : is_correct
        })

           
df = pd.DataFrame(new_data)
print(df.head())
print(df.columns)

#Text Preprocessing
def lowercasing(txt):
    return txt.lower()

def remove_punctuation(txt):
    return txt.translate(str.maketrans('','',string.punctuation))

def remove_numbers(txt):
    return "".join([i for i in txt if not i.isdigit()])

def remove_stopwords(txt):
    stop_words = set(stopwords.words('english'))
    try:
        words = word_tokenize(txt)
    except TypeError as e:
        print(f"Error Occured {e}")
        return ""
    cleaned_words  = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)


df['text'] = df['text'].apply(lowercasing)
df['text'] = df['text'].apply(remove_numbers)
df['text'] = df['text'].apply(remove_punctuation)
df['text'] = df['text'].apply(remove_stopwords)

#Exploratry Data Analysis(EDA)
print('Total number of questions',df['question'].nunique())
print('Total Number of options :-', len(df))

plt.figure()
sns.histplot(x=df['is_correct'],palette='viridis')
plt.title('Distribution of correct / incorrect answers')

avg_question_len = df['question'].str.split().str.len()
avg_option_len = df['text'].str.split().str.len()

plt.figure()
sns.histplot(x=avg_option_len,palette='deep')
plt.title('Avg option Length')

plt.figure()
sns.histplot(x=avg_question_len,palette='rainbow')
plt.title('Avg Question length')


plt.show()

#Semantic Similiarity
df['semantic_similarity'] = df.apply(
    lambda row : get_semantic_similarity(row['question'],row['text']),
    axis=1
)
print("After apply semantic similarity",df.head())




#TF-IDF Vectorization
x_text = df['question'].fillna('') + " " + df['text'].fillna('')

vectorizer= TfidfVectorizer()


y = df['is_correct']

x_train_text,x_test_text,y_train,y_test = train_test_split(x_text,y,random_state=42,test_size=0.33)


x_train_tf = vectorizer.fit_transform(x_train_text)
x_test_tf = vectorizer.transform(x_test_text)
print('Length of vocabulary:', len(vectorizer.vocabulary_))

semantic_similarity_train = df.loc[y_train.index, 'semantic_similarity'].fillna(0).values.astype(float).reshape(-1, 1)
semantic_similarity_test = df.loc[y_test.index, 'semantic_similarity'].fillna(0).values.astype(float).reshape(-1, 1)


x_train_combined = hstack([x_train_tf, semantic_similarity_train]).tocsr().astype(np.float64)
x_test_combined = hstack([x_test_tf, semantic_similarity_test]).tocsr().astype(np.float64)


#Model Training and evaluation


#Logistic Regresson

lg = LogisticRegression()
lg.fit(x_train_combined,y_train)

y_pred_lg = lg.predict(x_test_combined)
accuracy_lg = accuracy_score(y_test,y_pred_lg)
f1_lg = f1_score(y_test,y_pred_lg)

print('Accuracy Score of Logisitic regression :-',accuracy_lg)
print('F1 Score of Logistic regression :-',f1_lg)

# Create a new DataFrame for analysis
test_results = pd.DataFrame({
    'question': df.loc[y_test.index, 'question'],
    'option_text': df.loc[y_test.index, 'text'],
    'true_label': y_test,
    'predicted_label': y_pred_lg
})

incorrect_predictions = test_results[test_results['true_label'] != test_results['predicted_label']]
print('\n Number of Incorrect prediction that our Logistic regression model made :-',len(incorrect_predictions))

for idx,row in incorrect_predictions.head(5).iterrows():
    print(f"\n Questions : {row['question']}")
    print(f"Option : {row['option_text']}")
    print(f"True Label: {'Correct' if row['true_label'] == 1 else 'Incorrect'}")
    print(f"Predicted Label: {'Correct' if row['predicted_label'] == 1 else 'Incorrect'}")
    print("---")

#Naive Bayes

nb = MultinomialNB()
nb.fit(x_train_tf,y_train) # Naive Bayes works better on TF-IDF Alone

y_pred_nb = nb.predict(x_test_tf)

accuracy_nb = accuracy_score(y_test,y_pred_nb)
f1_nb = f1_score(y_test,y_pred_nb)

print('\nAccuracy Score of Naive Bayes :-',accuracy_nb)
print('\n F1 score of naive bayes :-',f1_nb)

#Random Forest Classifier

rf = RandomForestClassifier()
rf.fit(x_train_combined,y_train)

y_pred_rf = rf.predict(x_test_combined)

accuracy_rf = accuracy_score(y_test,y_pred_rf)
f1_rf = f1_score(y_test,y_pred_rf)


print('Accuracy Score  of random forest:-',accuracy_rf)
print('F1-Score of random forest:-',f1_rf)


print('\n --- Final Comparison Chart of Models --- \n')

final_perfomance_table = {
    'Accuracy LG' : accuracy_lg,
    'F1-Score LG' : f1_lg,
    'Accuracy NB' : accuracy_nb,
    'F1-Score NB' : f1_nb,
    'Accuracy RF' : accuracy_rf,
    'F1-Score RF' : f1_rf
}

print(final_perfomance_table)

scores_df = pd.DataFrame(list(final_perfomance_table.items()),columns=['Metric','Score'])

scores_df['Type'] = scores_df['Metric'].apply(lambda x: x.split(' ')[0])
scores_df['Model'] = scores_df['Metric'].apply(lambda x : x.split(' ')[1])


plt.figure(figsize=(12,7))
sns.barplot(x='Model',y='Score',hue='Type',data=scores_df,palette='viridis')
plt.title('Model Perfomances Visualized : Accuracy vs F1-Score')

plt.show()


#JobLib
best_model = lg
joblib.dump(best_model,'best_model.joblib')
joblib.dump(vectorizer,'vectorizer.joblib')
