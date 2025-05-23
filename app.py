# Импорт
import re
import os
import spacy
import pandas as pd
import numpy as np
import torch
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from natasha import Segmenter, NewsNERTagger, NewsEmbedding, Doc
import emoji
import re
import pymorphy2
import json
from lime.lime_text import LimeTextExplainer
import joblib


# Получаем путь к текущему файлу
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

# Загрузка статистики по признакам
with open(os.path.join(MODELS_DIR,"df_stats.json"), 'r', encoding='utf-8') as f:
    df_stats = json.load(f)

# Загрузка топ-вирусных слов
top_viral_words = joblib.load(os.path.join(MODELS_DIR, "top_viral_words.pkl"))

# Загружаем русскую модель spaCy
nlp = spacy.load("ru_core_news_sm")

# Загрузка моделей
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
bert_model = SentenceTransformer(os.path.join(MODELS_DIR, "best_sentence_transformer"))

# Инициализация Natasha
segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

# Инициализация морфологического анализатора
morph = pymorphy2.MorphAnalyzer()

# Загрузка сленга
slang_dict = {}
try:
    with open(os.path.join(MODELS_DIR, "slang_dict.txt"), "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                slang_dict[k] = v
except FileNotFoundError:
    print("⚠️ Словарь сленга не найден.")

# Токенизатор для сентимента
tokenizer_sentiment = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny-sentiment-balanced")
model_sentiment = AutoModelForSequenceClassification.from_pretrained("cointegrated/rubert-tiny-sentiment-balanced")
# %%
# Пример использования
explainer = LimeTextExplainer(class_names=["Не вирусный", "Вирусный"])

def remove_repeated_letters(text, max_repeats=2):
    def reduce_repeats(match):
        char = match.group(0)[0]
        return char * max_repeats
    return re.sub(r'(.)\1{2,}', reduce_repeats, text)

def convert_emoji_to_text(text):
    return emoji.demojize(text, delimiters=(" :", ": "))

def replace_slang(text, slang_dict=None):
    words = text.split()
    replaced = [slang_dict[word] if slang_dict and word in slang_dict else word for word in words]
    return ' '.join(replaced)

def preprocess_tweet(text, slang_dict=None):
    text = str(text).lower()
    text = convert_emoji_to_text(text)
    if slang_dict:
        text = replace_slang(text, slang_dict)
    text = remove_repeated_letters(text)
    return text.strip()

def extract_features_ru(text):
    # Проверка на некорректный тип данных
    if not isinstance(text, str) or len(str(text).strip()) == 0:
        print("❌ Ошибка: Текст пустой или имеет некорректный тип:", repr(text))
        # Возвращаем дефолтные значения
        return {
            'tweet_length': 0,
            'word_count': 0,
            'has_hashtag': 0,
            'has_mention': 0,
            'has_url': 0,
            'contains_caps': 0,
            'contains_emoji': 0,
            'contains_question': 0,
            'contains_exclamation': 0,
            'avg_word_length': 0,
            'punctuation_density': 0,
            'sentiment_score': 0.0,
            'reading_ease': 100.0,
            'subjectivity_score': 0.0,
            'polarity_score': 0.0,
            'named_entities_count': 0,
            'keyword_density': 0,
            'contains_keywords': 0,
            'url_count': 0,
            'caps_word_count': 0,
            'emoji_count': 0,
            'sentence_count': 0
        }

    text = str(text)
    features = {}
    try:
        words = word_tokenize(text.lower())
    except Exception as e:
        print(f"❌ Ошибка токенизации: {e}")
        return {k: 0 for k in features}  # Или вернуть дефолтные значения

    char_count = len(text)
    features['tweet_length'] = char_count
    word_count = len(words)
    features['word_count'] = word_count
    features['has_hashtag'] = int(bool(re.search(r'#\w+', text)))
    features['has_mention'] = int(bool(re.search(r'@\w+', text)))
    features['has_url'] = int(bool(re.search(r'https?://\S+', text)))
    caps_words = [w for w in words if w.isupper() and len(w) > 1]
    features['contains_caps'] = int(len(caps_words) > 0)
    emoji_list = [c for c in text if c in emoji.EMOJI_DATA]
    features['contains_emoji'] = int(len(emoji_list) > 0)
    features['contains_question'] = int('?' in text)
    features['contains_exclamation'] = int('!' in text)
    features['avg_word_length'] = round(sum(len(word) for word in words) / word_count, 2) if word_count else 0
    punctuation_count = sum(1 for c in text if c in '.,!?;:')
    features['punctuation_density'] = round(punctuation_count / word_count, 2) if word_count else 0
    try:
        inputs = tokenizer_sentiment(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model_sentiment(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
        sentiment_score = probs[1] - probs[0]
        features['sentiment_score'] = float(sentiment_score)
        avg_word_len = sum(len(w) for w in words) / word_count if word_count else 0
        features['reading_ease'] = 100 - (avg_word_len * 10)
        features['subjectivity_score'] = 1.0 - float(abs(sentiment_score))
        features['polarity_score'] = float(probs[1])
    except Exception as e:
        features['sentiment_score'] = 0.0
        avg_word_len = sum(len(w) for w in words) / word_count if word_count else 0
        features['reading_ease'] = 100 - (avg_word_len * 10)
        features['subjectivity_score'] = 0.0
        features['polarity_score'] = 0.0
    try:
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)
        entities = [(span.text, span.type) for span in doc.spans]
        features['named_entities_count'] = len(entities)
    except:
        features['named_entities_count'] = 0
    keywords_found = [k for k in KEYWORDS if k in text.lower()]
    features['keyword_density'] = round(len(keywords_found) / word_count, 2) if word_count else 0
    features['contains_keywords'] = int(len(keywords_found) > 0)
    features['url_count'] = len(re.findall(r'https?://\S+', text))
    features['caps_word_count'] = len(caps_words)
    features['emoji_count'] = len(emoji_list)
    sentences = sent_tokenize(text)
    features['sentence_count'] = len(sentences)
    return features


def is_valid_viral_word(word):
        """
        Проверяет, является ли слово подходящим для использования
        в качестве вирусного слова (не имя собственное, не дата и т.д.)
        """
        # Исключаем паттерны вроде club12345|name, id12345, ссылки
        invalid_patterns = [
            r"^club\d+\|.*$",     # club12345|name
            r"^id\d+$",           # id123456
            r"^https?://.*$",     # ссылки
            r".*\|.*",            # любые конструкции с |
        ]
        for pattern in invalid_patterns:
            if re.match(pattern, word):
                return False
    
        # Проверяем через NER: если слово — именованная сущность, исключаем
        doc = nlp(word)
        for ent in doc.ents:
            if ent.label_ in ["PER", "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "CARDINAL"]:
                return False
    
        # Слово прошло все проверки
        return True


# Генерация рекомендаций
def generate_strategic_recommendations(tweet_text, df_stats, top_viral_words, explanation_list=None):
    recommendations = []
    factors = {
        'tweet_length': {'value': len(tweet_text), 'threshold': df_stats['tweet_length']['good']},
        'has_hashtag': {'value': int('#' in tweet_text), 'threshold': df_stats['has_hashtag']['good']},
        'contains_emoji': {'value': int(any(c in emoji.EMOJI_DATA for c in tweet_text)), 'threshold': df_stats['contains_emoji']['good']}
    }
    if factors['tweet_length']['value'] < factors['tweet_length']['threshold']:
        recommendations.append("Увеличьте длину поста — короткие посты реже становятся вирусными.")
    if factors['has_hashtag']['value'] < factors['has_hashtag']['threshold']:
        recommendations.append("Добавьте хэштеги — они увеличивают охват.")
    if factors['contains_emoji']['value'] < factors['contains_emoji']['threshold']:
        recommendations.append("Добавьте эмодзи — они привлекают внимание.")
    if explanation_list:
        negative_words = sorted([(w, s) for w, s in explanation_list if s < -0.05], key=lambda x: x[1])
        for word, score in negative_words:
            suggestions = [w for w in top_viral_words if w != word and is_valid_viral_word(w)][:3]
            if suggestions:
                examples = "', '".join(suggestions)
                recommendations.append(f"Замените слово '{word}' на более вирусное. Например: '{examples}'")
    return recommendations or ["Пост выглядит хорошо оптимизированным."]

def predict_viral(text):
    # Защита от None, чисел, пустых значений
    if not isinstance(text, str) or len(str(text).strip()) == 0:
        print("❌ Ошибка: Текст должен быть непустой строкой.")
        return {
            "class": "Ошибка",
            "confidence": 0.0,
            "explanation_html": "<div>Ошибка: текст пустой или имеет неверный формат.</div>",
            "recommendations": ["Текст пустой или имеет неверный формат."]
        }

    text = str(text).strip()
    processed_text = preprocess_tweet(text, slang_dict)
    
    # BERT эмбеддинг
    bert_embedding = bert_model.encode([processed_text], batch_size=1, show_progress_bar=False, convert_to_numpy=True)

    # Числовые признаки
    numeric_features = extract_features_ru(processed_text)
    numeric_df = pd.DataFrame([numeric_features])
    numeric_scaled = scaler.transform(numeric_df)

    # Объединение
    X_combined = np.hstack((bert_embedding, numeric_scaled))

    # Прогноз
    proba = model.predict_proba(X_combined)[0]
    predicted_class = model.predict(X_combined)[0]

    class_label = "Да" if predicted_class == 1 else "Нет"
    confidence = proba[predicted_class]

    # LIME объяснение
    def transform_for_lime(texts):
        processed_texts = [preprocess_tweet(t, slang_dict) for t in texts]
        bert_emb = bert_model.encode(processed_texts, batch_size=1, convert_to_numpy=True)
        return np.hstack((bert_emb, np.tile(numeric_scaled[0], (len(texts), 1))))

    exp = explainer.explain_instance(
        text,
        transform_for_lime,
        num_features=10,
        num_samples=500
    )

    # --- отображение в Jupyter ---
    print("\n📊 Влияние слов на виральность:")
    #exp.show_in_notebook(text=True, labels=(1,), predict_proba=False)
    # Вместо show_in_notebook
    explanation = exp.as_list(label=1)
    print("Explanation for class 1:")
    print(explanation)

    explanation_html = exp.as_html()

    # Рекомендации
    explanation_list = exp.as_list()
    recommendations = generate_strategic_recommendations(text, df_stats, top_viral_words, explanation_list)

    return {
        "class": class_label,
        "confidence": confidence,
        "explanation_html": explanation_html,
        "recommendations": recommendations
    }

text = input("Введите Ваше сообщение здесь: ...")

import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
KEYWORDS = ['выиграй', 'приз', 'бесплатно', 'денег', 'кликни', 'акция', 'скидка']
result = predict_viral(text)

print(f"Класс: {result['class']}")
print(f"Уверенность: {result['confidence']:.2%}")
print("Рекомендации:")
for r in result['recommendations']:
    print(" -", r)

# Сохраняем LIME только если он доступен
if 'explanation_html' in result:
    with open("lime_explanation.html", "w", encoding="utf-8") as f:
        f.write(result["explanation_html"])
else:
    print("⚠️ Нет данных для сохранения визуализации LIME.")