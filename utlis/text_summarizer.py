import re
import nltk
from collections import Counter
from transformers import pipeline
import numpy as np

# تحميل المكتبات المطلوبة
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class TextSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.setup_transformers()
    
    def setup_transformers(self):
        """إعداد نماذج Transformers للتلخيص"""
        try:
            self.summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                device=-1  # استخدام CPU
            )
            print("✅ تم تحميل نموذج BART للتلخيص")
        except Exception as e:
            print(f"⚠️ تعذر تحميل نموذج BART: {e}")
            self.summarizer = None
    
    def extractive_summary(self, text, num_sentences=3):
        """التلخيص الاستخراجي باستخدام تكرار الكلمات"""
        # تنظيف النص
        clean_text = re.sub(r'\s+', ' ', text)
        
        # تقسيم إلى جمل
        sentences = sent_tokenize(clean_text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # تقسيم إلى كلمات وإزالة كلمات الوقف
        words = word_tokenize(clean_text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        # حساب تكرار الكلمات
        word_freq = Counter(words)
        
        # تسجيل نقاط للجمل بناءً على تكرار الكلمات
        sentence_scores = {}
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            sentence_words = [word for word in sentence_words if word.isalnum()]
            
            score = 0
            word_count = 0
            for word in sentence_words:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            
            if word_count > 0:
                sentence_scores[sentence] = score / word_count
        
        # اختيار أفضل الجمل
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        summary_sentences = [sent[0] for sent in top_sentences[:num_sentences]]
        
        # ترتيب الجمل حسب ظهورها في النص الأصلي
        ordered_sentences = []
        for sentence in sentences:
            if sentence in summary_sentences:
                ordered_sentences.append(sentence)
        
        return ' '.join(ordered_sentences)
    
    def abstractive_summary(self, text, max_length=150, min_length=50):
        """التلخيص التجريدي باستخدام BART"""
        if not self.summarizer:
            return "نموذج التلخيص التجريدي غير متاح. استخدم التلخيص الاستخراجي بدلاً من ذلك."
        
        try:
            # تقسيم النص إذا كان طويلاً جداً
            if len(text) > 1024:
                text = text[:1024]
            
            result = self.summarizer(
                text, 
                max_length=max_length, 
                min_length=min_length,
                do_sample=False
            )
            
            return result[0]['summary_text']
            
        except Exception as e:
            print(f"خطأ في التلخيص التجريدي: {e}")
            return self.extractive_summary(text)
    
    def get_summary_stats(self, original_text, summary):
        """إحصائيات التلخيص"""
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        
        compression_ratio = (1 - summary_words / original_words) * 100 if original_words > 0 else 0
        
        return {
            'original_word_count': original_words,
            'summary_word_count': summary_words,
            'compression_ratio': round(compression_ratio, 2)
        }
    
    def multi_summary(self, text, methods=['extractive', 'abstractive']):
        """تلخيص متعدد الطرق"""
        results = {}
        
        if 'extractive' in methods:
            results['extractive'] = self.extractive_summary(text)
        
        if 'abstractive' in methods and self.summarizer:
            results['abstractive'] = self.abstractive_summary(text)
        
        return results
    
    def key_phrases_extraction(self, text, num_phrases=5):
        """استخراج العبارات المفتاحية"""
        # تنظيف النص
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = word_tokenize(clean_text)
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # إنشاء عبارات من كلمتين
        bigrams = []
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            bigrams.append(bigram)
        
        # حساب التكرار
        word_freq = Counter(words)
        bigram_freq = Counter(bigrams)
        
        # أفضل الكلمات والعبارات
        top_words = [word for word, _ in word_freq.most_common(num_phrases)]
        top_bigrams = [bigram for bigram, _ in bigram_freq.most_common(num_phrases)]
        
        return {
            'keywords': top_words,
            'key_phrases': top_bigrams
        }