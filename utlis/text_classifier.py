import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

class NewsClassifier:
    def __init__(self, model_path='models'):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.config = None
        self.load_components()
    
    def load_components(self):
        """تحميل جميع مكونات النموذج"""
        try:
            # تحميل النموذج
            model_file = os.path.join(self.model_path, 'lstm_classifier.h5')
            self.model = load_model(model_file)
            print("✅ تم تحميل النموذج بنجاح")
            
            # تحميل المُرمز
            tokenizer_file = os.path.join(self.model_path, 'tokenizer.pkl')
            with open(tokenizer_file, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print("✅ تم تحميل المُرمز بنجاح")
            
            # تحميل مُرمز التصنيفات
            encoder_file = os.path.join(self.model_path, 'label_encoder.pkl')
            with open(encoder_file, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("✅ تم تحميل مُرمز التصنيفات بنجاح")
            
            # تحميل الإعدادات
            config_file = os.path.join(self.model_path, 'config.pkl')
            with open(config_file, 'rb') as f:
                self.config = pickle.load(f)
            print("✅ تم تحميل الإعدادات بنجاح")
            
        except Exception as e:
            print(f"❌ خطأ في تحميل مكونات النموذج: {str(e)}")
            raise e
    
    def preprocess_text(self, text):
        """معالجة النص قبل التنبؤ"""
        # تحويل النص إلى تسلسل أرقام
        sequence = self.tokenizer.texts_to_sequences([text])
        
        # تطبيق padding
        padded = pad_sequences(
            sequence, 
            maxlen=self.config['max_len'], 
            padding='post', 
            truncating='post'
        )
        
        return padded
    
    def predict_single(self, text):
        """التنبؤ لنص واحد"""
        # معالجة النص
        processed_text = self.preprocess_text(text)
        
        # التنبؤ
        prediction = self.model.predict(processed_text, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        
        # تحويل الرقم إلى اسم الفئة
        class_name = self.config['class_names'][predicted_class]
        
        # احتمالات جميع الفئات
        all_probabilities = {}
        for i, prob in enumerate(prediction[0]):
            class_name_i = self.config['class_names'][i]
            all_probabilities[class_name_i] = float(prob)
        
        return {
            'predicted_class': class_name,
            'confidence': confidence,
            'all_probabilities': all_probabilities
        }
    
    def predict_batch(self, texts):
        """التنبؤ لمجموعة من النصوص"""
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        return results
    
    def get_class_names(self):
        """الحصول على أسماء الفئات"""
        return list(self.config['class_names'].values())
    
    def analyze_text_stats(self, text):
        """تحليل إحصائيات النص"""
        words = text.split()
        return {
            'word_count': len(words),
            'char_count': len(text),
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0
        }