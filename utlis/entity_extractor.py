import re
from collections import defaultdict, Counter
import nltk
from transformers import pipeline

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import ne_chunk, pos_tag

class EntityExtractor:
    def __init__(self):
        self.setup_ner_model()
        
        # قواميس الكيانات الشائعة في الأخبار
        self.location_keywords = {
            'countries': ['america', 'usa', 'china', 'russia', 'germany', 'france', 'italy', 'spain', 'japan', 'korea'],
            'cities': ['washington', 'beijing', 'moscow', 'london', 'paris', 'tokyo', 'new york', 'los angeles']
        }
        
        self.organization_keywords = {
            'companies': ['google', 'microsoft', 'apple', 'amazon', 'facebook', 'tesla', 'netflix', 'twitter'],
            'institutions': ['nasa', 'fbi', 'cia', 'un', 'who', 'fifa', 'olympics']
        }
    
    def setup_ner_model(self):
        """إعداد نموذج التعرف على الكيانات المسماة"""
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=-1  # استخدام CPU
            )
            print("✅ تم تحميل نموذج BERT للكيانات المسماة")
        except Exception as e:
            print(f"⚠️ تعذر تحميل نموذج BERT: {e}")
            self.ner_pipeline = None
    
    def extract_with_nltk(self, text):
        """استخراج الكيانات باستخدام NLTK"""
        entities = {
            'PERSON': [],
            'ORGANIZATION': [],
            'LOCATION': [],
            'DATE': [],
            'MONEY': [],
            'PERCENT': []
        }
        
        try:
            # تقسيم النص إلى جمل
            sentences = sent_tokenize(text)
            
            for sentence in sentences:
                # تقسيم إلى كلمات وتحديد أنواع الكلمات
                tokens = word_tokenize(sentence)
                pos_tags = pos_tag(tokens)
                
                # التعرف على الكيانات المسماة
                chunks = ne_chunk(pos_tags)
                
                current_entity = []
                current_label = None
                
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        if current_label != chunk.label():
                            if current_entity and current_label:
                                entity_text = ' '.join(current_entity)
                                if current_label in entities:
                                    entities[current_label].append(entity_text)
                            
                            current_entity = [chunk[0][0]]
                            current_label = chunk.label()
                        else:
                            current_entity.append(chunk[0][0])
                    else:
                        if current_entity and current_label:
                            entity_text = ' '.join(current_entity)
                            if current_label in entities:
                                entities[current_label].append(entity_text)
                        current_entity = []
                        current_label = None
                
                # إضافة آخر كيان
                if current_entity and current_label and current_label in entities:
                    entity_text = ' '.join(current_entity)
                    entities[current_label].append(entity_text)
            
        except Exception as e:
            print(f"خطأ في استخراج الكيانات باستخدام NLTK: {e}")
        
        return entities
    
    def extract_with_transformers(self, text):
        """استخراج الكيانات باستخدام Transformers"""
        entities = defaultdict(list)
        
        if not self.ner_pipeline:
            return dict(entities)
        
        try:
            # تقسيم النص إذا كان طويلاً
            max_length = 512
            if len(text) > max_length:
                text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            else:
                text_chunks = [text]
            
            for chunk in text_chunks:
                results = self.ner_pipeline(chunk)
                
                for entity in results:
                    entity_type = entity['entity_group']
                    entity_text = entity['word']
                    confidence = entity['score']
                    
                    # تنظيف النص
                    entity_text = entity_text.strip()
                    if entity_text and confidence > 0.7:  # عتبة الثقة
                        entities[entity_type].append({
                            'text': entity_text,
                            'confidence': confidence
                        })
            
        except Exception as e:
            print(f"خطأ في استخراج الكيانات باستخدام Transformers: {e}")
        
        return dict(entities)
    
    def extract_dates(self, text):
        """استخراج التواريخ باستخدام التعبيرات النمطية"""
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YYYY or MM/DD/YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'  # DD Month YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dates.append(match.group())
        
        return list(set(dates))  # إزالة المكررات
    
    def extract_numbers(self, text):
        """استخراج الأرقام والنسب والأموال"""
        patterns = {
            'money': r'\$[\d,]+(?:\.\d{2})?|\b\d+\s*(?:dollar|euro|pound|yen)s?\b',
            'percentage': r'\d+(?:\.\d+)?%',
            'numbers': r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'
        }
        
        results = {}
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            results[category] = matches
        
        return results
    
    def extract_events(self, text):
        """استخراج الأحداث المحتملة"""
        # كلمات مفتاحية تدل على الأحداث
        event_keywords = [
            'conference', 'meeting', 'summit', 'election', 'war', 'attack',
            'protest', 'strike', 'announcement', 'launch', 'release',
            'merger', 'acquisition', 'bankruptcy', 'investment'
        ]
        
        events = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in event_keywords:
                if keyword in sentence_lower:
                    events.append({
                        'keyword': keyword,
                        'sentence': sentence.strip(),
                        'context': sentence
                    })
                    break  # تجنب إضافة نفس الجملة عدة مرات
        
        return events
    
    def comprehensive_extract(self, text):
        """استخراج شامل لجميع الكيانات"""
        results = {
            'nltk_entities': self.extract_with_nltk(text),
            'transformer_entities': self.extract_with_transformers(text),
            'dates': self.extract_dates(text),
            'numbers': self.extract_numbers(text),
            'events': self.extract_events(text)
        }
        
        return results
    
    def merge_entities(self, nltk_entities, transformer_entities):
        """دمج نتائج الاستخراج من مصادر مختلفة"""
        merged = defaultdict(list)
        
        # إضافة كيانات NLTK
        for entity_type, entities in nltk_entities.items():
            merged[entity_type].extend(entities)
        
        # إضافة كيانات Transformers
        for entity_type, entities in transformer_entities.items():
            if isinstance(entities, list) and entities:
                if isinstance(entities[0], dict):
                    entity_texts = [e['text'] for e in entities]
                else:
                    entity_texts = entities
                merged[entity_type].extend(entity_texts)
        
        # إزالة المكررات
        for entity_type in merged:
            merged[entity_type] = list(set(merged[entity_type]))
        
        return dict(merged)
    
    def get_entity_statistics(self, entities):
        """إحصائيات الكيانات المستخرجة"""
        stats = {}
        total_entities = 0
        
        for entity_type, entity_list in entities.items():
            if isinstance(entity_list, list):
                count = len(entity_list)
                stats[entity_type] = count
                total_entities += count
        
        stats['total'] = total_entities
        return stats