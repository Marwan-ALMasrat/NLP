import streamlit as st
import pandas as pd
import json
import os
from collections import Counter
import re

# إعدادات الصفحة
st.set_page_config(
    page_title="محلل الأخبار الذكي",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تحميل CSS مخصص
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 2rem;
    }
    
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1E88E5;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# دوال تحليل بسيطة (محلية)
def simple_text_analysis(text):
    """تحليل نص بسيط باستخدام أدوات Python الأساسية"""
    if not text:
        return {}
    
    # إحصائيات أساسية
    words = text.split()
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # عد الكلمات
    word_count = len(words)
    char_count = len(text)
    sentence_count = len(sentences)
    
    # الكلمات الأكثر تكراراً
    word_freq = Counter(words)
    most_common = word_freq.most_common(10)
    
    # تحليل بسيط للمشاعر (كلمات مفتاحية)
    positive_words = ['جيد', 'ممتاز', 'رائع', 'إيجابي', 'سعيد', 'نجح', 'تقدم', 'فوز']
    negative_words = ['سيء', 'سلبي', 'فشل', 'مشكلة', 'خطأ', 'حزين', 'صعب', 'خسارة']
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    sentiment = "محايد"
    if positive_count > negative_count:
        sentiment = "إيجابي"
    elif negative_count > positive_count:
        sentiment = "سلبي"
    
    # تصنيف بسيط حسب الكلمات المفتاحية
    sports_words = ['كرة', 'مباراة', 'فريق', 'لاعب', 'بطولة', 'ملعب']
    politics_words = ['حكومة', 'رئيس', 'وزير', 'انتخابات', 'سياسة', 'برلمان']
    tech_words = ['تكنولوجيا', 'كمبيوتر', 'إنترنت', 'تطبيق', 'برنامج', 'رقمي']
    
    category_scores = {
        'رياضة': sum(1 for word in words if word in sports_words),
        'سياسة': sum(1 for word in words if word in politics_words),
        'تكنولوجيا': sum(1 for word in words if word in tech_words)
    }
    
    predicted_category = max(category_scores.items(), key=lambda x: x[1])
    if predicted_category[1] == 0:
        predicted_category = ('عام', 0)
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'sentence_count': sentence_count,
        'most_common_words': most_common,
        'sentiment': sentiment,
        'positive_words': positive_count,
        'negative_words': negative_count,
        'category': predicted_category[0],
        'category_confidence': predicted_category[1] / word_count if word_count > 0 else 0
    }

def simple_summarize(text, num_sentences=3):
    """تلخيص بسيط عن طريق أخذ أول وأهم الجمل"""
    if not text:
        return ""
    
    sentences = [s.strip() for s in text.split('.') if s.strip() and len(s) > 20]
    
    if len(sentences) <= num_sentences:
        return '. '.join(sentences) + '.'
    
    # أخذ الجملة الأولى + جمل عشوائية من الوسط
    summary_sentences = [sentences[0]]  # الجملة الأولى
    
    # إضافة جمل من الوسط حسب الطول
    if len(sentences) > 3:
        mid_sentences = sentences[1:-1]
        # ترتيب حسب الطول (الجمل الأطول عادة أكثر أهمية)
        mid_sentences.sort(key=len, reverse=True)
        summary_sentences.extend(mid_sentences[:num_sentences-1])
    
    return '. '.join(summary_sentences) + '.'

def extract_entities(text):
    """استخراج كيانات بسيط باستخدام regex والكلمات المفتاحية"""
    entities = []
    
    # البحث عن أسماء (كلمات تبدأ بحرف كبير)
    names = re.findall(r'\b[A-Z][a-z]+\b', text)
    for name in set(names):
        entities.append({'text': name, 'label': 'شخص', 'start': text.find(name)})
    
    # البحث عن أرقام
    numbers = re.findall(r'\b\d+\b', text)
    for number in set(numbers):
        entities.append({'text': number, 'label': 'رقم', 'start': text.find(number)})
    
    # البحث عن تواريخ بسيطة
    dates = re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b', text)
    for date in set(dates):
        entities.append({'text': date, 'label': 'تاريخ', 'start': text.find(date)})
    
    return entities

# العنوان الرئيسي
st.markdown('<h1 class="main-header">📰 محلل الأخبار الذكي</h1>', unsafe_allow_html=True)
st.markdown("---")

# تحذير بسيط
st.info("🔧 **وضع التطوير**: يتم استخدام أدوات تحليل مبسطة حاليًا")

# الشريط الجانبي
st.sidebar.title("🎛️ لوحة التحكم")
st.sidebar.markdown("---")

# اختيار نوع التحليل
analysis_type = st.sidebar.selectbox(
    "نوع التحليل",
    ["تحليل شامل", "تحليل المشاعر", "تصنيف النص", "تلخيص النص", "استخراج الكيانات"]
)

# خيارات إضافية
st.sidebar.markdown("### ⚙️ الإعدادات")
show_stats = st.sidebar.checkbox("عرض الإحصائيات التفصيلية", True)
show_word_freq = st.sidebar.checkbox("عرض تكرار الكلمات", True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 حول التطبيق")
st.sidebar.info(
    """
    **المميزات الحالية:**
    - تحليل النصوص الأساسي
    - تصنيف مبسط
    - تلخيص تلقائي
    - استخراج الكيانات
    - تحليل المشاعر
    
    **ملاحظة:** هذا إصدار مبسط للاختبار
    """
)

# المحتوى الرئيسي
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 إدخال النص")
    
    # إدخال النص
    user_text = st.text_area(
        "أدخل النص للتحليل:",
        height=300,
        placeholder="اكتب أو الصق النص هنا..."
    )

with col2:
    st.header("📊 المقاييس السريعة")
    
    if user_text:
        analysis = simple_text_analysis(user_text)
        
        st.metric("عدد الكلمات", analysis['word_count'])
        st.metric("عدد الأحرف", analysis['char_count'])
        st.metric("عدد الجمل", analysis['sentence_count'])
        
        # المشاعر
        sentiment_color = {
            'إيجابي': 'green',
            'سلبي': 'red',
            'محايد': 'gray'
        }
        
        st.markdown(f"**المشاعر:** <span style='color: {sentiment_color[analysis['sentiment']]}'>{analysis['sentiment']}</span>", 
                   unsafe_allow_html=True)

# معالجة التحليل
if user_text and st.button("🚀 بدء التحليل", type="primary"):
    
    with st.spinner("جاري التحليل..."):
        analysis = simple_text_analysis(user_text)
        
        if analysis_type == "تحليل شامل":
            st.header("🔍 التحليل الشامل")
            
            # تبويبات للنتائج
            tab1, tab2, tab3, tab4 = st.tabs(["الإحصائيات", "التصنيف", "التلخيص", "الكيانات"])
            
            with tab1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("الكلمات", analysis['word_count'])
                    st.metric("الجمل", analysis['sentence_count'])
                
                with col2:
                    st.metric("الكلمات الإيجابية", analysis['positive_words'])
                    st.metric("الكلمات السلبية", analysis['negative_words'])
                
                with col3:
                    st.markdown(f"**الفئة المتوقعة:** {analysis['category']}")
                    st.markdown(f"**المشاعر:** {analysis['sentiment']}")
                
                if show_word_freq and analysis['most_common_words']:
                    st.subheader("أكثر الكلمات تكراراً")
                    freq_df = pd.DataFrame(analysis['most_common_words'], columns=['الكلمة', 'التكرار'])
                    st.bar_chart(freq_df.set_index('الكلمة'))
            
            with tab2:
                st.subheader("🏷️ تصنيف النص")
                st.success(f"**الفئة المتوقعة:** {analysis['category']}")
                confidence = analysis['category_confidence'] * 100
                st.progress(min(confidence, 100) / 100)
                st.write(f"مستوى الثقة: {confidence:.1f}%")
            
            with tab3:
                st.subheader("📄 ملخص النص")
                summary = simple_summarize(user_text)
                st.write(summary)
                
                original_words = analysis['word_count']
                summary_words = len(summary.split())
                compression = (1 - summary_words/original_words) * 100 if original_words > 0 else 0
                
                st.info(f"تم ضغط النص من {original_words} كلمة إلى {summary_words} كلمة ({compression:.1f}% ضغط)")
            
            with tab4:
                st.subheader("🔍 الكيانات المستخرجة")
                entities = extract_entities(user_text)
                
                if entities:
                    entities_df = pd.DataFrame(entities)
                    st.dataframe(entities_df)
                    
                    # إحصائيات الكيانات
                    entity_counts = pd.Series([e['label'] for e in entities]).value_counts()
                    st.bar_chart(entity_counts)
                else:
                    st.info("لم يتم العثور على كيانات واضحة في النص")
        
        elif analysis_type == "تحليل المشاعر":
            st.header("😊 تحليل المشاعر")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_emoji = {
                    'إيجابي': '😊',
                    'سلبي': '😞',
                    'محايد': '😐'
                }
                
                st.markdown(f"## {sentiment_emoji[analysis['sentiment']]} {analysis['sentiment']}")
                
                st.metric("كلمات إيجابية", analysis['positive_words'])
                st.metric("كلمات سلبية", analysis['negative_words'])
            
            with col2:
                # مخطط بسيط للمشاعر
                sentiment_data = pd.DataFrame({
                    'النوع': ['إيجابي', 'سلبي'],
                    'العدد': [analysis['positive_words'], analysis['negative_words']]
                })
                st.bar_chart(sentiment_data.set_index('النوع'))
        
        elif analysis_type == "تصنيف النص":
            st.header("🏷️ تصنيف النص")
            
            st.success(f"**الفئة المتوقعة:** {analysis['category']}")
            
            confidence = analysis['category_confidence'] * 100
            st.progress(min(confidence, 100) / 100)
            st.write(f"مستوى الثقة: {confidence:.1f}%")
        
        elif analysis_type == "تلخيص النص":
            st.header("📄 تلخيص النص")
            
            summary = simple_summarize(user_text)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### الملخص:")
                st.write(summary)
            
            with col2:
                original_words = analysis['word_count']
                summary_words = len(summary.split())
                compression = (1 - summary_words/original_words) * 100 if original_words > 0 else 0
                
                st.metric("الطول الأصلي", f"{original_words} كلمة")
                st.metric("طول الملخص", f"{summary_words} كلمة")
                st.metric("نسبة الضغط", f"{compression:.1f}%")
        
        elif analysis_type == "استخراج الكيانات":
            st.header("🔍 استخراج الكيانات")
            
            entities = extract_entities(user_text)
            
            if entities:
                st.success(f"تم العثور على {len(entities)} كيان")
                
                entities_df = pd.DataFrame(entities)
                st.dataframe(entities_df)
                
                # توزيع الكيانات
                if len(entities) > 1:
                    entity_counts = pd.Series([e['label'] for e in entities]).value_counts()
                    st.subheader("توزيع أنواع الكيانات")
                    st.bar_chart(entity_counts)
            else:
                st.info("لم يتم العثور على كيانات واضحة في النص")

# معلومات إضافية
if user_text:
    st.markdown("---")
    st.markdown("### 📈 إحصائيات متقدمة")
    
    with st.expander("عرض التفاصيل"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_word_length = sum(len(word) for word in user_text.split()) / len(user_text.split()) if user_text.split() else 0
            st.metric("متوسط طول الكلمة", f"{avg_word_length:.1f}")
        
        with col2:
            avg_sentence_length = analysis['word_count'] / analysis['sentence_count'] if analysis['sentence_count'] > 0 else 0
            st.metric("متوسط طول الجملة", f"{avg_sentence_length:.1f}")
        
        with col3:
            unique_words = len(set(user_text.split()))
            diversity = unique_words / analysis['word_count'] * 100 if analysis['word_count'] > 0 else 0
            st.metric("تنوع المفردات", f"{diversity:.1f}%")

st.markdown("---")
st.markdown("### 💡 نصائح")
with st.expander("نصائح لتحسين التحليل"):
    st.markdown("""
    - **للنصوص العربية:** تأكد من استخدام النص الصحيح
    - **للتصنيف:** استخدم نصوص واضحة تحتوي على كلمات مفتاحية
    - **للتلخيص:** النصوص الأطول تعطي ملخصات أفضل
    - **للكيانات:** النصوص التي تحتوي على أسماء وأرقام تعطي نتائج أوضح
    - **هذا إصدار تجريبي:** سيتم تحسين الأداء في الإصدارات القادمة
    """)

st.markdown("---")
st.markdown("**📧 للتواصل والدعم الفني:** [اتصل بنا](mailto:support@example.com)")
