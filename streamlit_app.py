import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import io
import base64
from PIL import Image
import json
import os

# استيراد الوحدات المخصصة
try:
    from utils.text_classifier import NewsClassifier
    from utils.text_summarizer import TextSummarizer
    from utils.entity_extractor import EntityExtractor
except ImportError as e:
    st.error(f"خطأ في استيراد الوحدات: {e}")
    st.stop()

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

# تهيئة الجلسة
@st.cache_resource
def initialize_models():
    """تحميل جميع النماذج"""
    try:
        classifier = NewsClassifier()
        summarizer = TextSummarizer()
        extractor = EntityExtractor()
        return classifier, summarizer, extractor
    except Exception as e:
        st.error(f"فشل في تحميل النماذج: {e}")
        return None, None, None

# دالة لإنشاء المخططات باستخدام Streamlit المدمج
def create_streamlit_chart(data, chart_type="bar", title=""):
    """إنشاء مخطط باستخدام مخططات Streamlit المدمجة"""
    st.subheader(title)
    
    if isinstance(data, dict):
        data = pd.DataFrame(list(data.items()), columns=['Category', 'Count'])
    
    if chart_type == "bar":
        st.bar_chart(data.set_index('Category') if 'Category' in data.columns else data)
    elif chart_type == "line":
        st.line_chart(data.set_index('Category') if 'Category' in data.columns else data)
    elif chart_type == "area":
        st.area_chart(data.set_index('Category') if 'Category' in data.columns else data)
    else:
        # Default to bar chart
        st.bar_chart(data.set_index('Category') if 'Category' in data.columns else data)

# دالة لعرض WordCloud
def display_wordcloud(text, title="سحابة الكلمات"):
    """عرض سحابة الكلمات"""
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            font_path=None,  # قد تحتاج لتعديل هذا للنصوص العربية
            max_words=100
        ).generate(text)
        
        # تحويل إلى صورة
        img = wordcloud.to_image()
        st.subheader(title)
        st.image(img, use_column_width=True)
        
    except Exception as e:
        st.error(f"خطأ في إنشاء سحابة الكلمات: {e}")

# تحميل النماذج
classifier, summarizer, extractor = initialize_models()

# العنوان الرئيسي
st.markdown('<h1 class="main-header">📰 محلل الأخبار الذكي</h1>', unsafe_allow_html=True)
st.markdown("---")

# الشريط الجانبي
st.sidebar.title("🎛️ لوحة التحكم")
st.sidebar.markdown("---")

# اختيار نوع التحليل
analysis_type = st.sidebar.selectbox(
    "نوع التحليل",
    ["تصنيف النصوص", "تلخيص النصوص", "استخراج الكيانات", "تحليل شامل"]
)

# خيارات إضافية
st.sidebar.markdown("### ⚙️ الإعدادات")
show_confidence = st.sidebar.checkbox("عرض مستوى الثقة", True)
show_stats = st.sidebar.checkbox("عرض الإحصائيات", True)
show_visualizations = st.sidebar.checkbox("عرض المخططات", True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 حول التطبيق")
st.sidebar.info(
    """
    **المميزات:**
    - تصنيف الأخبار إلى فئات مختلفة
    - تلخيص النصوص الطويلة
    - استخراج الكيانات المسماة
    - تحليل شامل للنصوص
    - سحابة الكلمات
    - إحصائيات تفصيلية
    """
)

# المحتوى الرئيسي
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 إدخال النص")
    
    # خيارات إدخال النص
    input_method = st.radio(
        "طريقة الإدخال:",
        ["كتابة مباشرة", "رفع ملف نصي", "URL"]
    )
    
    user_text = ""
    
    if input_method == "كتابة مباشرة":
        user_text = st.text_area(
            "أدخل النص للتحليل:",
            height=300,
            placeholder="اكتب أو الصق النص هنا..."
        )
    
    elif input_method == "رفع ملف نصي":
        uploaded_file = st.file_uploader(
            "اختر ملف نصي",
            type=['txt', 'docx', 'pdf']
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "text/plain":
                    user_text = str(uploaded_file.read(), "utf-8")
                else:
                    st.warning("يتم دعم ملفات .txt فقط حاليا")
            except Exception as e:
                st.error(f"خطأ في قراءة الملف: {e}")
    
    elif input_method == "URL":
        url = st.text_input("أدخل رابط المقال:")
        if url and st.button("استخراج النص من URL"):
            try:
                # هنا يمكن إضافة كود لاستخراج النص من URL
                st.info("هذه الميزة قيد التطوير")
            except Exception as e:
                st.error(f"خطأ في استخراج النص: {e}")

with col2:
    st.header("📊 المقاييس السريعة")
    
    if user_text:
        # إحصائيات أساسية
        word_count = len(user_text.split())
        char_count = len(user_text)
        sentence_count = len([s for s in user_text.split('.') if s.strip()])
        
        st.metric("عدد الكلمات", word_count)
        st.metric("عدد الأحرف", char_count)
        st.metric("عدد الجمل", sentence_count)
        
        # تقييم سهولة القراءة (تقريبي)
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        if avg_words_per_sentence < 15:
            readability = "سهل"
            color = "green"
        elif avg_words_per_sentence < 25:
            readability = "متوسط"
            color = "orange"
        else:
            readability = "صعب"
            color = "red"
        
        st.markdown(f"**سهولة القراءة:** <span style='color: {color}'>{readability}</span>", 
                   unsafe_allow_html=True)

# معالجة التحليل
if user_text and st.button("🚀 بدء التحليل", type="primary"):
    
    with st.spinner("جاري التحليل..."):
        
        if analysis_type == "تصنيف النصوص":
            st.header("🏷️ نتائج التصنيف")
            
            if classifier:
                try:
                    result = classifier.classify(user_text)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(f"**الفئة المتوقعة:** {result.get('category', 'غير محدد')}")
                        
                        if show_confidence and 'confidence' in result:
                            confidence = result['confidence']
                            st.progress(confidence)
                            st.write(f"مستوى الثقة: {confidence:.2%}")
                    
                    with col2:
                        if show_visualizations and 'probabilities' in result:
                            prob_data = pd.DataFrame(
                                list(result['probabilities'].items()),
                                columns=['Category', 'Probability']
                            )
                            st.subheader("توزيع الاحتمالات")
                            st.bar_chart(prob_data.set_index('Category'))
                            
                except Exception as e:
                    st.error(f"خطأ في التصنيف: {e}")
            else:
                st.error("نموذج التصنيف غير متاح")
        
        elif analysis_type == "تلخيص النصوص":
            st.header("📄 ملخص النص")
            
            if summarizer:
                try:
                    summary = summarizer.summarize(user_text)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("### الملخص:")
                        st.write(summary.get('summary', 'لم يتم إنشاء ملخص'))
                    
                    with col2:
                        if show_stats:
                            original_length = len(user_text.split())
                            summary_length = len(summary.get('summary', '').split())
                            compression_ratio = (1 - summary_length/original_length) * 100 if original_length > 0 else 0
                            
                            st.metric("الطول الأصلي", f"{original_length} كلمة")
                            st.metric("طول الملخص", f"{summary_length} كلمة")
                            st.metric("نسبة الضغط", f"{compression_ratio:.1f}%")
                            
                except Exception as e:
                    st.error(f"خطأ في التلخيص: {e}")
            else:
                st.error("نموذج التلخيص غير متاح")
        
        elif analysis_type == "استخراج الكيانات":
            st.header("🔍 الكيانات المستخرجة")
            
            if extractor:
                try:
                    entities = extractor.extract(user_text)
                    
                    if entities:
                        # عرض الكيانات في جدول
                        entities_df = pd.DataFrame(entities)
                        st.dataframe(entities_df)
                        
                        if show_visualizations:
                            # إحصائيات الكيانات
                            entity_counts = entities_df['label'].value_counts()
                            st.subheader("توزيع أنواع الكيانات")
                            st.bar_chart(entity_counts)
                    else:
                        st.info("لم يتم العثور على كيانات في النص")
                        
                except Exception as e:
                    st.error(f"خطأ في استخراج الكيانات: {e}")
            else:
                st.error("نموذج استخراج الكيانات غير متاح")
        
        elif analysis_type == "تحليل شامل":
            st.header("🔍 التحليل الشامل")
            
            # تشغيل جميع أنواع التحليل
            tabs = st.tabs(["التصنيف", "التلخيص", "الكيانات", "إحصائيات إضافية"])
            
            with tabs[0]:
                if classifier:
                    try:
                        result = classifier.classify(user_text)
                        st.success(f"**الفئة:** {result.get('category', 'غير محدد')}")
                        if 'confidence' in result:
                            st.progress(result['confidence'])
                    except Exception as e:
                        st.error(f"خطأ في التصنيف: {e}")
            
            with tabs[1]:
                if summarizer:
                    try:
                        summary = summarizer.summarize(user_text)
                        st.write(summary.get('summary', 'لم يتم إنشاء ملخص'))
                    except Exception as e:
                        st.error(f"خطأ في التلخيص: {e}")
            
            with tabs[2]:
                if extractor:
                    try:
                        entities = extractor.extract(user_text)
                        if entities:
                            st.dataframe(pd.DataFrame(entities))
                        else:
                            st.info("لم يتم العثور على كيانات")
                    except Exception as e:
                        st.error(f"خطأ في استخراج الكيانات: {e}")
            
            with tabs[3]:
                # إحصائيات إضافية
                words = user_text.split()
                word_freq = pd.Series(words).value_counts().head(10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("أكثر الكلمات تكراراً")
                    st.bar_chart(word_freq)
                
                with col2:
                    if show_visualizations:
                        display_wordcloud(user_text)

# معلومات إضافية
st.markdown("---")
st.markdown("### 💡 نصائح للاستخدام")
with st.expander("اضغط لعرض النصائح"):
    st.markdown("""
    - **للحصول على أفضل النتائج:** استخدم نصوص واضحة ومكتملة
    - **التصنيف:** يعمل بشكل أفضل مع المقالات الإخبارية الكاملة
    - **التلخيص:** مناسب للنصوص الطويلة (أكثر من 100 كلمة)
    - **استخراج الكيانات:** يتعرف على الأشخاص، الأماكن، والمنظمات
    - **الأداء:** قد يستغرق التحليل بضع ثوان حسب طول النص
    """)

st.markdown("---")
st.markdown("**تم تطويره بواسطة:** فريق تطوير محلل الأخبار الذكي")
