# 📁 מבנה פרויקט MultiPromptify - מדריך מפורט

## 🎯 סקירה כללית
MultiPromptify הוא כלי ליצירת וריאציות של prompts מתבניות יחידות. הפרויקט מאורגן במבנה מודולרי עם הפרדה ברורה בין הספרייה המרכזית, ממשק המשתמש, והקבצים המשותפים.

---

## 📂 src/ - ספריית הקוד המרכזית

### 📚 src/multipromptify/ - ספריית הליבה הראשית

#### **קבצים מרכזיים:**
- **`__init__.py`** - נקודת כניסה לספרייה, מייצא את המחלקות הראשיות למשתמש חיצוני
- **`multipromptify.py`** - המנוע הראשי של הספרייה, מתאם את כל תהליך יצירת הוריאציות
- **`template_parser.py`** - מנתח ומוודא תקינות של תבניות המשתמש עם כללי הוריאציות
- **`cli.py`** - ממשק שורת פקודה לשימוש בספרייה ללא ממשק גרפי

#### 🔧 src/multipromptify/augmentations/ - מנועי ההרחבה

**📁 קבצי בסיס:**
- **`__init__.py`** - מייצא את כל ה-augmenters לשימוש קל
- **`base.py`** - מחלקת בסיס המגדירה ממשק אחיד לכל מנועי ההרחבה
- **`pipeline.py`** - מתאם הפעלה של מספר augmenters ברצף
- **`other.py`** - מנועי הרחבה כלליים נוספים

**📁 text/ - הרחבות טקסטואליות:**
- **`surface.py`** - שינויים משטחיים (אותיות גדולות/קטנות, פיסוק, רווחים)
- **`paraphrase.py`** - יצירת ניסוחים חלופיים של הטקסט באמצעות AI
- **`context.py`** - הוספת קונטקסט רלוונטי לפני או אחרי הטקסט המקורי

**📁 structure/ - הרחבות מבניות:**
- **`fewshot.py`** - יצירת דוגמאות few-shot מהנתונים הקיימים
- **`shuffle.py`** - ערבוב סדר אפשרויות בשאלות בחירה

---

### 🖥️ src/ui/ - ממשק המשתמש (Streamlit)

#### **קבצים מרכזיים:**
- **`main.py`** - נקודת כניסה להפעלת אפליקציית Streamlit
- **`__init__.py`** - אתחול חבילת ממשק המשתמש

#### **📁 pages/ - דפי האפליקציה:**
- **`load.py`** - עמוד ראשי ומתאם זרימת העבודה הכללית
- **`upload_data.py`** - עמוד העלאת קבצי נתונים (CSV, JSON)
- **`template_builder.py`** - עמוד בניית תבניות אינטראקטיבי עם תצוגה מקדימה
- **`generate_variations.py`** - עמוד יצירת וריאציות והצגת התוצאות
- **`results_display.py`** - פונקציות להצגה מעוצבת של תוצאות הוריאציות



#### **📁 utils/ - עזרים לממשק:**
- **`debug_helpers.py`** - כלים לדיבוג ולטעינת נתוני דמו
- **`progress_indicator.py`** - מציג סטטוס התקדמות בממשק
- **`upload_csv.py`** - עזרים להעלאת ועיבוד קבצי CSV
- **`map_csv_to_json.py`** - המרה בין פורמטי נתונים שונים

---

### 🔗 src/shared/ - קבצים משותפים

- **`constants.py`** - הגדרות קבועות ופרמטרים למערכת כולה
- **`model_client.py`** - התחברות וניהול קריאות למודלים של AI (Together API)
- **`benchmark_loader.py`** - טעינת נתוני benchmark להערכת ביצועים
- **`__init__.py`** - אתחול חבילת הקבצים המשותפים

---

## 🧪 tests/ - בדיקות וחומרי עזר

### **📁 integration/ - בדיקות אינטגרציה:**
- **`test_multipromptify.py`** - בדיקות מקיפות לתפקוד המערכת כולה

### **📁 examples/ - דוגמאות שימוש:**
- **`usage_examples.py`** - דוגמאות קוד ותרחישי שימוש שונים
- **`sample_data.csv`** - נתוני דוגמה לניסויים וחדמושים

---

## 🔄 זרימת עבודה טיפוסית

1. **העלאת נתונים** (`upload_data.py`) → העלאת CSV/JSON
2. **בניית תבנית** (`template_builder.py`) → הגדרת כללי הרחבה
3. **יצירת וריאציות** (`generate_variations.py`) → הפעלת המנועים
4. **הצגת תוצאות** (`results_display.py`) → תצוגה ויצוא

---

## 🎯 נקודות כניסה למערכת

- **🖥️ ממשק גרפי:** `python src/ui/main.py`
- **⚡ שורת פקודה:** `python -m multipromptify.cli`
- **📚 ספרייה:** `from multipromptify import MultiPromptify`

---

## 🛠️ טיפים לתחזוקה

- **הוספת augmenter חדש:** יצור קובץ חדש ב-`augmentations/` וירש מ-`BaseAxisAugmenter`
- **עדכון UI:** ערוך קבצים ב-`ui/pages/` או `ui/components/`
- **שינוי הגדרות:** ערוך `shared/constants.py`
- **בדיקות:** הוסף ל-`tests/integration/` או `tests/examples/`

---

## 📋 רשימת קבצים מלאה

```
src/
├── __init__.py
├── multipromptify/
│   ├── __init__.py
│   ├── multipromptify.py (28KB, 640 שורות)
│   ├── template_parser.py (9KB, 233 שורות)
│   ├── cli.py (10KB, 326 שורות)
│   └── augmentations/
│       ├── __init__.py
│       ├── base.py
│       ├── pipeline.py
│       ├── other.py
│       ├── text/
│       │   ├── __init__.py
│       │   ├── surface.py
│       │   ├── paraphrase.py
│       │   └── context.py
│       └── structure/
│           ├── __init__.py
│           ├── fewshot.py
│           └── shuffle.py
├── ui/
│   ├── __init__.py
│   ├── main.py (1.5KB, 43 שורות)
│   ├── pages/
│   │   ├── load.py
│   │   ├── upload_data.py
│   │   ├── template_builder.py
│   │   ├── generate_variations.py
│   │   └── results_display.py

│   └── utils/
│       ├── __init__.py
│       ├── debug_helpers.py
│       ├── progress_indicator.py
│       ├── upload_csv.py
│       └── map_csv_to_json.py
└── shared/
    ├── __init__.py
    ├── constants.py
    ├── model_client.py
    └── benchmark_loader.py

tests/
├── integration/
│   └── test_multipromptify.py
└── examples/
    ├── usage_examples.py
    └── sample_data.csv
```

---

## 📝 הערות חשובות

1. **ייבוא מודולים:** כל הייבואים במערכת משתמשים בנתיבים אבסולוטיים החל מ-`src`
2. **מודולריות:** כל augmenter הוא עצמאי וניתן להוסיף חדשים בקלות
3. **ממשק אחיד:** כל הקומפוננטים מממשים ממשקים סטנדרטיים
4. **קלות תחזוקה:** המבנה מאפשר שינויים מקומיים ללא השפעה על רכיבים אחרים

המבנה מאפשר הרחבה קלה ותחזוקה נוחה עם הפרדה ברורה בין אחריויות! 🚀 