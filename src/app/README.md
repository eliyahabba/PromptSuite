# MultiPromptify UI Components

This directory contains the user interface components for the MultiPromptify 2.0 application.

## Page Structure

The application now has a **streamlined 3-page workflow**:

### 📁 **Page 1: Upload Data** (`upload_data.py`)
- Upload CSV files, JSON files, or use sample datasets
- Validate data format and preview contents
- Create custom data if needed

### 🔧 **Page 2: Build Template** (`template_builder.py`) 
- Choose from template suggestions or build custom templates
- Configure field variations (surface, paraphrase, etc.)
- Set up instruction templates and few-shot examples
- Validate template against uploaded data

### ⚡ **Page 3: Generate Variations** (`generate_variations.py`)
- Configure generation settings (max variations, rows to use, etc.)
- Generate prompt variations using the template
- **View complete results immediately** with full display functionality
- Export variations in multiple formats (JSON, CSV, TXT)

## Components

### `/components/` - Shared UI Components
- **`results_display.py`** - Comprehensive results display functionality
- **`__init__.py`** - Package initialization with exports

### `/utils/` - Utility Modules
- **`progress_indicator.py`** - Progress bar and step indicators
- **`debug_helpers.py`** - Development and debugging utilities

## Key Features

### ✅ **Streamlined User Experience**
- **No redundant pages** - Users see complete results immediately after generation
- **Progressive disclosure** - Each step builds on the previous one
- **Consistent styling** - Shared components ensure uniform look and feel

### 🔧 **Modular Architecture**
- **Reusable components** - Display logic shared between pages
- **Clean separation** - Each page has a specific responsibility
- **Easy maintenance** - Centralized functionality in `/components/`

### 📊 **Enhanced Results Display**
- **Immediate feedback** - Full results shown on generation page
- **Rich visualization** - Highlighted field changes and prompt structure
- **Multiple export formats** - JSON, CSV, and plain text options
- **Pagination** - Handle large result sets efficiently

## Recent Changes

**🗑️ Removed Page 4 (Show Results)**
- Page 4 was redundant since Page 3 now shows complete results
- Simplified navigation from 4 pages to 3 pages  
- Removed unnecessary "Continue to Results Page" button
- Updated progress indicators and navigation logic

## File Status

- ✅ **Active Files**: `upload_data.py`, `template_builder.py`, `generate_variations.py`
- 📦 **Shared Components**: `components/results_display.py`
- 🔧 **Utilities**: `utils/progress_indicator.py`, `utils/debug_helpers.py`
- 🏠 **Main Entry**: `load.py` (updated for 3-page navigation)
- ⚠️ **Legacy**: `show_results.py` (kept for reference but not used in navigation)

## Usage

Run the application with:

```bash
cd src/ui
python load.py --step=1 --debug=False
```

Or use the streamlit runner:

```bash
streamlit run src/ui/run_streamlit.py
```

The application will guide users through the 3-step process with automatic validation and navigation between steps.

## 🚀 Quick Start

### Installation

```bash
# Install with UI support
pip install -e ".[ui]"

# Or install requirements directly
pip install streamlit>=1.28.0
```

### Launch the UI

```bash
# From project root
python src/ui/run_streamlit.py

# Or use the demo script
python demo_ui.py
```

The interface will open in your browser at `http://localhost:8501`

## 📋 Interface Overview

### Step 1: Upload Data 📁
- **Upload Files**: Support for CSV and JSON formats
- **Sample Datasets**: Pre-built datasets for testing (Sentiment Analysis, Q&A, Multiple Choice, etc.)
- **Custom Data**: Create datasets manually with the built-in editor
- **Data Preview**: View data structure and column information

### Step 2: Template Builder 🔧
- **Template Suggestions**: Smart suggestions based on your data columns
- **Custom Templates**: Build templates with Python f-string syntax
- **Variation Types**: Support for 6 variation types (semantic, paraphrase, non-semantic, lexical, syntactic, surface)
- **Real-time Validation**: Instant feedback on template syntax and data compatibility
- **Live Preview**: Test templates with sample data before generation

### Step 3: Generate Variations ⚡
- **Generation Settings**: Configure max variations, variations per field, random seeds
- **Few-shot Examples**: Add examples to prompts with flexible input methods
- **Progress Tracking**: Real-time progress bars and status updates
- **Estimation**: Preview how many variations will be generated
- **Quick Download**: Immediate export options for generated results

### Step 4: View Results 🎉
- **Comprehensive Analysis**: Statistics, distribution charts, field analysis
- **Search & Filter**: Advanced filtering by content, row, length, and field values
- **Multiple Views**: All variations, analysis dashboard, search interface, export options
- **Export Formats**: JSON, CSV, TXT, and custom format templates

## 🎯 Key Features

### Smart Template Suggestions
- Automatically detects data patterns
- Suggests compatible templates based on column names
- Shows which fields are available vs. missing
- Provides usage examples for each suggestion

### Template Validation
- Real-time syntax checking
- Column availability verification
- Variation type validation
- Clear error messages and suggestions

### Sample Datasets
The UI includes several built-in sample datasets:

1. **Sentiment Analysis** - Text classification with sentiment labels
2. **Question Answering** - Q&A pairs with context
3. **Multiple Choice** - MCQ with options and subjects
4. **Text Classification** - Intent and category classification

### Advanced Features
- **Pagination**: Handle large result sets efficiently
- **Search Highlighting**: Visual search result highlighting
- **Export Customization**: Define custom export templates
- **Responsive Design**: Works on desktop and mobile devices
- **Progress Tracking**: Real-time feedback during generation

## 🔧 Technical Details

### Architecture
The UI is built with Streamlit and consists of 4 main components:

1. `upload_data.py` - Data loading and sample dataset management
2. `template_builder.py` - Template creation and validation interface
3. `generate_variations.py` - Variation generation with progress tracking
4. `show_results.py` - Results display, analysis, and export

### Session State Management
- Persistent data across steps
- Navigation state tracking
- Progress indicators
- Error handling and recovery

### Integration with MultiPromptify Core
- Direct integration with `MultiPromptify` class
- Real-time template parsing and validation
- Efficient variation generation
- Comprehensive statistics and metadata

## 🎨 Customization

### Styling
The UI uses custom CSS for improved appearance:
- Gradient headers for each step
- Info boxes for guidance
- Responsive column layouts
- Modern color scheme

### Template Suggestions
You can modify template suggestions in `load.py`:

```python
'template_suggestions': [
    {
        'name': 'Your Template Name',
        'template': '{instruction:semantic}: {your_field:paraphrase}',
        'description': 'Description of your template',
        'sample_data': {
            'your_field': ['example1', 'example2']
        }
    }
]
```

## 🚦 Usage Examples

### Basic Workflow
1. Start the UI: `python src/ui/run_streamlit.py`
2. Upload data or select a sample dataset
3. Choose a template suggestion or create a custom one
4. Configure generation settings and run
5. Explore results and export in your preferred format

### Advanced Usage
- Use URL parameters to start at specific steps: `?step=2`
- Enable debug mode for development: `?debug=true`
- Combine parameters: `?step=3&debug=true`

## 🐛 Troubleshooting

### Common Issues

**"Please upload data first"**
- Make sure you've completed Step 1 before proceeding
- Check that your data was successfully loaded

**"Template validation errors"**
- Verify your template syntax uses correct `{field:variation_type}` format
- Ensure all referenced fields exist in your data
- Check that variation types are supported

**"No variations generated"**
- Verify your template is valid
- Check that you have data loaded
- Try reducing max_variations if memory issues occur

### Performance Tips
- Use sample datasets for testing before processing large files
- Limit max_variations for initial testing
- Use pagination in results view for large datasets

## 📚 Further Reading

- [MultiPromptify Core Documentation](../../README.md)
- [Template Syntax Guide](../../docs/templates.md)
- [API Reference](../../docs/api.md)
- [CLI Usage](../../docs/cli.md) 