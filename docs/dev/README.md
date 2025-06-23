# 🏗️ Developer Documentation

This directory contains technical documentation for developers working on MultiPromptify.

## 📋 Documentation Files

### 🔧 Implementation Details
- **[Implementation Summary](implementation-summary.md)** - Core 2.0 implementation overview
- **[UI Implementation Summary](ui-implementation-summary.md)** - Web interface implementation details

### 📁 Project Organization  
- **[Project Structure](project-structure.md)** - Complete codebase organization guide
- **[Renaming Guide](renaming-guide.md)** - Future package renaming strategy

### 📦 Publishing & Distribution
- **[Publishing Guide](publishing-guide.md)** - PyPI package publishing instructions

## 🎯 Quick Navigation

### For New Contributors
1. Start with [Project Structure](project-structure.md) to understand the codebase
2. Review [Implementation Summary](implementation-summary.md) for technical overview
3. Check [UI Implementation](ui-implementation-summary.md) for web interface details

### For Maintainers
- [Publishing Guide](publishing-guide.md) - Package publishing workflow
- [Renaming Guide](renaming-guide.md) - Future package renaming strategy

## 📚 Related Documentation

- **[User Documentation](../README.md)** - Main project documentation
- **[API Documentation](../api-guide.md)** - Python API reference

## Augmenters and Variation Types

MultiPromptify supports a variety of augmenters for prompt variation:
- `format_structure` (`FORMAT_STRUCTURE_VARIATION`): Semantic-preserving format changes (separators, casing, field order)
- `typos and noise` (`TYPOS_AND_NOISE_VARIATION`): Injects typos, random case, whitespace, and punctuation noise
- `enumerate` (`ENUMERATE_VARIATION`): Adds enumeration to list fields (1. 2. 3. 4., A. B. C. D., roman, etc.)
- `paraphrase_with_llm`, `context`, `shuffle`, `multidoc`, and more

See the main README and API guide for template examples using these augmenters.

---

*This documentation is maintained by the MultiPromptify development team.* 