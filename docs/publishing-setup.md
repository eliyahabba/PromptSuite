# MultiPromptify Trusted Publishers Setup Guide

## 📋 Task Checklist

### ✅ What's Already Done (prepared for you):
- [x] GitHub Workflow file (`.github/workflows/publish.yml`)
- [x] pyproject.toml configured with setuptools_scm
- [x] Proper project structure
- [x] Package tested and working

### 🎯 What You Need to Do:

## Step 1: Upload to GitHub

```bash
# Add all new files
git add .
git commit -m "Add GitHub Actions publishing workflow"
git push origin main
```

## Step 2: Configure GitHub Environments

1. Go to your GitHub repository: `https://github.com/ehabba/MultiPromptifyPipeline`
2. Click on **Settings** (tab at the top)
3. In the left menu, click on **Environments**
4. Click on **New environment**

### Create Environment for TestPyPI:
- **Name**: `testpypi`
- **Protection rules**: 
  - ☑️ Required reviewers (if you want approval before publishing)
  - ☑️ Restrict deployment branches → `Selected branches` → `main`

### Create Environment for PyPI:
- **Name**: `pypi`  
- **Protection rules**:
  - ☑️ Required reviewers (recommended!)
  - ☑️ Restrict deployment branches → `Selected branches` → `main`

## Step 3: Configure Trusted Publisher on TestPyPI

1. Go to **TestPyPI**: https://test.pypi.org/
2. Login / Register for an account
3. Go to **Account settings** → **Publishing**
4. Click on **GitHub** tab
5. Click on **Add a new pending publisher**

### Fill in the details:
- **PyPI Project Name**: `multipromptify`
- **Owner**: `ehabba`
- **Repository name**: `MultiPromptifyPipeline`
- **Workflow name**: `publish.yml`
- **Environment name**: `testpypi`

## Step 4: Configure Trusted Publisher on PyPI

1. Go to **PyPI**: https://pypi.org/
2. Login / Register for an account (if you don't have one)
3. Go to **Account settings** → **Publishing**
4. Click on **GitHub** tab
5. Click on **Add a new pending publisher**

### Fill in the details:
- **PyPI Project Name**: `multipromptify`
- **Owner**: `ehabba`
- **Repository name**: `MultiPromptifyPipeline`
- **Workflow name**: `publish.yml`
- **Environment name**: `pypi`

## Step 5: Test Everything Works

### Option 1: Publish with Tag
```bash
# Create a new tag
git tag v2.0.0
git push origin v2.0.0
```

### Option 2: Manual Trigger
1. Go to GitHub → **Actions**
2. Click on **Publish MultiPromptify Package**
3. Click on **Run workflow**
4. Select branch `main`
5. Click **Run workflow**

## 🔍 What Will Happen When You Run:

1. **Tests** - Tests the package works on all Python versions
2. **Build** - Builds and validates the package
3. **TestPyPI** - Publishes to TestPyPI (for testing)
4. **PyPI** - Publishes to official PyPI
5. **GitHub Release** - Creates automatic GitHub Release

## 📦 How to Verify It Worked:

### After publishing to TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ multipromptify
```

### After publishing to PyPI:
```bash
pip install multipromptify
```

## 🚨 Common Issues and Solutions:

### "No matching distribution found"
- Check that workflow name is exactly `publish.yml`
- Verify environment names are correct (`testpypi`, `pypi`)

### "Permission denied" 
- Verify you configured Trusted Publishers correctly
- Check that repo and owner are correct

### "Project name already exists"
- Is the name `multipromptify` taken? Change to `ehabba-multipromptify`
- Or choose a different name

### Workflow doesn't run
- Verify there's a commit with the `.github/workflows/publish.yml` file
- Check in Actions that the workflow appears

## 🎉 Success!

When this works, people will be able to install:
```bash
pip install multipromptify
```

And you'll be able to publish new versions simply with:
```bash
git tag v2.0.1
git push origin v2.0.1
```

## 🔄 Version Updates in the Future

Version will be determined automatically from git tags thanks to setuptools_scm:
- `v2.0.0` → version `2.0.0`
- `v2.1.0` → version `2.1.0`
- Additional commits → `2.1.0.dev1`, `2.1.0.dev2` etc.

## 📞 Need Help?

If something doesn't work, let me know where you're stuck and I'll help! 