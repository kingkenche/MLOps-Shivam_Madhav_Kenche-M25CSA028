# GitHub Upload Instructions (Without Git CLI)

## ⚠️ Git CLI Not Available

Since Git command line is not available, please use one of these methods:

---

## Method 1: GitHub Desktop (Recommended)

1. **Download GitHub Desktop** (if not installed):
   - Visit: https://desktop.github.com/
   - Download and install

2. **Add Repository:**
   - Open GitHub Desktop
   - File → Add Local Repository
   - Choose: `G:\Ass-1\GitHub_Submission\Assignment1`
   - If prompted, click "create a repository"

3. **Configure:**
   - Repository → Repository Settings
   - Set name: `MLOps-Shivam_Madhav_Kenche-M25CSA028`
   - Set remote URL: `https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028.git`

4. **Create Branch:**
   - Branch → New Branch → Name: `Assignment-1`

5. **Commit Changes:**
   - Review all 19 files in the changes tab
   - Summary: "Add Assignment 1: Q1(a) ResNet and Q1(b) SVM Classification - Complete Submission"
   - Click "Commit to Assignment-1"

6. **Publish:**
   - Click "Publish branch" or "Push origin"

---

## Method 2: Direct Web Upload

1. **Go to GitHub:**
   - https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028

2. **Create Branch:**
   - Click branch dropdown → Type "Assignment-1" → Create branch

3. **Upload Files:**
   - Click "Add file" → "Upload files"
   - **Drag ALL folders from:** `G:\Ass-1\GitHub_Submission\Assignment1`
   - Keep folder structure intact

4. **Commit:**
   - Commit message: "Add Assignment 1: Q1(a) ResNet and Q1(b) SVM Classification - Complete Submission"
   - Choose "Commit directly to the Assignment-1 branch"
   - Click "Commit changes"

---

## Method 3: Install Git and Use CLI

1. **Download Git:**
   - Visit: https://git-scm.com/download/win
   - Install with default settings
   - Restart PowerShell/Terminal

2. **Then run these commands:**

```bash
cd G:\Ass-1\GitHub_Submission\Assignment1
git init
git config user.name "SHIVAM MADHAV KENCHE"
git config user.email "your-email@example.com"
git remote add origin https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028.git
git checkout -b Assignment-1
git add .
git commit -m "Add Assignment 1: Q1(a) ResNet and Q1(b) SVM Classification - Complete Submission"
git push -u origin Assignment-1
```

---

## ⚠️ Important Notes

### Large Files Warning
The `.pth` model files are ~45MB each (90MB total). GitHub has:
- File size limit: 100MB per file ✅ (We're under this)
- Repository size warning: 1GB
- Push size limit: 2GB

**Your submission is 90.16MB total - well within limits!**

### Files to Upload (19 files):
```
Assignment1/
├── README.md
├── M25CSA028_SHIVAM_MADHAV_KENCHE_Ass1_Report.md
├── M25CSA028_SHIVAM_MADHAV_KENCHE_Ass1_Report.pdf
├── index.html
├── SUBMISSION_CHECKLIST.md
├── GIT_PUSH_GUIDE.md
├── SUBMISSION_READY.txt
├── models/
│   ├── resnet.py
│   └── svm_classifier.py
├── notebooks/
│   ├── Q1a_Submission_Colab.ipynb
│   └── Assignment1_2.ipynb
└── results/
    ├── mnist_q1a_results_final.csv
    ├── fashion_q1a_results_final.csv
    ├── q1b_svm_results.csv
    ├── mnist_resnet18_bs16_SGD_lr0.001_best.pth
    ├── fashion_resnet18_bs16_Adam_lr0.0001_best.pth
    ├── mnist_training_curves.png
    ├── fashion_training_curves.png
    └── combined_training_curves.png
```

---

## After Upload - Verify

1. **Check Repository:**
   - https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028/tree/Assignment-1
   - Ensure all 19 files are present

2. **Test Colab Link:**
   - https://colab.research.google.com/drive/1aEUrsukzyS6WkJQORQW1TazpADNcQgTv?usp=sharing
   - Should open your notebook

3. **GitHub Pages (Optional):**
   - Repository Settings → Pages
   - Source: Deploy from branch
   - Branch: Assignment-1, folder: / (root)
   - Save
   - Visit: https://kingkenche.github.io/MLOps-Shivam_Madhav_Kenche-M25CSA028/

---

## 🚀 Recommended: Method 2 (Web Upload)

**Easiest and fastest** - No additional software needed!

1. Open: https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028
2. Create "Assignment-1" branch
3. Upload all files from `G:\Ass-1\GitHub_Submission\Assignment1`
4. Commit changes

**Done!** ✅
