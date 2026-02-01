# 📋 Submission Checklist

## Before You Submit

Use this checklist to ensure you have everything ready for submission.

### ✅ Code Files

- [ ] All source code files are committed to Git
- [ ] `.gitignore` is properly configured
- [ ] No model weights (`.pth`, `.pt` files) in the repository
- [ ] Code is well-commented and readable

### ✅ Training Completed

- [ ] Model trained for 25-30 epochs
- [ ] Best model saved locally (not pushed to Git)
- [ ] Training completed without errors

### ✅ WandB Dashboard

- [ ] All training runs are logged to WandB
- [ ] Gradient flow visualizations are visible
- [ ] Weight update visualizations are visible
- [ ] Training/validation curves are complete
- [ ] Model complexity metrics (FLOPs, parameters) are logged
- [ ] WandB dashboard is accessible (set to public or shared)

### ✅ GitHub Repository

- [ ] Repository created with naming: `name_rollnumber_lab2_worksheet`
- [ ] All code files pushed to GitHub
- [ ] Branch created with proper naming convention
- [ ] README.md updated with your findings
- [ ] Model weights are NOT pushed (verify with `git status`)

### ✅ Report Document

- [ ] Report includes model architecture details
- [ ] FLOPs analysis results documented
- [ ] Training results (accuracy, loss) included
- [ ] Gradient flow observations documented
- [ ] Weight update observations documented
- [ ] Screenshots/links to WandB visualizations
- [ ] Conclusions and key findings

### ✅ Classroom Submission

Prepare the following for classroom submission:

1. **Report**: PDF or document with all findings
2. **GitHub Link**: URL to your branch
   ```
   https://github.com/<username>/<repo>/tree/name_rollnumber_lab2_worksheet
   ```
3. **WandB Link**: URL to your WandB project
   ```
   https://wandb.ai/<username>/<project-name>
   ```

## Submission Template

Copy this template for your classroom submission:

```
Student Name: [Your Name]
Roll Number: [Your Roll Number]
Assignment: Lab 2 Worksheet

1. Report: [Attach PDF/Document]

2. GitHub Branch Link:
   https://github.com/[username]/[repo]/tree/[name_rollnumber_lab2_worksheet]

3. WandB Dashboard Link:
   https://wandb.ai/[username]/[project-name]

Key Results:
- Model: ResNet-18
- Best Validation Accuracy: [X.XX]%
- Total Epochs: [XX]
- Total FLOPs: [X.XX]M/G
- Total Parameters: [X.XX]M
```

## Final Verification Commands

Run these commands before submitting:

```bash
# 1. Check Git status (should show no .pth or .pt files)
git status

# 2. Verify .gitignore is working
git ls-files | grep -E '\.(pth|pt)$'
# (Should return nothing)

# 3. List all files to be pushed
git ls-files

# 4. Check your branch name
git branch
```

## Common Issues Before Submission

### ❌ Model weights in Git
**Fix**: 
```bash
git rm --cached models/*.pth checkpoints/*.pth
git commit -m "Remove model weights"
```

### ❌ WandB dashboard not accessible
**Fix**: Make your WandB project public:
1. Go to your WandB project
2. Settings → Visibility → Public

### ❌ Missing visualizations in WandB
**Fix**: Re-run training with proper WandB initialization

## Ready to Submit?

Double-check:
1. ✅ GitHub repository is accessible
2. ✅ WandB dashboard is accessible
3. ✅ Report is complete with all observations
4. ✅ No model weights in Git repository

---

**Good luck with your submission! 🎉**
