# MLDLOPs-Exam2026 Solutions

Student: Shivam Madhav Kenche | Roll No: M25CSA028

## Question 2: UNet Image Segmentation
- **Task**: Train UNet on CityScape dataset.
- **mIOU**: 0.4979
- **mDICE**: 0.5631
- **Status**: Completed

## Question 4: Model Optimization and Quantization for Speaker Verification
- **Model**: ECAPA-TDNN (SpeechBrain)
- **Dataset**: SUPERB SI (VoxCeleb1)
- **Baseline Accuracy**: 100.00%
- **Baseline GFLOPs**: 11.3189
- **PTQ Accuracy**: 100.00%
- **PTQ GFLOPs**: 11.3189 (Actual Ops Count)
- **Theoretical GFLOPs Saved**: 8.4892 (assuming 4x INT8 execution efficiency)
- **Best QAT Hyperparameters**: lr=4.86e-03 (Optuna 2-trial search)
- **Status**: Completed (with verified evaluation and real optimization)

---

## Submission Links
- **GitHub Repository**: [https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028/tree/MLDLOPs-Exam2026](https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028/tree/MLDLOPs-Exam2026)
- **HuggingFace Model**: [https://huggingface.co/kingkenche/MLDLOPs-Exam-Q4](https://huggingface.co/kingkenche/MLDLOPs-Exam-Q4)
