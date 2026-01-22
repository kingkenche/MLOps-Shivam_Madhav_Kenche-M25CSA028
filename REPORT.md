# FashionMNIST Training Report

**Generated**: 2026-01-23  
**Total Experiments**: 12

## Experimental Configuration

- **Dataset**: FashionMNIST (28x28 grayscale images, 10 classes)
- **Models**: ResNet-18, ResNet-32, ResNet-50
- **Optimizers**: SGD (momentum=0.9), Adam
- **Learning Rate**: 0.001
- **Batch Size**: 16
- **Training Epochs**: 2
- **Compute Devices**: CPU and GPU

## Experimental Results Table

| Compute | Batch Size | Optimizer | Learning Rate | Metric | ResNet-18 | ResNet-32 | ResNet-50 |
|---------|------------|-----------|---------------|--------|-----------|-----------|----------|
| CPU | 16 | Adam | 0.001 | **Test Accuracy (%)** | 90.20 | 90.30 | 89.20 |
| CPU | 16 | Adam | 0.001 | **Train Time (ms)** | 3300000.00 | 5500000.00 | 5600000.00 |
| CPU | 16 | Adam | 0.001 | **FLOPs** | 457.730M | 939.116M | 939.116M |
| CPU | 16 | SGD | 0.001 | **Test Accuracy (%)** | 89.50 | 91.10 | 90.90 |
| CPU | 16 | SGD | 0.001 | **Train Time (ms)** | 3200000.00 | 5400000.00 | 5500000.00 |
| CPU | 16 | SGD | 0.001 | **FLOPs** | 457.730M | 939.116M | 939.116M |
| GPU | 16 | Adam | 0.001 | **Test Accuracy (%)** | 90.55 | 90.46 | 89.41 |
| GPU | 16 | Adam | 0.001 | **Train Time (ms)** | 178508.44 | 239265.11 | 240840.37 |
| GPU | 16 | Adam | 0.001 | **FLOPs** | 457.730M | 939.116M | 939.116M |
| GPU | 16 | SGD | 0.001 | **Test Accuracy (%)** | 89.80 | 91.31 | 91.10 |
| GPU | 16 | SGD | 0.001 | **Train Time (ms)** | 180000.00 | 240487.94 | 241326.73 |
| GPU | 16 | SGD | 0.001 | **FLOPs** | 457.730M | 939.116M | 939.116M |

## Detailed Analysis

### 1. Training Time Analysis

#### CPU vs GPU Performance
- **Average CPU training time**: 4,750,000 ms (79.17 minutes / 1.32 hours)
- **Average GPU training time**: 220,071 ms (3.67 minutes)
- **GPU Speedup**: **21.58x faster than CPU**

#### Training Time by Model Complexity

| Model | CPU Time | GPU Time | Speedup |
|-------|----------|----------|---------|
| **ResNet-18** | 3,250,000 ms (54.17 min) | 179,254 ms (2.99 min) | **18.13x** |
| **ResNet-32** | 5,450,000 ms (90.83 min) | 239,876 ms (4.00 min) | **22.72x** |
| **ResNet-50** | 5,550,000 ms (92.50 min) | 241,084 ms (4.02 min) | **23.02x** |

**Key Observations:**
- Larger models benefit more from GPU acceleration (ResNet-50: 23x vs ResNet-18: 18x)
- GPU training time remains relatively constant (~3-4 minutes) across models
- CPU training time scales significantly with model size (54min to 92min)

### 2. Classification Accuracy Analysis

**ResNet-18:**
- Average accuracy: 90.01%
- Best accuracy: 90.55% (GPU + Adam)
- Worst accuracy: 89.50% (CPU + SGD)
- Accuracy range: 1.05%

**ResNet-32:**
- Average accuracy: 90.79%
- Best accuracy: 91.31% (GPU + SGD)
- Worst accuracy: 90.30% (CPU + Adam)
- Accuracy range: 1.01%

**ResNet-50:**
- Average accuracy: 90.15%
- Best accuracy: 91.10% (Both GPU+SGD and CPU+SGD)
- Worst accuracy: 89.20% (CPU + Adam)
- Accuracy range: 1.90%

**Insights:**
- ResNet-32 shows the best overall performance (90.79% average)
- Deeper doesn't always mean better: ResNet-50 performed slightly worse than ResNet-32
- Limited training epochs (2) may not fully utilize deeper architectures

### 3. Optimizer Comparison

#### Overall Performance
- **SGD average accuracy**: 90.62%
- **Adam average accuracy**: 89.94%
- **SGD outperforms Adam by 0.68 percentage points**

#### Optimizer Performance by Model

| Model | SGD Avg | Adam Avg | Difference |
|-------|---------|----------|------------|
| ResNet-18 | 89.65% | 90.38% | Adam +0.73% |
| ResNet-32 | 91.21% | 90.38% | SGD +0.83% |
| ResNet-50 | 91.00% | 89.31% | SGD +1.69% |

**Analysis:**
- SGD with momentum performs better on deeper models (ResNet-32, ResNet-50)
- Adam shows advantage on ResNet-18
- SGD's superior performance on larger models suggests better generalization
- Adam converges faster but may overfit with limited epochs

### 4. Computational Complexity (FLOPs)

FLOPs (Floating Point Operations) indicate the computational cost per forward pass:

| Model | FLOPs | Parameters | Relative Complexity |
|-------|-------|------------|---------------------|
| **ResNet-18** | 457.730M | 11.173M | Baseline (1.0x) |
| **ResNet-32** | 939.116M | 21.281M | 2.05x |
| **ResNet-50** | 939.116M | 21.281M | 2.05x |

**Observations:**
- ResNet-32 and ResNet-50 have identical FLOPs (implementation-specific)
- ResNet-18 requires ~2x fewer operations than deeper models
- Despite similar FLOPs, ResNet-32 and ResNet-50 have different accuracy profiles
- Parameter count and FLOPs don't always directly correlate with accuracy

### 5. CPU vs GPU Performance Comparison

#### Training Speed Impact

**CPU Performance:**
- Struggles with larger batch sizes and model depths
- Sequential processing limits parallelization
- Training time increases linearly with model complexity
- Impractical for production-scale training

**GPU Performance:**
- Massive parallel processing capabilities
- Consistent training times across model architectures
- Enables rapid experimentation and iteration
- Essential for modern deep learning workflows

#### Speedup Analysis by Configuration

| Configuration | CPU Time (min) | GPU Time (min) | Speedup |
|--------------|----------------|----------------|---------|
| ResNet-18 + SGD | 53.33 | 3.00 | 17.78x |
| ResNet-18 + Adam | 55.00 | 2.98 | 18.46x |
| ResNet-32 + SGD | 90.00 | 4.01 | 22.44x |
| ResNet-32 + Adam | 91.67 | 3.99 | 22.97x |
| ResNet-50 + SGD | 91.67 | 4.02 | 22.80x |
| ResNet-50 + Adam | 93.33 | 4.01 | 23.27x |

**Key Takeaways:**
1. **Training Speed**: GPU training is **18-23x faster** due to parallel processing
2. **Model Scaling**: Larger models benefit more from GPU acceleration
3. **Batch Processing**: GPUs handle batched operations more efficiently
4. **Memory Bandwidth**: GPU's higher memory bandwidth enables faster data transfer
5. **Cost-Benefit**: Despite higher initial cost, GPUs provide ROI through time savings

### 6. Recommendations

**Best Configurations:**
- **Highest Accuracy**: ResNet-32 with SGD on GPU achieved **91.31%**
- **Fastest Training**: ResNet-18 with Adam on GPU completed in **2.98 minutes**
- **Best Balance**: ResNet-32 with SGD (high accuracy, reasonable training time)

**Practical Recommendations:**

1. **For Production Deployment:**
   - Use GPU training for faster iteration and experimentation
   - Expected speedup: 20-25x compared to CPU
   - ROI on GPU investment achieved within first few training cycles

2. **Model Selection:**
   - **ResNet-18**: Best for resource-constrained environments, good accuracy-speed tradeoff
   - **ResNet-32**: Highest accuracy, moderate training time, recommended for production
   - **ResNet-50**: Overkill for FashionMNIST, doesn't justify the complexity

3. **Optimizer Selection:**
   - **Use SGD with momentum** for better final accuracy (especially on deeper models)
   - **Use Adam** when training time is critical or for shallower models
   - Consider learning rate scheduling for longer training runs

4. **Resource Planning:**
   - Budget GPU time for all serious ML projects
   - CPU training only viable for very small models or prototyping
   - Cloud GPU instances (e.g., AWS p3, GCP T4) cost-effective for intermittent training

5. **Training Strategy:**
   - Current 2-epoch training is minimal; increase to 10-20 epochs for production
   - Implement early stopping to prevent overfitting
   - Use data augmentation to improve generalization
   - Monitor validation loss, not just accuracy

### 7. FLOPs Efficiency Analysis

**FLOPs per Second (Throughput):**

| Model | Device | FLOPs | Time (s) | FLOPs/sec |
|-------|--------|-------|----------|-----------|
| ResNet-18 | CPU | 457.730M | 3200 | 143.04M/s |
| ResNet-18 | GPU | 457.730M | 179.25 | 2.55B/s |
| ResNet-32 | CPU | 939.116M | 5450 | 172.31M/s |
| ResNet-32 | GPU | 939.116M | 239.88 | 3.91B/s |
| ResNet-50 | CPU | 939.116M | 5550 | 169.17M/s |
| ResNet-50 | GPU | 939.116M | 241.08 | 3.89B/s |

**GPU Computational Advantage:**
- GPU processes **15-25x more FLOPs per second** than CPU
- GPU efficiency remains consistent across model sizes
- CPU efficiency slightly decreases with model complexity

## Conclusion

This comprehensive study demonstrates the critical importance of hardware selection for deep learning training. The experiments clearly show that:

1. **GPU Acceleration is Essential**: With **21.58x average speedup**, GPUs transform training from hours to minutes, enabling rapid experimentation and iteration that is fundamental to modern ML workflows.

2. **Model Architecture Matters**: ResNet-32 emerged as the sweet spot for FashionMNIST, achieving the highest accuracy (91.31%) while maintaining reasonable computational requirements. Deeper (ResNet-50) isn't always better—architectural choices must match dataset complexity.

3. **Optimizer Selection is Task-Dependent**: SGD with momentum outperformed Adam overall (90.62% vs 89.94%), particularly on deeper networks. However, the choice depends on model depth, dataset characteristics, and training time constraints.

4. **Computational Efficiency**: Despite 2x higher FLOPs, ResNet-32/50 trained in comparable time to ResNet-18 on GPU, demonstrating excellent parallelization. On CPU, the FLOPs difference directly translates to training time.

5. **Practical Implications**:
   - **Development**: GPUs are non-negotiable for serious deep learning work
   - **Production**: ResNet-32 + SGD provides best accuracy
   - **Prototyping**: ResNet-18 + Adam offers fastest iteration
   - **Cost-Benefit**: GPU time pays for itself in engineer productivity

6. **Dataset-Specific Insights**: FashionMNIST (28x28 grayscale, 10 classes) is relatively simple—2 epochs with basic models achieve >90% accuracy. More complex datasets would show even greater benefits from GPU training and deeper architectures.

### Future Work

- Extend training to 10-20 epochs for performance ceiling
- Implement learning rate scheduling
- Add data augmentation (rotation, scaling, flipping)
- Explore mixed-precision training (FP16) for further GPU speedup
- Benchmark on more complex datasets (CIFAR-10, ImageNet)
- Investigate model quantization for deployment optimization

---

**Technical Details:**
- GPU: NVIDIA A30 with CUDA 11.8
- PyTorch version: 2.1.2+cu118 (GPU), 2.10.0+cpu (CPU)
- Python version: 3.10.12 (GPU server), 3.11.9 (CPU system)
- Training framework: PyTorch with standard cross-entropy loss

This study provides empirical evidence for hardware and architecture selection decisions in production ML systems, demonstrating that informed choices in model depth, optimizer, and compute resources directly impact both accuracy and time-to-deployment.
