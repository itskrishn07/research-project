# Research Design
## Reliability-Aware Evaluation of Post-Hoc Explainable AI Methods for Deep Learning-Based Medical Image Classification

**Project type:** Final-year B.Tech AI/ML research project  
**Primary domain:** Explainable AI (XAI) + Medical Image Classification  
**Status:** Phase 0 — Research Design Lock  
**Version:** 1.0

---

## 1. Project Title

### Official title
**Reliability-Aware Evaluation of Post-Hoc Explainable AI Methods for Deep Learning-Based Medical Image Classification**

### Working short title
**Reliable XAI for Medical Image Classification**

---

## 2. Research Problem

Deep learning models can perform medical image classification effectively, but their decision-making process is often difficult to interpret. Explainable AI (XAI) methods such as Grad-CAM, Integrated Gradients, and SHAP can generate explanations indicating which image regions or features contributed to a prediction.

However, a visually convincing explanation is not necessarily a reliable explanation. Different XAI methods can produce different explanations for the same model prediction, and an explanation may highlight a region that is not actually responsible for the model's decision.

Therefore, this research will investigate the **reliability of post-hoc XAI explanations** for deep-learning-based medical image classification using quantitative evaluation rather than visual inspection alone.

---

## 3. Core Research Question

> **Which post-hoc XAI methods provide reliable and faithful explanations for deep-learning-based medical image classification, and under what conditions does explanation reliability change?**

---

## 4. Research Objectives

### Objective 1
Develop and validate a deep-learning-based medical image classification model using a chest X-ray dataset.

### Objective 2
Generate post-hoc explanations using:
- Grad-CAM
- Integrated Gradients
- SHAP

### Objective 3
Quantitatively evaluate explanations using multiple dimensions:
- Localization
- Faithfulness
- Stability
- Computational efficiency

### Objective 4
Investigate whether model confidence is associated with explanation reliability.

### Objective 5
Investigate whether pathology characteristics, such as lesion/pathology size and complexity, affect explanation quality.

### Objective 6
Investigate whether agreement between different XAI methods is associated with explanation reliability.

### Objective 7
Develop a reproducible experimental framework and produce findings suitable for a research paper.

---

## 5. Research Gap

Existing research has evaluated individual XAI methods and has also compared multiple explanation methods using metrics such as localization, faithfulness, stability, and computational cost. Medical XAI research also contains studies using expert annotations to assess whether explanations correspond to clinically relevant regions.

The proposed research does **not** claim that comparing Grad-CAM, Integrated Gradients, and SHAP is itself novel.

Instead, the intended research gap is to conduct a **systematic, reliability-oriented evaluation** in a controlled medical-imaging setting, examining multiple dimensions of explanation quality together and studying how explanation reliability changes with factors such as:

1. XAI method
2. Model confidence
3. Pathology characteristics
4. Input perturbations
5. Computational cost
6. Agreement between different explanation methods

The exact novelty claim will be finalized after the focused literature review and exact-match comparison with existing studies.

---

## 6. Research Questions

### RQ1 — Localization
How accurately do Grad-CAM, Integrated Gradients, and SHAP identify pathology-related regions compared with expert annotations?

### RQ2 — Faithfulness
How faithfully do the explanations represent the features or regions used by the trained model for its prediction?

### RQ3 — Stability
How stable are the explanations when the input image undergoes small controlled perturbations?

### RQ4 — Computational efficiency
What trade-off exists between explanation quality and computational cost for the three XAI methods?

### RQ5 — Confidence dependence
Does explanation reliability change with the model's prediction confidence?

### RQ6 — Pathology dependence
Does explanation reliability change according to pathology characteristics such as size, location, or complexity?

### RQ7 — Cross-method agreement
Is higher agreement between different XAI methods associated with higher explanation reliability?

---

## 7. Research Hypotheses

These are hypotheses to be experimentally tested, not predetermined conclusions.

### H1
The three XAI methods will show measurable differences in localization performance.

### H2
The three XAI methods will show measurable differences in perturbation-based faithfulness.

### H3
Explanation stability will vary across XAI methods and input perturbation types.

### H4
Explanation reliability will vary across different model-confidence levels.

### H5
Explanation quality will vary according to pathology characteristics.

### H6
Higher cross-method explanation agreement may be associated with higher explanation reliability.

---

## 8. Dataset Strategy

### Primary training dataset
**CheXpert**

Use:
- Chest X-ray images
- Available disease labels
- Training/validation data as appropriate for the selected task

### Expert-annotation evaluation dataset
**CheXlocalize**

Use:
- Expert/radiologist localization annotations
- Ground-truth regions for evaluating explanation localization

### Why this combination?

The research requires both:
1. A sufficiently large medical image dataset for training a deep-learning classifier.
2. Expert localization annotations for quantitatively evaluating whether XAI explanations correspond to medically relevant image regions.

CheXpert + CheXlocalize provides a suitable experimental foundation for this purpose.

---

## 9. Medical Classification Task

### Initial task
Chest X-ray classification.

### Initial model
**DenseNet-121**

The initial implementation should use one clearly defined target pathology/task so that the experimental pipeline can be developed and validated before expanding the scope.

The final pathology/task will be selected after verifying:
- Label quality
- Availability of corresponding expert annotations
- Sample size
- Class balance
- Relevance to the research questions

Do not expand to many pathologies until the complete pipeline works for the initial task.

---

## 10. Baseline Deep Learning Model

### Primary model
**DenseNet-121**

### Initial approach
Transfer learning followed by fine-tuning on the selected chest X-ray classification task.

### Baseline evaluation

The classifier should be evaluated before XAI experiments.

Potential metrics:
- AUROC
- AUPRC
- Sensitivity
- Specificity
- F1-score, where appropriate

The final metric set will depend on the selected classification formulation and class distribution.

### Important principle

XAI experiments will be conducted on a validated model. Model performance itself is not the primary research contribution; it provides the model whose explanations will be studied.

---

## 11. XAI Methods

### Method 1 — Grad-CAM
Gradient-based class activation mapping for identifying important spatial regions in CNN feature representations.

### Method 2 — Integrated Gradients
Gradient-based attribution method that estimates feature contributions along a path from a baseline input to the actual input.

### Method 3 — SHAP
Feature attribution framework based on estimating contributions of features to a model prediction.

### Comparison principle

For fair comparison, the methods should be evaluated using:
- The same trained model
- The same preprocessing pipeline
- The same test images
- The same prediction target
- The same evaluation subset wherever possible

---

## 12. Explanation Evaluation Framework

The primary evaluation framework will have four dimensions.

### 12.1 Localization

**Research question:**
Does the explanation highlight the region corresponding to the expert-annotated pathology?

Potential metrics:
- Intersection over Union (IoU)
- Mean IoU (mIoU), where appropriate
- Hit Rate

Ground truth:
- Expert/radiologist annotations from CheXlocalize

---

### 12.2 Faithfulness

**Research question:**
Does changing or removing the region identified as important by an explanation meaningfully affect the model's prediction?

Proposed procedure:

1. Generate an XAI explanation.
2. Identify the most important regions/pixels.
3. Perturb, mask, or remove selected important regions.
4. Run the model again.
5. Measure the change in prediction/confidence.
6. Compare the results across XAI methods.

Potential evaluation approaches:
- Confidence drop
- Deletion curves
- AOPC or related perturbation-based metrics

The exact faithfulness metric will be finalized after implementation and literature review.

---

### 12.3 Stability

**Research question:**
Does a small, controlled change in the input cause an unnecessarily large change in the explanation?

Potential perturbations:
- Small additive noise
- Brightness changes
- Contrast changes
- Small controlled transformations

Procedure:

1. Generate explanation for original image.
2. Apply controlled perturbation.
3. Generate explanation again.
4. Compare the two explanations.
5. Repeat across images and XAI methods.

Potential stability measures will be selected based on established literature.

---

### 12.4 Computational Efficiency

Measure, where practical:
- Explanation generation time
- Number of model evaluations
- Memory/computational overhead

This dimension will be used to study the trade-off between explanation quality and computational cost.

---

## 13. Additional Reliability Analysis

### 13.1 Model Confidence

For each prediction, record:
- Predicted class
- Prediction confidence/probability
- XAI method
- Localization result
- Faithfulness result
- Stability result

Analyze whether explanation quality changes across confidence levels.

---

### 13.2 Pathology Characteristics

Where annotations/data permit, analyze factors such as:
- Pathology size
- Number of annotated regions
- Spatial complexity
- Potentially location-related characteristics

Goal:

> Determine whether certain pathology characteristics make reliable explanation more difficult.

---

### 13.3 Cross-Method Agreement

Compare the spatial/feature-level agreement between:
- Grad-CAM
- Integrated Gradients
- SHAP

Then investigate whether explanation agreement is associated with independently measured explanation reliability.

This is an exploratory research hypothesis and should not be presented as a guaranteed novel contribution before literature verification.

---

## 14. Experimental Design

### Controlled variables

Keep the following consistent during primary comparison:
- Dataset/task
- Model checkpoint
- Input preprocessing
- Test image subset
- Target class
- Evaluation procedure

### Independent variables
- XAI method
- Confidence level
- Input perturbation
- Pathology characteristics

### Dependent variables
- Localization metrics
- Faithfulness metrics
- Stability metrics
- Computational cost

---

## 15. Experimental Pipeline

```text
CheXpert
   |
   v
Data Preprocessing
   |
   v
DenseNet-121 Training
   |
   v
Baseline Model Evaluation
   |
   v
Test Image
   |
   +------------------+------------------+
   |                  |                  |
   v                  v                  v
Grad-CAM              IG                SHAP
   |                  |                  |
   +------------------+------------------+
                      |
                      v
              Explanation Maps
                      |
          +-----------+-----------+-----------+
          |           |           |           |
          v           v           v           v
     Localization Faithfulness Stability  Efficiency
          |           |           |           |
          +-----------+-----------+-----------+
                      |
                      v
             Confidence Analysis
                      |
                      v
           Pathology Analysis
                      |
                      v
           Cross-Method Agreement
                      |
                      v
             Statistical Analysis
                      |
                      v
             Research Findings
                      |
                      v
                Research Paper
```

---

## 16. Statistical Analysis Plan

The study should report more than simple average scores.

Depending on the final experimental design and data distribution, analysis may include:
- Descriptive statistics
- Confidence intervals, including bootstrap intervals where appropriate
- Paired comparisons between XAI methods
- Wilcoxon signed-rank test or another appropriate paired test
- Effect sizes
- Correlation analysis
- Multiple-comparison correction where necessary

The exact statistical tests will be finalized after the dataset, number of samples, and paired/unpaired structure are established.

---

## 17. Reproducibility Requirements

Every major experiment should record:
- Dataset/version
- Model architecture
- Model checkpoint
- Random seed
- Preprocessing configuration
- XAI method
- XAI parameters
- Number of evaluated images
- Evaluation metrics
- Runtime
- Experiment ID

All important results should be saved as structured CSV/JSON files along with figures.

---

## 18. Expected Outputs

The completed project should produce:

1. A validated chest X-ray classification model.
2. Grad-CAM explanations.
3. Integrated Gradients explanations.
4. SHAP explanations.
5. Automated localization evaluation.
6. Faithfulness evaluation.
7. Stability evaluation.
8. Computational efficiency measurements.
9. Confidence-dependent analysis.
10. Pathology-dependent analysis.
11. Cross-method agreement analysis, if feasible.
12. Statistical comparison of XAI methods.
13. Visualization/dashboard for demonstration.
14. Reproducible experiment results.
15. Research paper.

---

## 19. Expected Research Contribution

The intended contribution is an empirical and reproducible framework for studying the reliability of post-hoc XAI explanations in medical image classification.

The research will attempt to provide evidence about:
- How different XAI methods behave under different evaluation criteria.
- Whether good localization implies good faithfulness.
- Whether explanation stability differs across methods.
- How computational cost relates to explanation quality.
- Whether model confidence is associated with explanation reliability.
- Whether pathology characteristics affect explanation reliability.
- Whether cross-method agreement can provide useful information about explanation reliability.

The contribution should be stated as an empirical evaluation and analysis unless the experiments justify a stronger methodological contribution.

---

## 20. Scope Boundaries

The project will **not** initially attempt to:

- Build a clinically deployable diagnostic system.
- Claim that the model can replace a radiologist.
- Develop a new XAI algorithm from the beginning.
- Evaluate a very large number of diseases simultaneously.
- Compare a large number of model architectures before the primary experiment is complete.
- Claim that one XAI method is universally best.
- Claim clinical validity based only on technical explanation metrics.

The project is primarily a **research study of explanation reliability**, not a medical diagnostic product.

---

## 21. Research Limitations to Track

Potential limitations include:
- Expert annotation availability and quality.
- Dataset-specific biases.
- Dependence on the selected model architecture.
- Dependence on the selected pathology/task.
- Limitations of existing XAI evaluation metrics.
- Computational cost of SHAP and other expensive explainers.
- Technical reliability not being equivalent to human or clinical trust.
- Results obtained on chest X-rays may not automatically generalize to other medical imaging modalities.

These limitations should be discussed honestly in the final paper.

---

## 22. Phase-Wise Execution Plan

### Phase 0 — Research Design
Status: **Current phase**

Lock:
- Research question
- Objectives
- Dataset strategy
- Model
- XAI methods
- Evaluation dimensions
- Experimental hypotheses

### Phase 1 — Dataset Setup
- Obtain/access CheXpert.
- Obtain/access CheXlocalize annotations.
- Verify dataset structure.
- Select initial classification task.
- Create reproducible data splits.

### Phase 2 — Preprocessing
- Image loading
- Resizing
- Normalization
- Label preparation
- Data augmentation where appropriate
- Data-loader validation

### Phase 3 — Baseline Model
- Implement DenseNet-121
- Transfer learning
- Training
- Validation
- Checkpointing

### Phase 4 — Baseline Evaluation
- AUROC/AUPRC and appropriate classification metrics
- Error analysis
- Select validated checkpoint

### Phase 5 — Grad-CAM
- Generate explanations
- Save heatmaps
- Validate visualization correctness

### Phase 6 — Integrated Gradients
- Generate explanations
- Validate attribution maps

### Phase 7 — SHAP
- Generate explanations
- Validate attribution maps

### Phase 8 — Localization
- Align XAI maps with expert annotations
- Threshold/normalize using a predefined protocol
- Calculate localization metrics

### Phase 9 — Faithfulness
- Define perturbation protocol
- Perform deletion/masking experiments
- Calculate faithfulness metrics

### Phase 10 — Stability
- Define perturbation types
- Generate repeated explanations
- Calculate stability metrics

### Phase 11 — Efficiency
- Record runtime
- Count model evaluations
- Measure relevant computational overhead

### Phase 12 — Additional Analysis
- Confidence analysis
- Pathology characteristics
- Cross-method agreement

### Phase 13 — Statistical Analysis
- Compare methods
- Calculate confidence intervals/effect sizes
- Test relevant hypotheses

### Phase 14 — Visualization/Dashboard
- Build research-result dashboard
- Display individual explanations
- Display aggregate metrics

### Phase 15 — Research Paper
- Introduction
- Literature review
- Gap
- Methodology
- Experiments
- Results
- Discussion
- Limitations
- Conclusion

---

## 23. Definition of Success

The project will be considered successful if it can:

1. Train and validate a reproducible medical image classifier.
2. Generate valid explanations using all three selected XAI methods.
3. Quantitatively evaluate those explanations.
4. Produce statistically interpretable comparative results.
5. Answer the defined research questions with experimental evidence.
6. Clearly document limitations and avoid unsupported claims.
7. Produce a reproducible research artifact and a paper based on the findings.

---

## 24. Important Research Principle

The project must remain **result-driven rather than conclusion-driven**.

Do not assume beforehand that:
- Grad-CAM will be best.
- Integrated Gradients will be best.
- SHAP will be best.
- High model confidence means high explanation reliability.
- High cross-method agreement means high reliability.

These are questions/hypotheses to test experimentally.

The final conclusions must be based on the collected evidence.

---

## 25. Immediate Next Step

After Phase 0 approval, do **not** immediately implement all XAI methods.

The next execution step is:

### Phase 1 — Dataset Setup

First verify:
1. CheXpert access.
2. CheXlocalize access.
3. Dataset directory structure.
4. Selected pathology/task.
5. Number of usable samples.
6. Availability and format of expert annotations.
7. A reproducible train/validation/test strategy.

Only after this verification should model implementation begin.

---

## 26. Mentor Approval Statement

> **We propose to study the reliability of post-hoc explainable AI methods for deep-learning-based medical image classification. Using a chest X-ray classification model and expert-annotated localization data, we will compare Grad-CAM, Integrated Gradients, and SHAP using localization, faithfulness, stability, and computational-efficiency measures. We will further investigate the effect of model confidence and pathology characteristics on explanation reliability and explore cross-method agreement as an additional research hypothesis. The objective is to conduct a reproducible empirical study that produces research findings suitable for a final-year project and research publication.**

---

## 27. Final Locked Research Definition

**Domain:** Explainable AI for Medical Imaging

**Problem:** Reliability of post-hoc explanations

**Dataset strategy:** CheXpert + CheXlocalize

**Baseline model:** DenseNet-121

**XAI methods:** Grad-CAM + Integrated Gradients + SHAP

**Primary evaluation dimensions:**
1. Localization
2. Faithfulness
3. Stability
4. Computational efficiency

**Additional analyses:**
1. Model confidence
2. Pathology characteristics
3. Cross-method agreement

**Primary output:** Reproducible XAI reliability evaluation framework

**Research outcome:** Empirical findings + final-year implementation + research paper

**Novelty status:** To be finalized after focused literature review; no unsupported claim of absolute novelty will be made.
