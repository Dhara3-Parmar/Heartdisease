# HeartCare: Final Project Implementation Documentation

## 1. Executive Summary & System Architecture
HeartCare is an end-to-end clinical decision support and risk prediction web application built on the UCI Cleveland Heart Disease dataset (`heart.csv`, 302 unique patients, 13 clinical features). The project combines a 100-tree `RandomForestClassifier` with a responsive multi-page Streamlit architecture, zero-dependency SVG/CSS visualizations, exact Shapley value interpretability, confidence-calibrated natural language explanations, statistical missing-value imputation, and quantized low-latency inference.

## 2. Summary of Methodologies
                              Module / Stage                                                                                                       Core Methodology     Implementation File
        1. Custom Responsive UI & WCAG Theme                         HTML/SVG/CSS without built-in layout libs; WCAG AA contrast >= 4.5:1 verified programmatically theme.py, components.py
2. Dynamic Input Validation & Error Recovery Data-driven rules from heart.csv; regex quick-entry parser; unit conversion (mmol/L -> mg/dL); 1-click fix suggestions input.py, validation.py
               3. Custom Thread-Safe Caching                                  LRU eviction + TTL expiration + Single-Flight request coalescing for concurrent users                cache.py
     4. Manual Tree Probability & Confidence      Per-tree probability distribution across 100 trees, vote spread (std), and interactive threshold + what-if curves           confidence.py
         5. Exact Shapley (SHAP) Attribution   Exact 2^13 = 8,192 coalition evaluation against reference background patients, verified against permutation sampling          shap_values.py
   6. Confidence-Calibrated NLG Explanations        Template-based natural language generation with 4 confidence tiers, hedging, and minimum-change counterfactuals         explain_text.py
     7. API Latency Profiling & Quantization   Stage-wise perf_counter profiling; 2D int16/int8/float32 quantized tree arrays eliminating Python loops & DataFrames              latency.py
     8. Statistical Missing Value Imputation      Detection of UCI anomaly codes (ca=4, thal=0) + 15% MCAR benchmark across KNN, MICE, and Random Forest Imputation           imputation.py
     9. Usability Heuristics & Feedback Loop                 Nielsen heuristic evaluation (P1-P10 fixed in v2), CSV-injection-safe feedback logger, and SUS scoring             feedback.py

## 3. Experimental Results
          Experiment / Benchmark                                                   Test Scope                                                                            Verified Result
  Input Validation & Auto-Repair                                 3,000 corrupted test records              100% error detection; 100% valid after applying suggested fix; ~0.08 ms/input
     Concurrent Prediction Cache      60 concurrent requests (12 threads, 10 unique patients)                     Model runs reduced from 60 to 10 (83.3% hit/coalesce rate); ~6x faster
    Manual Confidence vs Sklearn                               302 unique patients, 100 trees                          Max probability difference = 0.0; 100% agreement at 50% threshold
     Exact SHAP vs Sampling SHAP                          30 patients (8,192 coalitions each)           Additivity error = 2.22e-16; correlation with Monte-Carlo sampling = 0.995-0.997
           NLG Fact Verification      120 generated explanations (60 patients x 2 thresholds)                         0 headline errors, 0 direction errors, 0 counterfactual mismatches
        WCAG Accessibility Audit                         16 foreground/background token pairs                         16 / 16 pairs pass WCAG AA (>= 4.5:1 text, >= 3.0:1 UI components)
Model Quantization & API Latency                100-tree RandomForest -> int16/float32 arrays ~56% tree memory reduction; 0.00e+00 prediction drift; >10x faster end-to-end API response
        Missing Value Imputation 15% missingness across 7 clinical features + UCI ca=4/thal=0           Column-wise best method selected via RMSE, Accuracy, and KS-test p-value (>0.05)

## 4. Deployment Challenges & Solutions
                                          Challenge                                                                                                    Root Cause                                                                             Engineering Solution
                C1. Single-Row Inference & SHAP Lag                    Calling sklearn predict_proba on pandas DataFrames inside loops took ~650 ms per SHAP call Packed all 100 trees into contiguous int16/float32 2D arrays (QuantizedForest) + PredictionCache
C2. Streamlit Multi-Page State & Link Compatibility Switching pages resets widget keys and st.page_link raises KeyError('url_pathname') on newer Streamlit builds            Implemented keep_inputs() state persistence and safe nav_link() fallback in common.py
             C3. Hidden Missing Values in heart.csv                  Raw dataset had 0 explicit NaNs, but contained undocumented UCI missing codes (ca=4, thal=0)       Built detect_hidden_missing() and replaced anomalous codes using column-optimal imputation
    C4. Uncalibrated User Trust in Borderline Cases                                                 Users treat 51% and 95% risk predictions with equal certainty   Added 4-tier confidence badges, tree disagreement histograms, and hedged NLG + counterfactuals
             C5. Color-Blind & Mobile Accessibility                                     Red/green bars alone fail color-vision deficiency and small 360px screens Added upward/downward icons + text badges, >=4.5:1 contrast tokens, and responsive SVG viewBoxes

## 5. Actionable Recommendations for Future Improvements
                                Recommendation        Priority                                                                                                      Actionable Next Step
R1. Probability Calibration (Platt / Isotonic)   High Priority        Calibrate Random Forest probabilities using CalibratedClassifierCV and Brier score evaluation on external cohorts.
          R2. Multi-Center External Validation   High Priority Validate model generalization across full UCI (Hungarian, Switzerland, VA Long Beach) datasets beyond 302 Cleveland rows.
  R3. FastAPI Inference Service + ONNX Runtime Medium Priority       Expose QuantizedForest via an async FastAPI endpoint so mobile apps and EHR systems can query predictions in <2 ms.
           R4. Automated Data Drift Monitoring Medium Priority    Track KS-statistics on incoming patient batches against training distributions and trigger alerts when p-value < 0.05.
          R5. Clinical Codebook & Localization    Low Priority                              Verify exact clinical definitions of categorical codes (cp, thal, slope) with cardiologists.