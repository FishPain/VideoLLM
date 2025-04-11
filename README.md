# 🧠 VideoLLM – AISG Challenge Submission

> ⚠️ **Note:** 83 questions could not be answered due to missing videos.  
> ✅ The model is now instructed to make **intelligent guesses** for these cases to improve overall completeness.

---

## ⚙️ Development Environment

| Resource        | Specification         |
|----------------|------------------------|
| GPU            | NVIDIA H100 SXM (1x)   |
| GPU Memory     | 80 GB                  |
| Disk Space     | 100 GB                 |

---

## 📊 Model Evaluation – Correctness & Robustness

Due to compute limitations, initial evaluations were conducted using **Qwen2.5-VL 7B**.  
Empirical trends suggest the **Qwen2.5-VL 32B** model offers a **~10–15% improvement** in both correctness and robustness.

| Configuration                                                          | Correctness (%) | Robustness (%) |
|------------------------------------------------------------------------|-----------------|----------------|
| Qwen2.5-VL 7B + Prompt Tuning                                          | 27.47           | 5.5            |
| Qwen2.5-VL 32B + Prompt Tuning                                         | 40.33           | 23.0           |
| Qwen2.5-VL 7B + Prompt Tuning + Smart Guessing for Missing Videos      | 32.87           | 5.8            |
| Qwen2.5-VL 7B + Base + Higher FPS (1 → 5)                              | 🔧 Not Tested    | 🔧 Not Tested   |
| Qwen2.5-VL 7B + Base + External Evaluator Integration                  | 🔧 Not Tested    | 🔧 Not Tested   |
| Qwen2.5-VL 7B + Base + Audio Captioning via Whisper                    | 🔧 Not Tested    | 🔧 Not Tested   |
| Qwen2.5-VL 7B + Base + RNN Adapter for Recursive Video Understanding   | 🔧 Not Tested    | 🔧 Not Tested   |

---

## 🔭 Future Work

- 🧠 Integrate **RNN-based adapters** for enhanced temporal modeling  
- 🔊 Incorporate **Whisper-generated audio transcripts** into prompts  
- 🛡️ Build fallback mechanisms for **incomplete or missing video data**  
- 🚀 Scale testing to include full evaluations on **Qwen2.5-VL 32B**  
- 🧪 Introduce a **separate evaluator model** for scoring candidate answers
